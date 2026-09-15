"""Seasonal PV + BESS modelling core.

The module is intentionally independent of Streamlit. It implements:
- solar sunrise/sunset and day/night classification;
- seasonal operating modes;
- PV self-consumption and PV charging;
- winter/transition grid charging when price arbitrage is profitable;
- separate tracking of PV-origin and grid-origin energy in the battery;
- 24 h autonomy sizing metrics;
- energy-balance validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo
import math
import numpy as np
import pandas as pd


SUMMER_MONTHS = {4, 5, 6, 7, 8, 9, 10}
TRANSITION_MONTHS = {3, 11}
WINTER_MONTHS = {12, 1, 2}


@dataclass(frozen=True)
class ModelConfig:
    latitude: float = 55.89
    longitude: float = 23.36
    timezone: str = "Europe/Vilnius"
    summer_months: tuple[int, ...] = (4, 5, 6, 7, 8, 9, 10)
    transition_months: tuple[int, ...] = (3, 11)
    winter_months: tuple[int, ...] = (12, 1, 2)
    low_price_quantile: float = 0.25
    high_price_quantile: float = 0.75
    degradation_cost_eur_kwh: float = 0.02
    minimum_arbitrage_margin_eur_kwh: float = 0.00


def _season(month: int, cfg: ModelConfig) -> str:
    if month in cfg.summer_months:
        return "summer"
    if month in cfg.transition_months:
        return "transition"
    return "winter"


def _solar_declination_rad(day_of_year: int) -> float:
    # Cooper approximation; sufficient for dispatch/day-night separation.
    return math.radians(23.45) * math.sin(
        2.0 * math.pi * (284 + day_of_year) / 365.0
    )


def _equation_of_time_minutes(day_of_year: int) -> float:
    # NOAA-style compact approximation.
    b = 2.0 * math.pi * (day_of_year - 81) / 364.0
    return 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)


def sunrise_sunset_hours(
    day: date | pd.Timestamp,
    latitude: float = 55.89,
    longitude: float = 23.36,
    timezone: str = "Europe/Vilnius",
) -> tuple[float, float, float]:
    """Return local-clock sunrise hour, sunset hour and daylight duration."""
    ts = pd.Timestamp(day).normalize()
    n = int(ts.dayofyear)
    lat = math.radians(latitude)
    dec = _solar_declination_rad(n)

    cos_h = -math.tan(lat) * math.tan(dec)
    cos_h = min(1.0, max(-1.0, cos_h))
    h = math.acos(cos_h)
    daylight_hours = 2.0 * math.degrees(h) / 15.0

    # Determine civil UTC offset at local noon, including DST.
    local_noon = datetime(ts.year, ts.month, ts.day, 12, 0, tzinfo=ZoneInfo(timezone))
    utc_offset_hours = local_noon.utcoffset().total_seconds() / 3600.0

    eot = _equation_of_time_minutes(n)
    time_correction = eot + 4.0 * longitude - 60.0 * utc_offset_hours
    solar_noon_minutes = 720.0 - time_correction
    half_day_minutes = daylight_hours * 30.0

    sunrise = (solar_noon_minutes - half_day_minutes) / 60.0
    sunset = (solar_noon_minutes + half_day_minutes) / 60.0
    return sunrise, sunset, daylight_hours


def build_solar_calendar(
    datetimes: pd.Series | pd.DatetimeIndex,
    cfg: ModelConfig = ModelConfig(),
) -> pd.DataFrame:
    dt = pd.to_datetime(pd.Series(datetimes), errors="coerce")
    out = pd.DataFrame({"datetime": dt})
    out = out.dropna(subset=["datetime"]).copy()
    out["date"] = out["datetime"].dt.normalize()

    unique_days = out["date"].drop_duplicates().sort_values()
    rows = []
    for d in unique_days:
        sunrise, sunset, day_length = sunrise_sunset_hours(
            d, cfg.latitude, cfg.longitude, cfg.timezone
        )
        rows.append(
            {
                "date": d,
                "sunrise_hour": sunrise,
                "sunset_hour": sunset,
                "day_length_h": day_length,
                "night_length_h": 24.0 - day_length,
                "season": _season(int(pd.Timestamp(d).month), cfg),
            }
        )

    cal = pd.DataFrame(rows)
    out = out.merge(cal, on="date", how="left")
    local_hour = (
        out["datetime"].dt.hour
        + out["datetime"].dt.minute / 60.0
        + out["datetime"].dt.second / 3600.0
    )
    out["is_day"] = (
        (local_hour >= out["sunrise_hour"])
        & (local_hour < out["sunset_hour"])
    )
    out["is_night"] = ~out["is_day"]
    return out


def create_dynamic_hourly_pv_profile(
    pv_daily: pd.DataFrame,
    cfg: ModelConfig = ModelConfig(),
) -> pd.DataFrame:
    """Distribute each day's specific PV yield over daylight hours.

    Daily PV energy is preserved exactly. Only the intra-day shape changes with
    sunrise/sunset through the year.
    """
    rows: list[dict] = []

    for _, row in pv_daily.iterrows():
        day = pd.Timestamp(row["date"]).normalize()
        daily_yield = float(row["yield_kwh_per_kw"])
        sunrise, sunset, _ = sunrise_sunset_hours(
            day, cfg.latitude, cfg.longitude, cfg.timezone
        )

        mids = np.arange(24, dtype=float) + 0.5
        weights = np.zeros(24, dtype=float)
        inside = (mids >= sunrise) & (mids < sunset)
        if inside.any():
            phase = (mids[inside] - sunrise) / max(sunset - sunrise, 1e-9) * np.pi
            weights[inside] = np.sin(phase)
        if weights.sum() <= 0:
            # Very defensive fallback; not expected at Lithuanian latitude.
            weights[12] = 1.0
        weights = weights / weights.sum()

        for hour in range(24):
            rows.append(
                {
                    "datetime": day + pd.Timedelta(hours=hour),
                    "pv_per_kw": daily_yield * float(weights[hour]),
                }
            )

    return pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)


def detect_timestep_hours(df: pd.DataFrame) -> float:
    ts = pd.to_datetime(df["datetime"], errors="coerce").sort_values()
    diffs = ts.diff().dropna()
    if len(diffs) == 0:
        return 1.0
    dt = diffs.median().total_seconds() / 3600.0
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Nepavyko nustatyti duomenų intervalo.")
    return float(dt)


def map_pv_to_load_dates(
    hourly_pv: pd.DataFrame,
    load_df: pd.DataFrame,
    cfg: ModelConfig = ModelConfig(),
) -> pd.DataFrame:
    """Map a reference calendar PV shape to actual-load calendar dates.

    For sub-hourly load, hourly PV energy is scaled to the load interval.
    """
    pv = hourly_pv.copy()
    load = load_df.copy()
    pv["datetime"] = pd.to_datetime(pv["datetime"])
    load["datetime"] = pd.to_datetime(load["datetime"])

    for frame in (pv, load):
        frame["month"] = frame["datetime"].dt.month
        frame["day"] = frame["datetime"].dt.day
        frame["hour"] = frame["datetime"].dt.hour

    profile = (
        pv.groupby(["month", "day", "hour"], as_index=False)["pv_per_kw"]
        .mean()
    )
    result = load.merge(profile, on=["month", "day", "hour"], how="left")
    result["pv_per_kw"] = result["pv_per_kw"].fillna(0.0)

    dt_hours = detect_timestep_hours(load)
    result["pv_per_kw"] = result["pv_per_kw"] * dt_hours

    result = result[["datetime", "load_kwh", "pv_per_kw"]].copy()
    solar = build_solar_calendar(result["datetime"], cfg)
    result = result.merge(
        solar[
            [
                "datetime", "season", "is_day", "is_night",
                "sunrise_hour", "sunset_hour", "day_length_h", "night_length_h",
            ]
        ],
        on="datetime",
        how="left",
    )
    return result.sort_values("datetime").reset_index(drop=True)


def add_price_signals(
    data: pd.DataFrame,
    charge_eff: float,
    discharge_eff: float,
    cfg: ModelConfig = ModelConfig(),
) -> pd.DataFrame:
    """Add daily cheap/expensive flags only where arbitrage is economically viable."""
    df = data.copy()
    if "buy_price_eur_kwh" not in df.columns:
        df["price_low_threshold"] = np.nan
        df["price_high_threshold"] = np.nan
        df["is_cheap_price"] = False
        df["is_expensive_price"] = False
        df["arbitrage_viable"] = False
        return df

    df["buy_price_eur_kwh"] = pd.to_numeric(df["buy_price_eur_kwh"], errors="coerce")
    df["date"] = pd.to_datetime(df["datetime"]).dt.normalize()

    daily = (
        df.groupby("date")["buy_price_eur_kwh"]
        .agg(
            price_low_threshold=lambda s: s.quantile(cfg.low_price_quantile),
            price_high_threshold=lambda s: s.quantile(cfg.high_price_quantile),
            price_min="min",
            price_max="max",
            price_std="std",
        )
        .reset_index()
    )

    rt_eff = max(charge_eff * discharge_eff, 1e-9)
    daily["required_high_price"] = (
        daily["price_low_threshold"] / rt_eff
        + cfg.degradation_cost_eur_kwh
        + cfg.minimum_arbitrage_margin_eur_kwh
    )
    daily["arbitrage_viable"] = (
        (daily["price_max"] - daily["price_min"] > 1e-6)
        & (daily["price_high_threshold"] > daily["required_high_price"])
    )

    df = df.merge(daily, on="date", how="left")
    df["is_cheap_price"] = (
        df["arbitrage_viable"].fillna(False)
        & (df["buy_price_eur_kwh"] <= df["price_low_threshold"])
    )
    df["is_expensive_price"] = (
        df["arbitrage_viable"].fillna(False)
        & (df["buy_price_eur_kwh"] >= df["price_high_threshold"])
    )
    return df.drop(columns=["date"])


def simulate_seasonal_bess(
    data: pd.DataFrame,
    bess_kwh: float,
    bess_kw: float,
    charge_eff: float,
    discharge_eff: float,
    soc_min: float,
    soc_max: float,
    soc_initial: float,
    grid_import_limit_kw: float,
    grid_export_limit_kw: float,
    cfg: ModelConfig = ModelConfig(),
) -> pd.DataFrame:
    """Seasonal dispatch model.

    Summer (Apr-Oct): PV -> load -> BESS; BESS is discharged primarily at night.
    Transition (Mar/Nov): PV self-consumption + price arbitrage when viable.
    Winter (Dec-Feb): PV self-consumption + grid charging at cheap prices and
    discharge at expensive prices when arbitrage is economically viable.
    """
    df = data.copy().sort_values("datetime").reset_index(drop=True)
    dt = detect_timestep_hours(df)

    if "season" not in df.columns or "is_day" not in df.columns:
        solar = build_solar_calendar(df["datetime"], cfg)
        df = df.merge(
            solar[["datetime", "season", "is_day", "is_night"]],
            on="datetime", how="left"
        )

    df = add_price_signals(df, charge_eff, discharge_eff, cfg)

    min_energy = bess_kwh * soc_min
    max_energy = bess_kwh * soc_max
    initial_energy = min(max(bess_kwh * soc_initial, min_energy), max_energy)

    # Energy above minimum SOC is source-tracked.
    usable_initial = max(initial_energy - min_energy, 0.0)
    pv_bucket = 0.0
    grid_bucket = usable_initial  # conservative: initial usable energy is grid/unknown

    records = []

    for _, row in df.iterrows():
        pv = max(float(row.get("pv_kwh", 0.0)), 0.0)
        load = max(float(row.get("load_kwh", 0.0)), 0.0)
        season = str(row.get("season", "summer"))
        is_night = bool(row.get("is_night", False))
        is_cheap = bool(row.get("is_cheap_price", False))
        is_expensive = bool(row.get("is_expensive_price", False))

        stored = pv_bucket + grid_bucket
        soc = min_energy + stored
        available_capacity = max(max_energy - soc, 0.0)

        direct_pv = min(pv, load)
        surplus = max(pv - load, 0.0)
        deficit = max(load - pv, 0.0)

        pv_charge = 0.0
        grid_charge = 0.0
        discharge = 0.0
        discharge_pv = 0.0
        discharge_grid = 0.0
        grid_load = 0.0
        export = 0.0
        curtailment = 0.0
        unserved = 0.0

        power_charge_limit = max(bess_kw * dt, 0.0)
        power_discharge_limit = max(bess_kw * dt, 0.0)

        # 1) PV surplus always gets first opportunity to charge the battery.
        if bess_kwh > 0 and bess_kw > 0 and surplus > 0 and available_capacity > 0:
            pv_charge = min(
                surplus,
                power_charge_limit,
                available_capacity / max(charge_eff, 1e-9),
            )
            stored_add = pv_charge * charge_eff
            pv_bucket += stored_add
            available_capacity -= stored_add

        export_available = max(surplus - pv_charge, 0.0)
        export = min(export_available, grid_export_limit_kw * dt)
        curtailment = max(export_available - export, 0.0)

        # 2) Decide whether battery should discharge for the remaining load.
        should_discharge = False
        if deficit > 0:
            if season == "summer":
                # Summer battery is reserved for the night-time load.
                should_discharge = is_night
            elif season == "transition":
                # Mixed mode: night coverage and expensive-price shaving.
                should_discharge = is_night or is_expensive
            else:  # winter
                # Winter dispatch is price-driven; no arbitrary discharge at medium price.
                should_discharge = is_expensive

        if should_discharge and bess_kwh > 0 and bess_kw > 0:
            stored = pv_bucket + grid_bucket
            available_output = stored * discharge_eff
            discharge = min(deficit, power_discharge_limit, available_output)
            if discharge > 0 and stored > 0:
                removed_stored = discharge / max(discharge_eff, 1e-9)
                pv_share = pv_bucket / stored
                grid_share = grid_bucket / stored
                removed_pv = min(pv_bucket, removed_stored * pv_share)
                removed_grid = min(grid_bucket, removed_stored * grid_share)
                # Numerical remainder correction.
                remainder = removed_stored - removed_pv - removed_grid
                if remainder > 1e-10:
                    extra_pv = min(max(pv_bucket - removed_pv, 0.0), remainder)
                    removed_pv += extra_pv
                    remainder -= extra_pv
                if remainder > 1e-10:
                    removed_grid += min(max(grid_bucket - removed_grid, 0.0), remainder)
                pv_bucket -= removed_pv
                grid_bucket -= removed_grid
                discharge_pv = removed_pv * discharge_eff
                discharge_grid = removed_grid * discharge_eff

        remaining_deficit = max(deficit - discharge, 0.0)
        grid_load = min(remaining_deficit, grid_import_limit_kw * dt)
        unserved = max(remaining_deficit - grid_load, 0.0)

        # 3) Grid charging is allowed only in transition/winter and only when
        # actual price signals show profitable arbitrage.
        if (
            season in {"transition", "winter"}
            and is_cheap
            and bess_kwh > 0
            and bess_kw > 0
        ):
            stored = pv_bucket + grid_bucket
            soc = min_energy + stored
            available_capacity = max(max_energy - soc, 0.0)
            remaining_charge_power = max(power_charge_limit - pv_charge, 0.0)
            grid_import_headroom = max(grid_import_limit_kw * dt - grid_load, 0.0)
            grid_charge = min(
                remaining_charge_power,
                grid_import_headroom,
                available_capacity / max(charge_eff, 1e-9),
            )
            grid_bucket += grid_charge * charge_eff

        soc = min_energy + pv_bucket + grid_bucket
        soc = min(max(soc, min_energy), max_energy)

        records.append(
            {
                "pv_direct_kwh": direct_pv,
                "pv_charge_kwh": pv_charge,
                "grid_charge_kwh": grid_charge,
                "bess_charge_kwh": pv_charge + grid_charge,
                "bess_discharge_kwh": discharge,
                "bess_discharge_pv_kwh": discharge_pv,
                "bess_discharge_grid_kwh": discharge_grid,
                "grid_import_load_kwh": grid_load,
                "grid_import_kwh": grid_load + grid_charge,
                "export_kwh": export,
                "curtailment_kwh": curtailment,
                "unserved_load_kwh": unserved,
                "soc_kwh": soc,
                "soc_pct": (soc / bess_kwh * 100.0) if bess_kwh > 0 else 0.0,
                "soc_pv_kwh": pv_bucket,
                "soc_grid_kwh": grid_bucket,
            }
        )

    rec = pd.DataFrame(records)
    return pd.concat([df.reset_index(drop=True), rec], axis=1)


def summarize_scenario(result: pd.DataFrame, bess_kwh: float, bess_kw: float, soc_min: float, soc_max: float, discharge_eff: float) -> dict:
    annual_load = float(result["load_kwh"].sum())
    annual_pv = float(result["pv_kwh"].sum())
    annual_grid = float(result["grid_import_kwh"].sum())
    grid_for_load = float(result["grid_import_load_kwh"].sum())
    grid_charge = float(result["grid_charge_kwh"].sum())
    annual_export = float(result["export_kwh"].sum())
    annual_charge = float(result["bess_charge_kwh"].sum())
    annual_discharge = float(result["bess_discharge_kwh"].sum())
    discharge_pv = float(result["bess_discharge_pv_kwh"].sum())
    discharge_grid = float(result["bess_discharge_grid_kwh"].sum())
    curtailment = float(result["curtailment_kwh"].sum())
    unserved = float(result["unserved_load_kwh"].sum())

    renewable_load_supply = float(result["pv_direct_kwh"].sum()) + discharge_pv
    renewable_self_sufficiency = renewable_load_supply / annual_load if annual_load > 0 else 0.0
    grid_dependency = annual_grid / annual_load if annual_load > 0 else 0.0
    self_consumption = (
        (annual_pv - annual_export - curtailment) / annual_pv if annual_pv > 0 else 0.0
    )

    usable_capacity = bess_kwh * max(soc_max - soc_min, 0.0)
    equivalent_cycles = (
        annual_discharge / (usable_capacity * max(discharge_eff, 1e-9))
        if usable_capacity > 0 else 0.0
    )

    if "buy_price_eur_kwh" in result.columns:
        actual_grid_cost = float(
            (result["grid_import_kwh"] * result["buy_price_eur_kwh"]).sum()
        )
        baseline_cost = float(
            (result["load_kwh"] * result["buy_price_eur_kwh"]).sum()
        )
    else:
        actual_grid_cost = np.nan
        baseline_cost = np.nan

    return {
        "pv_kw": float(result["pv_kw"].iloc[0]) if "pv_kw" in result.columns and len(result) else np.nan,
        "bess_kwh": float(bess_kwh),
        "bess_kw": float(bess_kw),
        "bess_duration_h": float(bess_kwh / bess_kw) if bess_kw > 0 else 0.0,
        "load_kwh": annual_load,
        "pv_generation_kwh": annual_pv,
        "grid_import_kwh": annual_grid,
        "grid_import_load_kwh": grid_for_load,
        "grid_charge_kwh": grid_charge,
        "export_kwh": annual_export,
        "bess_charge_kwh": annual_charge,
        "bess_discharge_kwh": annual_discharge,
        "bess_discharge_pv_kwh": discharge_pv,
        "bess_discharge_grid_kwh": discharge_grid,
        "curtailment_kwh": curtailment,
        "unserved_load_kwh": unserved,
        "self_sufficiency_pct": renewable_self_sufficiency * 100.0,
        "grid_dependency_pct": grid_dependency * 100.0,
        "self_consumption_pct": self_consumption * 100.0,
        "equivalent_cycles": equivalent_cycles,
        "actual_grid_cost_eur": actual_grid_cost,
        "baseline_energy_cost_eur": baseline_cost,
    }


def run_seasonal_scenario(
    base_data: pd.DataFrame,
    pv_kw: float,
    bess_kwh: float,
    bess_kw: float,
    charge_eff: float,
    discharge_eff: float,
    soc_min: float,
    soc_max: float,
    soc_initial: float,
    grid_import_limit_kw: float,
    grid_export_limit_kw: float,
    cfg: ModelConfig = ModelConfig(),
) -> tuple[dict, pd.DataFrame]:
    df = base_data.copy()
    df["pv_kw"] = float(pv_kw)
    df["pv_kwh"] = df["pv_per_kw"] * float(pv_kw)
    result = simulate_seasonal_bess(
        df,
        bess_kwh=bess_kwh,
        bess_kw=bess_kw,
        charge_eff=charge_eff,
        discharge_eff=discharge_eff,
        soc_min=soc_min,
        soc_max=soc_max,
        soc_initial=soc_initial,
        grid_import_limit_kw=grid_import_limit_kw,
        grid_export_limit_kw=grid_export_limit_kw,
        cfg=cfg,
    )
    summary = summarize_scenario(result, bess_kwh, bess_kw, soc_min, soc_max, discharge_eff)
    return summary, result


def _series_stats(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return {"mean": np.nan, "median": np.nan, "p90": np.nan, "p95": np.nan, "max": np.nan}
    return {
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max()),
    }


def calculate_sizing_metrics(
    base_data: pd.DataFrame,
    pv_kw: float,
    discharge_eff: float = 0.95,
    soc_min: float = 0.05,
    soc_max: float = 0.95,
    summer_reserve_pct: float = 15.0,
    cfg: ModelConfig = ModelConfig(),
) -> dict:
    """Calculate technical sizing indicators independent of dispatch optimization."""
    df = base_data.copy().sort_values("datetime").reset_index(drop=True)
    if "season" not in df.columns or "is_day" not in df.columns:
        solar = build_solar_calendar(df["datetime"], cfg)
        df = df.merge(
            solar[["datetime", "season", "is_day", "is_night", "day_length_h", "night_length_h"]],
            on="datetime", how="left"
        )
    df["pv_kwh"] = df["pv_per_kw"] * float(pv_kw)
    df["date"] = pd.to_datetime(df["datetime"]).dt.normalize()

    usable_factor = max((soc_max - soc_min) * discharge_eff, 1e-9)

    # Summer night demand and daily PV surplus.
    summer = df[df["season"] == "summer"].copy()
    summer_night = (
        summer[summer["is_night"]]
        .groupby("date")["load_kwh"].sum()
    )
    summer_stats = _series_stats(summer_night)
    summer_target = max(
        summer_stats["mean"] * (1.0 + summer_reserve_pct / 100.0),
        summer_stats["p90"],
    ) if len(summer_night) else np.nan
    c_summer = summer_target / usable_factor if np.isfinite(summer_target) else np.nan

    summer_day = summer[summer["is_day"]].copy()
    summer_day_load = float(summer_day["load_kwh"].sum())
    summer_day_direct = float(np.minimum(summer_day["pv_kwh"], summer_day["load_kwh"]).sum())
    summer_day_coverage = summer_day_direct / summer_day_load if summer_day_load > 0 else np.nan

    summer["pv_surplus_kwh"] = np.maximum(summer["pv_kwh"] - summer["load_kwh"], 0.0)
    daily_surplus = summer.groupby("date")["pv_surplus_kwh"].sum()
    summer_compare = pd.concat(
        [summer_night.rename("night_load_kwh"), daily_surplus.rename("pv_surplus_kwh")],
        axis=1,
    ).fillna(0.0)
    # PV charge input required to later deliver night energy.
    required_charge_input = summer_compare["night_load_kwh"] / max(discharge_eff * 0.95, 1e-9)
    if len(summer_compare):
        full_charge_days_pct = float((summer_compare["pv_surplus_kwh"] >= required_charge_input).mean() * 100.0)
    else:
        full_charge_days_pct = np.nan

    # Winter technical daytime demand.
    winter = df[df["season"] == "winter"].copy()
    winter_day_load = (
        winter[winter["is_day"]]
        .groupby("date")["load_kwh"].sum()
    )
    winter_stats = _series_stats(winter_day_load)
    c_winter_technical = winter_stats["p90"] / usable_factor if len(winter_day_load) else np.nan

    winter_load_total = float(winter["load_kwh"].sum())
    winter_pv_direct = float(np.minimum(winter["pv_kwh"], winter["load_kwh"]).sum())
    winter_pv_coverage = winter_pv_direct / winter_load_total if winter_load_total > 0 else np.nan

    # If genuine variable prices exist in winter, estimate expensive-period demand.
    price_based_winter = np.nan
    c_winter_price = np.nan
    if "buy_price_eur_kwh" in winter.columns and len(winter):
        signaled = add_price_signals(winter, 0.95, discharge_eff, cfg)
        expensive = signaled[signaled["is_expensive_price"]]
        if len(expensive):
            daily_expensive = expensive.groupby(expensive["datetime"].dt.normalize())["load_kwh"].sum()
            if len(daily_expensive):
                price_based_winter = float(daily_expensive.quantile(0.90))
                c_winter_price = price_based_winter / usable_factor

    # Rolling 24 h energy. This works for hourly/sub-hourly time series.
    dt = detect_timestep_hours(df)
    window = max(1, int(round(24.0 / dt)))
    rolling_24 = df["load_kwh"].rolling(window=window, min_periods=window).sum().dropna()
    autonomy_stats = _series_stats(rolling_24)
    c_24_p90 = autonomy_stats["p90"] / usable_factor if len(rolling_24) else np.nan
    c_24_max = autonomy_stats["max"] / usable_factor if len(rolling_24) else np.nan

    # Monthly daylight/night hours.
    day_table = (
        df.assign(month=df["datetime"].dt.month)
        .groupby("month", as_index=False)[["day_length_h", "night_length_h"]]
        .mean()
    )

    return {
        "summer_night_stats": summer_stats,
        "summer_target_delivered_kwh": float(summer_target) if np.isfinite(summer_target) else np.nan,
        "summer_bess_nominal_kwh": float(c_summer) if np.isfinite(c_summer) else np.nan,
        "summer_day_pv_coverage_pct": float(summer_day_coverage * 100.0) if np.isfinite(summer_day_coverage) else np.nan,
        "summer_full_charge_days_pct": full_charge_days_pct,
        "winter_day_stats": winter_stats,
        "winter_bess_technical_kwh": float(c_winter_technical) if np.isfinite(c_winter_technical) else np.nan,
        "winter_pv_coverage_pct": float(winter_pv_coverage * 100.0) if np.isfinite(winter_pv_coverage) else np.nan,
        "winter_expensive_p90_kwh": price_based_winter,
        "winter_bess_price_based_kwh": c_winter_price,
        "autonomy_24h_stats": autonomy_stats,
        "autonomy_bess_p90_kwh": float(c_24_p90) if np.isfinite(c_24_p90) else np.nan,
        "autonomy_bess_max_kwh": float(c_24_max) if np.isfinite(c_24_max) else np.nan,
        "monthly_daylight": day_table,
    }


def validate_energy_balance(result: pd.DataFrame, charge_eff: float, discharge_eff: float) -> dict:
    """Return maximum interval and annual balance residuals for QA."""
    # Load-side balance: PV direct + battery + grid + unserved = load.
    load_rhs = (
        result["pv_direct_kwh"]
        + result["bess_discharge_kwh"]
        + result["grid_import_load_kwh"]
        + result["unserved_load_kwh"]
    )
    load_residual = result["load_kwh"] - load_rhs

    # PV-side balance: direct + charge input + export + curtailment = PV.
    pv_rhs = (
        result["pv_direct_kwh"]
        + result["pv_charge_kwh"]
        + result["export_kwh"]
        + result["curtailment_kwh"]
    )
    pv_residual = result["pv_kwh"] - pv_rhs

    return {
        "max_abs_load_balance_kwh": float(load_residual.abs().max()),
        "annual_load_balance_kwh": float(load_residual.sum()),
        "max_abs_pv_balance_kwh": float(pv_residual.abs().max()),
        "annual_pv_balance_kwh": float(pv_residual.sum()),
    }
