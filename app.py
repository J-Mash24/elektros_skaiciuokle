# ============================================================
# PV + BESS ENERGETINIO OPTIMIZAVIMO SKAIČIUOKLĖ
# Streamlit application
# ============================================================
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ============================================================
# 1. PROGRAMĖLĖS KONFIGŪRACIJA
# ============================================================

st.set_page_config(
    page_title="PV + BESS skaičiuoklė",
    layout="wide"
)

st.title("PV ir BESS energetinio optimizavimo skaičiuoklė")

st.caption(
    "Valandinis PV generacijos, elektros vartojimo, "
    "baterijų kaupimo ir ekonominio efektyvumo modelis."
)


# ============================================================
# 2. PAGALBINĖS FUNKCIJOS
# ============================================================

def solar_shape(hour, sunrise=6, sunset=20):
    """
    Supaprastinta PV generacijos paros kreivė.
    """

    if hour < sunrise or hour >= sunset:
        return 0.0

    x = (
        (hour - sunrise)
        / (sunset - sunrise)
        * np.pi
    )

    return max(np.sin(x), 0.0)


def create_solar_weights():
    """
    Sukuria 24 valandų PV generacijos svorius.
    """

    hours = np.arange(24)

    weights = np.array(
        [
            solar_shape(hour)
            for hour in hours
        ],
        dtype=float
    )

    if weights.sum() == 0:
        raise ValueError(
            "Nepavyko suformuoti saulės generacijos profilio."
        )

    return weights / weights.sum()


def read_reference_pv_files(
    uploaded_files,
    reference_pv_kw
):
    """
    Perskaito vieną arba kelis referencinės PV elektrinės Excel failus.

    Programa automatiškai ieško Excel lapo, kuriame yra:
    - Statistical Period
    - PV Yield (kWh)

    Tikrinamos ir 1, ir 2 Excel antraštės eilutės.
    """

    frames = []

    required_columns = [
        "Statistical Period",
        "PV Yield (kWh)"
    ]

    for uploaded_file in uploaded_files:

        uploaded_file.seek(0)

        excel = pd.ExcelFile(uploaded_file)

        matching_df = None
        matching_sheet = None

        # Tikriname visus Excel lapus
        for sheet in excel.sheet_names:

            # Tikriname header=0 ir header=1
            for header_row in [0, 1]:

                uploaded_file.seek(0)

                try:

                    candidate = pd.read_excel(
                        uploaded_file,
                        sheet_name=sheet,
                        header=header_row
                    )

                except Exception:
                    continue

                if all(
                    col in candidate.columns
                    for col in required_columns
                ):

                    matching_df = candidate
                    matching_sheet = sheet
                    break

            if matching_df is not None:
                break

        if matching_df is None:

            raise ValueError(
                f"Faile '{uploaded_file.name}' nepavyko rasti "
                "lapo su stulpeliais "
                "'Statistical Period' ir 'PV Yield (kWh)'. "
                f"Rasti lapai: {excel.sheet_names}"
            )

        temp = matching_df[
            [
                "Statistical Period",
                "PV Yield (kWh)"
            ]
        ].copy()

        temp.columns = [
            "date",
            "pv_ref_kwh"
        ]

        temp["source_file"] = uploaded_file.name
        temp["source_sheet"] = matching_sheet

        frames.append(temp)

    if len(frames) == 0:

        raise ValueError(
            "Nepavyko perskaityti nė vieno PV failo."
        )

    pv = pd.concat(
        frames,
        ignore_index=True
    )

    pv["date"] = pd.to_datetime(
        pv["date"],
        errors="coerce"
    )

    pv["pv_ref_kwh"] = pd.to_numeric(
        pv["pv_ref_kwh"],
        errors="coerce"
    )

    pv = pv.dropna(
        subset=[
            "date",
            "pv_ref_kwh"
        ]
    )

    pv = (
        pv.groupby(
            "date",
            as_index=False
        )["pv_ref_kwh"]
        .sum()
    )

    pv = pv.sort_values(
        "date"
    ).reset_index(drop=True)

    pv["yield_kwh_per_kw"] = (
        pv["pv_ref_kwh"]
        / reference_pv_kw
    )

    return pv


def create_hourly_pv_profile(
    pv_daily
):
    """
    Dieninį 1 kW PV profilį paverčia valandiniu.
    """

    solar_weights = create_solar_weights()

    rows = []

    for _, row in pv_daily.iterrows():

        for hour in range(24):

            timestamp = (
                row["date"]
                + pd.Timedelta(hours=hour)
            )

            pv_per_kw = (
                row["yield_kwh_per_kw"]
                * solar_weights[hour]
            )

            rows.append(
                {
                    "datetime": timestamp,
                    "pv_per_kw": pv_per_kw
                }
            )

    hourly = pd.DataFrame(rows)

    hourly = hourly.sort_values(
        "datetime"
    ).reset_index(drop=True)

    return hourly

def map_pv_profile_to_load_dates(
    hourly_pv,
    load_df
):
    """
    Referencinės PV elektrinės profilį pritaiko faktinio
    vartojimo kalendorinėms datoms.

    Naudojami:
    mėnuo + diena + valanda.

    Todėl PV ir vartojimo duomenys nebūtinai turi būti tų pačių metų.
    """

    pv = hourly_pv.copy()
    load = load_df.copy()

    pv["month"] = pv["datetime"].dt.month
    pv["day"] = pv["datetime"].dt.day
    pv["hour"] = pv["datetime"].dt.hour

    # Jei vienodų kalendorinių momentų būtų daugiau,
    # imame vidutinę specifinę generaciją.
    pv_profile = (
        pv.groupby(
            [
                "month",
                "day",
                "hour"
            ],
            as_index=False
        )["pv_per_kw"]
        .mean()
    )

    load["month"] = load["datetime"].dt.month
    load["day"] = load["datetime"].dt.day
    load["hour"] = load["datetime"].dt.hour

    result = load.merge(
        pv_profile,
        on=[
            "month",
            "day",
            "hour"
        ],
        how="left"
    )

    # Vasario 29 d. ar kitų neatitikimų atveju
    result["pv_per_kw"] = (
        result["pv_per_kw"]
        .fillna(0)
    )

    return result[
        [
            "datetime",
            "load_kwh",
            "pv_per_kw"
        ]
    ].sort_values(
        "datetime"
    ).reset_index(drop=True)


def create_synthetic_load(
    datetime_series,
    annual_load_kwh
):
    """
    Sukuria sintetinį valandinį vartojimo profilį.
    """

    hour_factors = {
        0: 0.60, 1: 0.55, 2: 0.55, 3: 0.55,
        4: 0.55, 5: 0.60, 6: 0.70, 7: 0.85,
        8: 1.00, 9: 1.10, 10: 1.15, 11: 1.20,
        12: 1.20, 13: 1.15, 14: 1.10, 15: 1.05,
        16: 1.00, 17: 0.95, 18: 0.90, 19: 0.85,
        20: 0.80, 21: 0.75, 22: 0.70, 23: 0.65
    }

    month_factors = {
        1: 1.10,
        2: 1.08,
        3: 1.03,
        4: 0.98,
        5: 0.95,
        6: 0.93,
        7: 0.93,
        8: 0.95,
        9: 0.98,
        10: 1.02,
        11: 1.06,
        12: 1.10
    }

    df = pd.DataFrame({
        "datetime": pd.to_datetime(datetime_series)
    })

    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    df["month"] = df["datetime"].dt.month

    df["hour_factor"] = df["hour"].map(hour_factors)

    df["day_factor"] = np.where(
        df["weekday"] < 5,
        1.0,
        0.75
    )

    df["month_factor"] = (
        df["month"].map(month_factors)
    )

    df["raw_factor"] = (
        df["hour_factor"]
        * df["day_factor"]
        * df["month_factor"]
    )

    df["load_kwh"] = (
        df["raw_factor"]
        / df["raw_factor"].sum()
        * annual_load_kwh
    )

    return df[
        [
            "datetime",
            "load_kwh"
        ]
    ]

def read_actual_load(
    uploaded_file,
    sheet_name=None
):
    """
    Perskaito faktinį elektros vartojimą.

    Atpažįsta, pvz.:
    - datetime / Data, valanda
    - load_kwh / Kiekis, kWh / Suvartojimas, kWh

    ESO failuose papildomai:
    - paliekamas P+ energijos tipas;
    - jei yra stulpelis 'Suvartojimas',
      pirmenybė teikiama faktiniams duomenims.
    """

    uploaded_file.seek(0)

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    if uploaded_file.name.lower().endswith(".csv"):

        df = pd.read_csv(
            uploaded_file
        )

    # --------------------------------------------------------
    # Excel
    # --------------------------------------------------------

    else:

        excel = pd.ExcelFile(
            uploaded_file
        )

        if sheet_name is None:

            # Pirmiausia ieškome lapo su "Valandiniai"
            matching_sheets = [
                sheet
                for sheet in excel.sheet_names
                if "valandin" in sheet.lower()
            ]

            if matching_sheets:

                sheet_name = matching_sheets[0]

            else:

                sheet_name = excel.sheet_names[0]

        uploaded_file.seek(0)

        df = pd.read_excel(
            uploaded_file,
            sheet_name=sheet_name
        )

    # Pašaliname nereikalingus tarpus pavadinimuose
    df.columns = [
        str(col).strip()
        for col in df.columns
    ]

    # --------------------------------------------------------
    # ESO filtrai
    # --------------------------------------------------------

    if "Energijos tipas" in df.columns:

        p_plus = (
            df["Energijos tipas"]
            .astype(str)
            .str.strip()
            .eq("P+")
        )

        if p_plus.any():

            df = df[p_plus].copy()

    if "Suvartojimas" in df.columns:

        faktinis = (
            df["Suvartojimas"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("faktinis")
        )

        if faktinis.any():

            df = df[faktinis].copy()

    # --------------------------------------------------------
    # Datos stulpelio paieška
    # --------------------------------------------------------

    datetime_candidates = [
        "datetime",
        "Data, valanda",
        "Data",
        "Laikotarpis"
    ]

    datetime_col = next(
        (
            col
            for col in datetime_candidates
            if col in df.columns
        ),
        None
    )

    if datetime_col is None:

        raise ValueError(
            "Nepavyko rasti datos / laiko stulpelio. "
            f"Faile yra stulpeliai: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # Energijos stulpelio paieška
    # --------------------------------------------------------

    load_candidates = [
        "load_kwh",
        "Kiekis, kWh",
        "Suvartojimas, kWh"
    ]

    load_col = next(
        (
            col
            for col in load_candidates
            if col in df.columns
        ),
        None
    )

    if load_col is None:

        raise ValueError(
            "Nepavyko rasti suvartojimo stulpelio. "
            f"Faile yra stulpeliai: {list(df.columns)}"
        )

    result = df[
        [
            datetime_col,
            load_col
        ]
    ].copy()

    result.columns = [
        "datetime",
        "load_kwh"
    ]

    result["datetime"] = pd.to_datetime(
        result["datetime"],
        errors="coerce"
    )

    if result["datetime"].dt.tz is not None:
        result["datetime"] = (
            result["datetime"]
            .dt.tz_localize(None)
        )

    result["load_kwh"] = pd.to_numeric(
        result["load_kwh"],
        errors="coerce"
    )

    result = result.dropna(
        subset=[
            "datetime",
            "load_kwh"
        ]
    )

    result = (
        result.groupby(
            "datetime",
            as_index=False
        )["load_kwh"]
        .sum()
    )

    result = result.sort_values(
        "datetime"
    ).reset_index(drop=True)

    return result

# ============================================================
# 3. BESS SIMULIAVIMO FUNKCIJA
# ============================================================
def detect_timestep_hours(df):
    """
    Automatiškai nustato duomenų intervalo trukmę valandomis.

    Pvz.:
    1 h    -> 1.0
    30 min -> 0.5
    15 min -> 0.25
    """

    timestamps = (
        pd.to_datetime(df["datetime"])
        .sort_values()
    )

    differences = (
        timestamps
        .diff()
        .dropna()
    )

    if len(differences) == 0:
        return 1.0

    median_delta = differences.median()

    dt_hours = (
        median_delta.total_seconds()
        / 3600
    )

    if dt_hours <= 0:

        raise ValueError(
            "Nepavyko nustatyti laiko intervalo."
        )

    return dt_hours

def simulate_bess(
    data,
    bess_kwh,
    bess_kw,
    charge_eff,
    discharge_eff,
    soc_min,
    soc_max,
    soc_initial,
    grid_import_limit_kw,
    grid_export_limit_kw
):
    """
    Valandinė PV + Load + BESS + Grid simuliacija.
    """

    df = data.copy()

    dt = detect_timestep_hours(df)

    min_energy = (
        bess_kwh
        * soc_min
    )

    max_energy = (
        bess_kwh
        * soc_max
    )

    soc = (
        bess_kwh
        * soc_initial
    )

    # Užtikriname ribas
    soc = min(
        max(soc, min_energy),
        max_energy
    )

    soc_values = []
    charge_values = []
    discharge_values = []

    grid_values = []
    export_values = []

    curtailment_values = []
    unserved_values = []

    direct_pv_values = []

    for _, row in df.iterrows():

        pv = max(
            float(row["pv_kwh"]),
            0.0
        )

        load = max(
            float(row["load_kwh"]),
            0.0
        )

        direct_pv = min(
            pv,
            load
        )

        charge = 0.0
        discharge = 0.0
        grid = 0.0
        export = 0.0
        curtailment = 0.0
        unserved = 0.0

        # ====================================================
        # PV perteklius
        # ====================================================

        if pv >= load:

            surplus = (
                pv - load
            )

            available_capacity = max(
                max_energy - soc,
                0.0
            )

            if bess_kwh > 0 and bess_kw > 0:

                charge = min(
                    surplus,
                    bess_kw * dt,
                    (
                        available_capacity
                        / charge_eff
                        if charge_eff > 0
                        else 0.0
                    )
                )

            soc += (
                charge
                * charge_eff
            )

            export_available = (
                surplus
                - charge
            )

            export = min(
                export_available,
                grid_export_limit_kw * dt
            )

            curtailment = max(
                export_available
                - export,
                0.0
            )

        # ====================================================
        # Energijos deficitas
        # ====================================================

        else:

            deficit = (
                load - pv
            )

            available_energy = max(
                soc - min_energy,
                0.0
            )

            if bess_kwh > 0 and bess_kw > 0:

                discharge = min(
                    deficit,
                    bess_kw * dt,
                    available_energy
                    * discharge_eff
                )

            if discharge_eff > 0:

                soc -= (
                    discharge
                    / discharge_eff
                )

            grid_required = (
                deficit
                - discharge
            )

            grid = min(
                grid_required,
                grid_import_limit_kw * dt
            )

            unserved = max(
                grid_required
                - grid,
                0.0
            )

        soc = min(
            max(soc, min_energy),
            max_energy
        )

        direct_pv_values.append(
            direct_pv
        )

        charge_values.append(
            charge
        )

        discharge_values.append(
            discharge
        )

        soc_values.append(
            soc
        )

        grid_values.append(
            grid
        )

        export_values.append(
            export
        )

        curtailment_values.append(
            curtailment
        )

        unserved_values.append(
            unserved
        )

    df["pv_direct_kwh"] = (
        direct_pv_values
    )

    df["bess_charge_kwh"] = (
        charge_values
    )

    df["bess_discharge_kwh"] = (
        discharge_values
    )

    df["soc_kwh"] = (
        soc_values
    )

    df["grid_import_kwh"] = (
        grid_values
    )

    df["export_kwh"] = (
        export_values
    )

    df["curtailment_kwh"] = (
        curtailment_values
    )

    df["unserved_load_kwh"] = (
        unserved_values
    )

    return df


# ============================================================
# 4. VIENO SCENARIJAUS FUNKCIJA
# ============================================================

def run_scenario(
    base_hourly,
    pv_kw,
    bess_kwh,
    bess_kw,
    charge_eff,
    discharge_eff,
    soc_min,
    soc_max,
    soc_initial,
    grid_import_limit_kw,
    grid_export_limit_kw
):

    df = base_hourly.copy()

    df["pv_kwh"] = (
        df["pv_per_kw"]
        * pv_kw
    )

    result = simulate_bess(
        data=df,
        bess_kwh=bess_kwh,
        bess_kw=bess_kw,
        charge_eff=charge_eff,
        discharge_eff=discharge_eff,
        soc_min=soc_min,
        soc_max=soc_max,
        soc_initial=soc_initial,
        grid_import_limit_kw=grid_import_limit_kw,
        grid_export_limit_kw=grid_export_limit_kw
    )

    annual_load = (
        result["load_kwh"].sum()
    )

    annual_pv = (
        result["pv_kwh"].sum()
    )

    annual_grid = (
        result["grid_import_kwh"].sum()
    )

    annual_export = (
        result["export_kwh"].sum()
    )

    annual_charge = (
        result["bess_charge_kwh"].sum()
    )

    annual_discharge = (
        result["bess_discharge_kwh"].sum()
    )

    annual_curtailment = (
        result["curtailment_kwh"].sum()
    )

    annual_unserved = (
        result["unserved_load_kwh"].sum()
    )

    # Energetinis savarankiškumas
    if annual_load > 0:

        self_sufficiency = (
            1
            - annual_grid
            / annual_load
        )

    else:

        self_sufficiency = 0.0

    # PV savos gamybos panaudojimas
    if annual_pv > 0:

        self_consumption = (
            annual_pv
            - annual_export
            - annual_curtailment
        ) / annual_pv

    else:

        self_consumption = 0.0

    usable_capacity = (
        bess_kwh
        * (
            soc_max
            - soc_min
        )
    )

    if (
        usable_capacity > 0
        and discharge_eff > 0
    ):

        equivalent_cycles = (
            annual_discharge
            / (
                usable_capacity
                * discharge_eff
            )
        )

    else:

        equivalent_cycles = 0.0

    if bess_kw > 0:

        bess_duration_h = (
            bess_kwh
            / bess_kw
        )

    else:

        bess_duration_h = 0.0

    summary = {
        "pv_kw":
            pv_kw,

        "bess_kwh":
            bess_kwh,

        "bess_kw":
            bess_kw,

        "bess_duration_h":
            bess_duration_h,

        "load_kwh":
            annual_load,

        "pv_generation_kwh":
            annual_pv,

        "grid_import_kwh":
            annual_grid,

        "export_kwh":
            annual_export,

        "bess_charge_kwh":
            annual_charge,

        "bess_discharge_kwh":
            annual_discharge,

        "curtailment_kwh":
            annual_curtailment,

        "unserved_load_kwh":
            annual_unserved,

        "self_sufficiency_pct":
            self_sufficiency * 100,

        "self_consumption_pct":
            self_consumption * 100,

        "equivalent_cycles":
            equivalent_cycles
    }

    return summary, result


# ============================================================
# 5. EKONOMINĖ FUNKCIJA
# ============================================================

def add_economics(
    scenarios,
    pv_cost_eur_kw,
    bess_cost_eur_kwh,
    bess_cost_eur_kw,
    electricity_buy_price,
    electricity_sell_price,
    pv_opex_rate,
    bess_opex_rate,
    discount_rate,
    project_years,
    annual_load_kwh
):

    df = scenarios.copy()

    df["pv_capex_eur"] = (
        df["pv_kw"]
        * pv_cost_eur_kw
    )

    df["bess_capex_eur"] = (
        df["bess_kwh"]
        * bess_cost_eur_kwh
        +
        df["bess_kw"]
        * bess_cost_eur_kw
    )

    df["total_capex_eur"] = (
        df["pv_capex_eur"]
        +
        df["bess_capex_eur"]
    )

    df["grid_cost_eur"] = (
        df["grid_import_kwh"]
        * electricity_buy_price
    )

    df["export_revenue_eur"] = (
        df["export_kwh"]
        * electricity_sell_price
    )

    df["opex_eur"] = (
        df["pv_capex_eur"]
        * pv_opex_rate
        +
        df["bess_capex_eur"]
        * bess_opex_rate
    )

    df["annual_cost_eur"] = (
        df["grid_cost_eur"]
        +
        df["opex_eur"]
        -
        df["export_revenue_eur"]
    )

    base_annual_cost = (
        annual_load_kwh
        * electricity_buy_price
    )

    df["annual_savings_eur"] = (
        base_annual_cost
        -
        df["annual_cost_eur"]
    )

    df["simple_payback_years"] = np.where(
        df["annual_savings_eur"] > 0,
        (
            df["total_capex_eur"]
            / df["annual_savings_eur"]
        ),
        np.nan
    )

    discount_factor = sum(
        1
        / (
            1 + discount_rate
        ) ** year
        for year in range(
            1,
            project_years + 1
        )
    )

    df["npc_eur"] = (
        df["total_capex_eur"]
        +
        df["annual_cost_eur"]
        * discount_factor
    )

    df["npv_eur"] = (
        -df["total_capex_eur"]
        +
        df["annual_savings_eur"]
        * discount_factor
    )

    return df


# ============================================================
# 6. OPTIMIZAVIMO FUNKCIJA
# ============================================================

def optimize_system(
    base_hourly,
    pv_sizes,
    bess_energy_sizes,
    bess_power_sizes,
    charge_eff,
    discharge_eff,
    soc_min,
    soc_max,
    soc_initial,
    grid_import_limit_kw,
    grid_export_limit_kw
):

    scenario_results = []

    total = (
        len(pv_sizes)
        * len(bess_energy_sizes)
        * len(bess_power_sizes)
    )

    progress = st.progress(0)

    counter = 0

    for pv_kw in pv_sizes:

        for bess_kwh in bess_energy_sizes:

            for bess_kw in bess_power_sizes:

                counter += 1

                progress.progress(
                    min(
                        counter / total,
                        1.0
                    )
                )

                # BESS = 0 / 0 leidžiamas
                if (
                    bess_kwh == 0
                    and bess_kw != 0
                ):
                    continue

                if (
                    bess_kwh > 0
                    and bess_kw == 0
                ):
                    continue

                summary, _ = run_scenario(
                    base_hourly=base_hourly,
                    pv_kw=pv_kw,
                    bess_kwh=bess_kwh,
                    bess_kw=bess_kw,
                    charge_eff=charge_eff,
                    discharge_eff=discharge_eff,
                    soc_min=soc_min,
                    soc_max=soc_max,
                    soc_initial=soc_initial,
                    grid_import_limit_kw=grid_import_limit_kw,
                    grid_export_limit_kw=grid_export_limit_kw
                )

                scenario_results.append(
                    summary
                )

    progress.empty()

    return pd.DataFrame(
        scenario_results
    )


# ============================================================
# 7. ŠONINĖ JUOSTA - PAGRINDINIAI PARAMETRAI
# ============================================================

st.sidebar.header(
    "Objekto parametrai"
)

annual_load_kwh = st.sidebar.number_input(
    "Metinis elektros suvartojimas, kWh",
    min_value=0.0,
    value=659955.0,
    step=1000.0
)

reference_pv_kw = st.sidebar.number_input(
    "Referencinės PV elektrinės galia, kW",
    min_value=0.01,
    value=21.12,
    step=0.01
)

grid_import_limit_kw = st.sidebar.number_input(
    "Importo galios riba, kW",
    min_value=0.0,
    value=750.0,
    step=10.0
)

grid_export_limit_kw = st.sidebar.number_input(
    "Eksporto galios riba, kW",
    min_value=0.0,
    value=1000.0,
    step=10.0
)


# ============================================================
# 8. PAGRINDINIAI SKIRTUKAI
# ============================================================

tab_data, tab_scenario, tab_optimization, tab_results = st.tabs(
    [
        "1. Duomenys",
        "2. Scenarijus",
        "3. Optimizavimas",
        "4. Rezultatai"
    ]
)


# ============================================================
# 9. DUOMENŲ SKIRTUKAS
# ============================================================

with tab_data:

    st.subheader(
        "PV generacijos duomenys"
    )

    pv_files = st.file_uploader(
        "Įkelkite referencinės PV elektrinės Excel failus",
        type=["xlsx"],
        accept_multiple_files=True
    )

    st.info(
        "Tikimasi Excel failų, kuriuose antroje eilutėje "
        "yra stulpeliai 'Statistical Period' ir 'PV Yield (kWh)'."
    )

    # ========================================================
    # PV FAILŲ PERŽIŪRA
    # ========================================================

    if pv_files:

        view_mode = st.radio(
            "Excel failų peržiūra",
            [
                "Failai atskirai",
                "Sujungti PV duomenys"
            ],
            horizontal=True
        )

        # ----------------------------------------------------
        # 1. Kiekvienas Excel failas rodomas atskirai
        # ----------------------------------------------------

        if view_mode == "Failai atskirai":

            st.write(
                f"Įkelta failų: {len(pv_files)}"
            )

            for file_index, uploaded_file in enumerate(pv_files):

                with st.expander(
                    f"{file_index + 1}. {uploaded_file.name}",
                    expanded=False
                ):

                    try:

                        # Grąžiname failo žymeklį į pradžią
                        uploaded_file.seek(0)

                        # Perskaitome visą Excel failą

                        uploaded_file.seek(0)

                        excel_file = pd.ExcelFile(
                            uploaded_file
                        )

                        selected_sheet = st.selectbox(
                            "Pasirinkite Excel lapą",
                            excel_file.sheet_names,
                        key=f"pv_sheet_{file_index}_{uploaded_file.name}"
                        )

                        uploaded_file.seek(0)

                        full_excel = pd.read_excel(
                            uploaded_file,
                            sheet_name=selected_sheet
                        )

                        st.write(
                            f"Eilučių: {len(full_excel)} | "
                            f"Stulpelių: {len(full_excel.columns)}"
                        )

                        st.write(
                            "Stulpeliai:",
                            list(full_excel.columns)
                        )

                        st.dataframe(
                            full_excel,
                            use_container_width=True,
                            height=550
                        )

                    except Exception as exc:

                        st.error(
                            f"Nepavyko perskaityti failo "
                            f"{uploaded_file.name}: {exc}"
                        )

        # ----------------------------------------------------
        # 2. Visi failai sujungiami į vieną PV lentelę
        # ----------------------------------------------------

        else:

            try:

                # Grąžiname visus failus į pradžią
                for uploaded_file in pv_files:
                    uploaded_file.seek(0)

                pv_daily = read_reference_pv_files(
                    pv_files,
                    reference_pv_kw
                )

                reference_generation = (
                    pv_daily["pv_ref_kwh"].sum()
                )

                specific_yield = (
                    reference_generation
                    / reference_pv_kw
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Referencinė PV galia",
                    f"{reference_pv_kw:.2f} kW"
                )

                col2.metric(
                    "Metinė referencinė generacija",
                    f"{reference_generation / 1000:.2f} MWh"
                )

                col3.metric(
                    "Specifinė generacija",
                    f"{specific_yield:.1f} kWh/kW"
                )

                st.write(
                    f"Sujungtų duomenų eilučių: {len(pv_daily)}"
                )

                st.dataframe(
                    pv_daily,
                    use_container_width=True,
                    height=600
                )

            except Exception as exc:

                st.error(
                    f"PV duomenų klaida: {exc}"
                )

    else:

        st.info(
            "PV Excel failai dar neįkelti."
        )

    # ========================================================
    # ELEKTROS VARTOJIMO DUOMENYS
    # ========================================================

    st.subheader(
        "Elektros vartojimo duomenys"
    )

    load_source = st.radio(
        "Vartojimo profilis",
        [
            "Modeliuotas profilis",
            "Faktinis profilis"
        ]
    )


    load_file = None
    load_sheet = None
    load_preview = None

    if load_source == "Faktinis profilis":

        load_file = st.file_uploader(
            "Įkelkite faktinio elektros vartojimo CSV arba Excel failą",
            type=[
                "csv",
                "xlsx"
            ],
            key="actual_load_file"
        )

        if load_file is not None:

            if load_file.name.lower().endswith(".xlsx"):

                load_file.seek(0)

                excel = pd.ExcelFile(
                    load_file
                )

                load_sheet = st.selectbox(
                    "Pasirinkite vartojimo duomenų Excel lapą",
                    excel.sheet_names,
                    key="load_sheet_selector"
                )

                st.caption(
                    "Aptikti lapai: "
                    + ", ".join(excel.sheet_names)
                )

                load_file.seek(0)

                try:

                    load_preview = pd.read_excel(
                        load_file,
                        sheet_name=load_sheet
                    )

                    st.write(
                        f"Eilučių: {len(load_preview)} | "
                        f"Stulpelių: {len(load_preview.columns)}"
                    )

                    st.dataframe(
                        load_preview,
                        use_container_width=True,
                        height=450
                    )

                except Exception as exc:

                    st.error(
                        f"Nepavyko parodyti Excel lapo: {exc}"
                    )

            else:

                load_file.seek(0)

                load_preview = pd.read_csv(
                    load_file
                )

                st.dataframe(
                    load_preview,
                    use_container_width=True,
                    height=450
                )
# ============================================================
# 10. BAZINIO PROFILIO PARUOŠIMAS
# ============================================================

base_hourly = None
load_df = None
data_ready = False

actual_load_total = None
first_timestamp = None
last_timestamp = None
dt_hours = None
peak_load_kw = None

if pv_files:

    try:

        pv_daily = read_reference_pv_files(
            pv_files,
            reference_pv_kw
        )

        hourly_pv = create_hourly_pv_profile(
            pv_daily
        )

        # ----------------------------------------------------
        # MODELIOJAMAS VARTOJIMAS
        # ----------------------------------------------------

        if load_source == "Modeliuotas profilis":

            load_df = create_synthetic_load(
                hourly_pv["datetime"],
                annual_load_kwh
            )

            base_hourly = pd.merge(
                hourly_pv,
                load_df,
                on="datetime",
                how="inner"
            )

        # ----------------------------------------------------
        # FAKTINIS VARTOJIMAS
        # ----------------------------------------------------

        else:

            if load_file is not None:

                load_df = read_actual_load(
                    load_file,
                    sheet_name=load_sheet
                )

                base_hourly = map_pv_profile_to_load_dates(
                    hourly_pv,
                    load_df
                )

        # ----------------------------------------------------
        # DUOMENŲ PATIKRA
        # ----------------------------------------------------

        if (
            load_df is not None
            and base_hourly is not None
            and len(base_hourly) > 0
        ):

            data_ready = True

            actual_load_total = (
                load_df["load_kwh"].sum()
            )

            first_timestamp = (
                load_df["datetime"].min()
            )

            last_timestamp = (
                load_df["datetime"].max()
            )

            dt_hours = detect_timestep_hours(
                load_df
            )

            peak_load_kw = (
                load_df["load_kwh"].max()
                / dt_hours
            )

    except Exception as exc:

        st.error(
            f"Duomenų paruošimo klaida: {exc}"
        )

# ============================================================
# 11. SCENARIJAUS SKIRTUKAS
# ============================================================

with tab_scenario:

    st.subheader(
        "Vieno PV + BESS scenarijaus analizė"
    )

    if not data_ready:

        st.warning(
            "Pirmiausia įkelkite PV duomenis "
            "ir paruoškite vartojimo profilį."
        )

    else:

        col1, col2, col3 = st.columns(3)

        with col1:

            pv_kw = st.number_input(
                "PV elektrinės galia, kW",
                min_value=0.0,
                value=730.0,
                step=10.0
            )

        with col2:

            bess_kwh = st.number_input(
                "BESS talpa, kWh",
                min_value=0.0,
                value=1000.0,
                step=50.0
            )

        with col3:

            bess_kw = st.number_input(
                "BESS galia, kW",
                min_value=0.0,
                value=500.0,
                step=25.0
            )

        if bess_kw > 0:

            st.metric(
                "BESS nominali trukmė",
                f"{bess_kwh / bess_kw:.2f} h"
            )

        with st.expander(
            "Išplėstiniai BESS parametrai"
        ):

            charge_eff_pct = st.slider(
                "Įkrovimo efektyvumas, %",
                50.0,
                100.0,
                95.0
            )

            discharge_eff_pct = st.slider(
                "Iškrovimo efektyvumas, %",
                50.0,
                100.0,
                95.0
            )

            soc_min_pct = st.slider(
                "Minimalus SOC, %",
                0.0,
                50.0,
                5.0
            )

            soc_max_pct = st.slider(
                "Maksimalus SOC, %",
                50.0,
                100.0,
                95.0
            )

            soc_initial_pct = st.slider(
                "Pradinis SOC, %",
                0.0,
                100.0,
                5.0
            )

        if st.button(
            "Skaičiuoti scenarijų",
            type="primary"
        ):

            if soc_min_pct >= soc_max_pct:

                st.error(
                    "Minimalus SOC turi būti mažesnis už maksimalų SOC."
                )

            else:

                scenario_summary, scenario_hourly = run_scenario(
                    base_hourly=base_hourly,
                    pv_kw=pv_kw,
                    bess_kwh=bess_kwh,
                    bess_kw=bess_kw,
                    charge_eff=charge_eff_pct / 100,
                    discharge_eff=discharge_eff_pct / 100,
                    soc_min=soc_min_pct / 100,
                    soc_max=soc_max_pct / 100,
                    soc_initial=soc_initial_pct / 100,
                    grid_import_limit_kw=grid_import_limit_kw,
                    grid_export_limit_kw=grid_export_limit_kw
                )

                st.session_state[
                    "scenario_summary"
                ] = scenario_summary

                st.session_state[
                    "scenario_hourly"
                ] = scenario_hourly


# ============================================================
# 12. OPTIMIZAVIMO SKIRTUKAS
# ============================================================

with tab_optimization:

    st.subheader(
        "Automatinė PV + BESS optimizacija"
    )

    if not data_ready:

        st.warning(
            "Pirmiausia įkelkite ir paruoškite duomenis."
        )

    else:

        st.markdown(
            "### Paieškos ribos"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            pv_min = st.number_input(
                "PV minimumas, kW",
                value=300,
                step=50
            )

            pv_max = st.number_input(
                "PV maksimumas, kW",
                value=1000,
                step=50
            )

            pv_step = st.number_input(
                "PV žingsnis, kW",
                value=50,
                min_value=1
            )

        with c2:

            bess_e_min = st.number_input(
                "BESS talpos minimumas, kWh",
                value=0,
                step=250
            )

            bess_e_max = st.number_input(
                "BESS talpos maksimumas, kWh",
                value=2000,
                step=250
            )

            bess_e_step = st.number_input(
                "BESS talpos žingsnis, kWh",
                value=250,
                min_value=1
            )

        with c3:

            bess_p_min = st.number_input(
                "BESS galios minimumas, kW",
                value=0,
                step=125
            )

            bess_p_max = st.number_input(
                "BESS galios maksimumas, kW",
                value=750,
                step=125
            )

            bess_p_step = st.number_input(
                "BESS galios žingsnis, kW",
                value=125,
                min_value=1
            )

        target_ssr = st.slider(
            "Minimalus energetinis savarankiškumas, %",
            min_value=0,
            max_value=100,
            value=60
        )

        st.markdown(
            "### Ekonominės prielaidos"
        )

        e1, e2, e3 = st.columns(3)

        with e1:

            pv_cost = st.number_input(
                "PV CAPEX, €/kW",
                value=650.0
            )

            bess_energy_cost = st.number_input(
                "BESS CAPEX, €/kWh",
                value=300.0
            )

            bess_power_cost = st.number_input(
                "BESS galios CAPEX, €/kW",
                value=100.0
            )

        with e2:

            buy_price = st.number_input(
                "Elektros pirkimo kaina, €/kWh",
                value=0.15,
                format="%.3f"
            )

            sell_price = st.number_input(
                "Eksporto vertė, €/kWh",
                value=0.05,
                format="%.3f"
            )

        with e3:

            discount_rate_pct = st.number_input(
                "Diskonto norma, %",
                value=5.0
            )

            project_years = st.number_input(
                "Projekto laikotarpis, metai",
                value=20,
                min_value=1
            )

            pv_opex_pct = st.number_input(
                "PV OPEX, % CAPEX/metus",
                value=1.0
            )

            bess_opex_pct = st.number_input(
                "BESS OPEX, % CAPEX/metus",
                value=1.0
            )

        if st.button(
            "Paleisti optimizaciją",
            type="primary"
        ):

            pv_sizes = np.arange(
                pv_min,
                pv_max + pv_step,
                pv_step
            )

            bess_energy_sizes = np.arange(
                bess_e_min,
                bess_e_max + bess_e_step,
                bess_e_step
            )

            bess_power_sizes = np.arange(
                bess_p_min,
                bess_p_max + bess_p_step,
                bess_p_step
            )

            raw_scenarios = optimize_system(
                base_hourly=base_hourly,
                pv_sizes=pv_sizes,
                bess_energy_sizes=bess_energy_sizes,
                bess_power_sizes=bess_power_sizes,
                charge_eff=0.95,
                discharge_eff=0.95,
                soc_min=0.05,
                soc_max=0.95,
                soc_initial=0.05,
                grid_import_limit_kw=grid_import_limit_kw,
                grid_export_limit_kw=grid_export_limit_kw
            )

            # Atmetame techniškai netinkamus scenarijus
            valid = raw_scenarios[
                raw_scenarios[
                    "unserved_load_kwh"
                ] < 0.001
            ].copy()

            economic = add_economics(
                scenarios=valid,
                pv_cost_eur_kw=pv_cost,
                bess_cost_eur_kwh=bess_energy_cost,
                bess_cost_eur_kw=bess_power_cost,
                electricity_buy_price=buy_price,
                electricity_sell_price=sell_price,
                pv_opex_rate=pv_opex_pct / 100,
                bess_opex_rate=bess_opex_pct / 100,
                discount_rate=discount_rate_pct / 100,
                project_years=int(project_years),
                annual_load_kwh=base_hourly["load_kwh"].sum()
            )

            st.session_state[
                "optimization_results"
            ] = economic

            acceptable = economic[
                economic[
                    "self_sufficiency_pct"
                ] >= target_ssr
            ]

            if len(acceptable) > 0:

                best = acceptable.loc[
                    acceptable[
                        "npc_eur"
                    ].idxmin()
                ]

                st.session_state[
                    "best_scenario"
                ] = best

            else:

                st.session_state[
                    "best_scenario"
                ] = None


# ============================================================
# 13. REZULTATŲ SKIRTUKAS
# ============================================================

with tab_results:

    st.subheader(
        "Rezultatai"
    )

    # ========================================================
    # VIENO SCENARIJAUS REZULTATAI
    # ========================================================

    if "scenario_summary" in st.session_state:

        summary = st.session_state[
            "scenario_summary"
        ]

        hourly_result = st.session_state[
            "scenario_hourly"
        ]

        st.markdown(
            "## Pasirinktas scenarijus"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "PV generacija",
            f"{summary['pv_generation_kwh'] / 1000:.1f} MWh"
        )

        c2.metric(
            "Importas",
            f"{summary['grid_import_kwh'] / 1000:.1f} MWh"
        )

        c3.metric(
            "Savarankiškumas",
            f"{summary['self_sufficiency_pct']:.1f} %"
        )

        c4.metric(
            "PV panaudojimas",
            f"{summary['self_consumption_pct']:.1f} %"
        )

        c5, c6, c7, c8 = st.columns(4)

        c5.metric(
            "Eksportas",
            f"{summary['export_kwh'] / 1000:.1f} MWh"
        )

        c6.metric(
            "BESS iškrovimas",
            f"{summary['bess_discharge_kwh'] / 1000:.1f} MWh"
        )

        c7.metric(
            "BESS ciklai",
            f"{summary['equivalent_cycles']:.0f}"
        )

        c8.metric(
            "PV apribojimas",
            f"{summary['curtailment_kwh'] / 1000:.1f} MWh"
        )

        # ----------------------------------------------------
        # VARTOJIMO DUOMENŲ PATIKRA
        # ----------------------------------------------------

        if (
            load_df is not None
            and actual_load_total is not None
        ):

            st.markdown(
                "### Vartojimo duomenų patikra"
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Suvartojimas faile",
                f"{actual_load_total / 1000:.1f} MWh"
            )

            c2.metric(
                "Intervalas",
                f"{dt_hours * 60:.0f} min."
            )

            c3.metric(
                "Didžiausia apkrova",
                f"{peak_load_kw:.1f} kW"
            )

            c4.metric(
                "Įrašų skaičius",
                f"{len(load_df):,}"
            )

            st.write(
                "Laikotarpis:",
                first_timestamp,
                "–",
                last_timestamp
            )

        # ----------------------------------------------------
        # MĖNESINIAI REZULTATAI
        # ----------------------------------------------------

        monthly = (
            hourly_result
            .set_index("datetime")
            [
                [
                    "load_kwh",
                    "pv_kwh",
                    "grid_import_kwh",
                    "export_kwh",
                    "bess_charge_kwh",
                    "bess_discharge_kwh"
                ]
            ]
            .resample("ME")
            .sum()
        )

        st.markdown(
            "### Mėnesinis energijos balansas"
        )

        chart_data = monthly[
            [
                "load_kwh",
                "pv_kwh",
                "grid_import_kwh"
            ]
        ]

        st.line_chart(
            chart_data
        )

        # ----------------------------------------------------
        # VIENOS DIENOS ANALIZĖ
        # ----------------------------------------------------

        st.markdown(
            "### Vienos dienos analizė"
        )

        min_date = hourly_result[
            "datetime"
        ].min().date()

        max_date = hourly_result[
            "datetime"
        ].max().date()

        selected_date = st.date_input(
            "Pasirinkite datą",
            value=min_date,
            min_value=min_date,
            max_value=max_date
        )

        day_data = hourly_result[
            hourly_result[
                "datetime"
            ].dt.date == selected_date
        ]

        if len(day_data) > 0:

            day_chart = day_data.set_index(
                "datetime"
            )[
                [
                    "load_kwh",
                    "pv_kwh",
                    "soc_kwh"
                ]
            ]

            st.line_chart(
                day_chart
            )

        # ----------------------------------------------------
        # ATSISIUNTIMAS
        # ----------------------------------------------------

        csv = (
            hourly_result
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            "Atsisiųsti valandinius rezultatus CSV",
            data=csv,
            file_name="PV_BESS_valandiniai_rezultatai.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "Dar nėra apskaičiuoto vieno scenarijaus."
        )

    # ========================================================
    # OPTIMIZAVIMO REZULTATAI
    # ========================================================

    if (
        "optimization_results"
        in st.session_state
    ):

        st.markdown(
            "---"
        )

        st.markdown(
            "## Optimizavimo rezultatai"
        )

        optimization_results = (
            st.session_state[
                "optimization_results"
            ]
        )

        if (
            "best_scenario"
            in st.session_state
            and
            st.session_state[
                "best_scenario"
            ]
            is not None
        ):

            best = st.session_state[
                "best_scenario"
            ]

            st.success(
                "Rastas reikalavimus atitinkantis optimalus variantas."
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Optimali PV galia",
                f"{best['pv_kw']:.0f} kW"
            )

            c2.metric(
                "Optimali BESS talpa",
                f"{best['bess_kwh']:.0f} kWh"
            )

            c3.metric(
                "Optimali BESS galia",
                f"{best['bess_kw']:.0f} kW"
            )

            c4.metric(
                "Savarankiškumas",
                f"{best['self_sufficiency_pct']:.1f} %"
            )

            c5, c6, c7, c8 = st.columns(4)

            c5.metric(
                "CAPEX",
                f"{best['total_capex_eur']:,.0f} €"
            )

            c6.metric(
                "NPC",
                f"{best['npc_eur']:,.0f} €"
            )

            c7.metric(
                "NPV",
                f"{best['npv_eur']:,.0f} €"
            )

            if pd.notna(
                best["simple_payback_years"]
            ):

                payback_text = (
                    f"{best['simple_payback_years']:.1f} m."
                )

            else:

                payback_text = "–"

            c8.metric(
                "Atsipirkimas",
                payback_text
            )

        else:

            st.warning(
                "Nė vienas scenarijus nepasiekė "
                "pasirinkto savarankiškumo reikalavimo."
            )

        # ====================================================
        # TOP 20
        # ====================================================

        st.markdown(
            "### 20 mažiausio NPC scenarijų"
        )

        top20 = (
            optimization_results
            .sort_values(
                "npc_eur"
            )
            .head(20)
            [
                [
                    "pv_kw",
                    "bess_kwh",
                    "bess_kw",
                    "bess_duration_h",
                    "self_sufficiency_pct",
                    "self_consumption_pct",
                    "grid_import_kwh",
                    "export_kwh",
                    "total_capex_eur",
                    "npc_eur",
                    "npv_eur",
                    "simple_payback_years"
                ]
            ]
            .round(2)
        )

        st.dataframe(
            top20,
            use_container_width=True
        )

        optimization_csv = (
            optimization_results
            .to_csv(
                index=False
            )
            .encode("utf-8")
        )

        st.download_button(
            "Atsisiųsti visus optimizavimo rezultatus CSV",
            data=optimization_csv,
            file_name="PV_BESS_optimizavimo_rezultatai.csv",
            mime="text/csv"
        )


# ============================================================
# 14. DUOMENŲ KOKYBĖS INFORMACIJA
# ============================================================

st.sidebar.markdown(
    "---"
)

st.sidebar.subheader(
    "Duomenų būsena"
)

if pv_files:

    st.sidebar.success(
        "PV profilis: įkeltas"
    )

else:

    st.sidebar.warning(
        "PV profilis: neįkeltas"
    )

if load_source == "Modeliuotas profilis":

    st.sidebar.warning(
        "Vartojimas: modeliuotas"
    )

elif load_file is not None:

    st.sidebar.success(
        "Vartojimas: faktinis"
    )

else:

    st.sidebar.warning(
        "Vartojimas: neįkeltas"
    )

st.sidebar.info(
    "Ekonominės prielaidos šiuo metu "
    "įvedamos vartotojo ir turi būti "
    "patikrintos pagal konkretaus projekto duomenis."
)

