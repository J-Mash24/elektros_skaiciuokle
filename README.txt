PV + BESS modelis v5

GitHub / Streamlit Cloud kataloge turi būti:
- app.py
- energy_model_core.py
- requirements.txt

Numatytasis PV režimas:
"Modeliuojamas PV profilis (nereikia failo)".

Tokiu režimu PV failo kelti nereikia. Modelis naudoja:
- vartojimo laiko eilutę (faktinę arba modeliuojamą);
- vartotojo nurodytą metinę specifinę PV generaciją;
- į modelį įrašytą mėnesinį sezoniškumą;
- astronominę dienos/nakties trukmę pagal Kairių / Šiaulių vietovę.

Jei norite naudoti faktinį referencinės PV elektrinės profilį, pasirinkite
"Referencinės PV elektrinės failai" ir įkelkite Plant Report failus.
