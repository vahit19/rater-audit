"""DICES-350 (Aroyo et al., 2023; CC-BY-4.0): 350 adversarial chatbot conversations, each rated for safety by all
123 raters, with an expert gold safety label per conversation. Label 1 = unsafe ('Yes'), 0 = safe ('No');
'Unsure' is treated as missing. The dataset authors' own list of raters removed by their quality checks is kept
for comparison."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data" / "dices_350.csv"
URL = "https://raw.githubusercontent.com/google-research-datasets/dices-dataset/main/350/diverse_safety_adversarial_dialog_350.csv"
AUTHORS_REMOVED = ['297514565398139', '297515609163939', '297515750682315', '297515617432733', '297541515566649',
                   '297541515769980', '297515629971478', '297059995361243', '297541522412126', '297540556928761',
                   '297541321453321', '297540562350921', '297540983991638', '297060365288109', '297514543980607',
                   '297515729806999', '297541271027233', '296709611112092', '296709543131761']
MAIN_TYPES = ["Racial", "Political", "Gendered & Sexist"]


def load():
    """Returns (matrix items x raters with 1/0/NaN, items table with gold/type/text, raters table)."""
    if not DATA.exists():
        import urllib.request
        urllib.request.urlretrieve(URL, DATA)
    d = pd.read_csv(DATA, dtype={"rater_id": str})
    d["label"] = d["Q_overall"].map({"Yes": 1.0, "No": 0.0})
    items = d.drop_duplicates("item_id").set_index("item_id")[["context", "response", "safety_gold", "harm_type", "degree_of_harm"]]
    items["gold"] = (items["safety_gold"] == "Yes").astype(int)
    items["type"] = np.where(items["harm_type"].isin(MAIN_TYPES), items["harm_type"], "Other")
    m = d.pivot_table(index="item_id", columns="rater_id", values="label", aggfunc="first").reindex(items.index)
    raters = d.groupby("rater_id").agg(median_time_ms=("answer_time_ms", "median")).reindex(m.columns)
    raters["authors_removed"] = raters.index.isin(AUTHORS_REMOVED)
    return m, items, raters
