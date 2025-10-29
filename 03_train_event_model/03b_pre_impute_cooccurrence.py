import os
import pandas as pd
import numpy as np
from datetime import datetime

# ======================================================
# 設定
# ======================================================
BASE_DIR = "/content/synthetic-medical-text-pipeline"
INPUT_DIR = os.path.join(BASE_DIR, "data/inputs")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"inspection_filled_03b_{timestamp}.csv")

# ======================================================
# データ読み込み
# ======================================================
print("📥 データ読み込み中...")
gender_df = pd.read_csv(os.path.join(INPUT_DIR, "gender.csv"))
disease_df = pd.read_csv(os.path.join(INPUT_DIR, "disease.csv"))
drug_df = pd.read_csv(os.path.join(INPUT_DIR, "drug.csv"))

print(f"✅ gender.csv: {len(gender_df):,}件, 列: {list(gender_df.columns)}")
print(f"✅ disease.csv: {len(disease_df):,}件, 列: {list(disease_df.columns)}")
print(f"✅ drug.csv: {len(drug_df):,}件, 列: {list(drug_df.columns)}")

# ======================================================
# 結合（patient_idキー）
# ======================================================
merged = (
    disease_df[["patient_id", "disease_name"]]
    .merge(drug_df[["patient_id", "drug_name"]], on="patient_id", how="left")
    .merge(gender_df[["patient_id", "gender"]], on="patient_id", how="left")
)
merged.dropna(subset=["patient_id"], inplace=True)
merged.fillna("不明", inplace=True)

print(f"🧩 統合データ件数: {len(merged):,}")

# ======================================================
# 疾患×薬剤×性別 → 検査共起マップ定義
# ======================================================
cooccurrence_map = {
    # 糖尿病関連
    ("糖尿病", "メトホルミン", "男"): ["HbA1c", "血糖", "eGFR"],
    ("糖尿病", "メトホルミン", "女"): ["HbA1c", "血糖", "eGFR"],
    ("糖尿病", "インスリン", "男"): ["HbA1c", "血糖", "Cr"],
    ("糖尿病", "インスリン", "女"): ["HbA1c", "血糖", "Cr"],

    # 高血圧
    ("高血圧", "アムロジピン", "男"): ["Na", "K", "Cr"],
    ("高血圧", "アムロジピン", "女"): ["Na", "K", "Ca"],
    ("高血圧", "ARB", "男"): ["Na", "K", "Cr"],
    ("高血圧", "ARB", "女"): ["Na", "K", "Cr"],

    # 高尿酸血症
    ("高尿酸血症", "フェブキソスタット", "男"): ["尿酸", "Cr"],
    ("高尿酸血症", "フェブキソスタット", "女"): ["尿酸", "Cr"],

    # 脂質異常症
    ("脂質異常症", "スタチン", "男"): ["LDL", "HDL", "TG"],
    ("脂質異常症", "スタチン", "女"): ["LDL", "HDL", "TG"],

    # 貧血
    ("貧血", "鉄剤", "女"): ["Hb", "Hct", "Fe"],

    # 心不全
    ("心不全", "利尿薬", "男"): ["BNP", "K", "Na"],
    ("心不全", "利尿薬", "女"): ["BNP", "K", "Na"],
}

# ======================================================
# 検査値分布テーブル
# ======================================================
value_ranges = {
    "HbA1c": (6.0, 0.5, "%"),
    "血糖": (110, 25, "mg/dL"),
    "eGFR": (65, 20, "mL/min/1.73m2"),
    "Na": (140, 4, "mmol/L"),
    "K": (4.1, 0.3, "mmol/L"),
    "Cr": (0.9, 0.2, "mg/dL"),
    "Ca": (9.2, 0.5, "mg/dL"),
    "尿酸": (5.5, 1.0, "mg/dL"),
    "LDL": (110, 25, "mg/dL"),
    "HDL": (55, 10, "mg/dL"),
    "TG": (100, 40, "mg/dL"),
    "Hb": (13.5, 1.2, "g/dL"),
    "Hct": (40, 4, "%"),
    "Fe": (90, 30, "µg/dL"),
    "BNP": (75, 50, "pg/mL"),
}

# ======================================================
# 検査データ生成
# ======================================================
generated_records = []

for _, row in merged.iterrows():
    disease = row["disease_name"]
    drug = row["drug_name"]
    gender = row["gender"]

    # 該当共起を検索（部分一致許容）
    match_keys = [k for k in cooccurrence_map.keys() if k[0] in disease and k[1] in drug and k[2] in gender]
    if not match_keys:
        continue

    inspections = cooccurrence_map[match_keys[0]]
    for insp in inspections:
        if insp not in value_ranges:
            continue
        mean, sd, unit = value_ranges[insp]
        value = np.round(np.random.normal(mean, sd), 2)
        generated_records.append({
            "patient_id": row["patient_id"],
            "gender": gender,
            "disease_name": disease,
            "drug_name": drug,
            "inspection_name": insp,
            "inspection_value": value,
            "unit": unit
        })

# ======================================================
# 保存
# ======================================================
out_df = pd.DataFrame(generated_records)
out_df.to_csv(OUTPUT_PATH, index=False)

print(f"💾 補完検査データ保存完了: {OUTPUT_PATH}")
print(f"🧾 生成件数: {len(out_df):,}")
print("🎉 Step03b（共起ベース検査補完）完了！")
