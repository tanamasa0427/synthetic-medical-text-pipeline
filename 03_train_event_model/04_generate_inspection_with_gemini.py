# ==============================================================
# 04_generate_inspection_with_gemini_safe_v25.py
# 安定版: 小バッチ処理 + キャッシュ + 自動保存
# ==============================================================

import os
import json
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from functools import lru_cache

# --------------------------------------------------------------
# Gemini 初期化
# --------------------------------------------------------------
USE_GEMINI = True
try:
    import google.generativeai as genai
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel(model_name="models/gemini-2.5-pro")
    print("✅ Geminiモデル読み込み完了")
except Exception as e:
    print("⚠️ Gemini初期化に失敗:", e)
    USE_GEMINI = False


# --------------------------------------------------------------
# パス設定
# --------------------------------------------------------------
BASE_DIR = "/content/synthetic-medical-text-pipeline"
INPUT_DIR = f"{BASE_DIR}/data/inputs"
OUTPUT_DIR = f"{INPUT_DIR}/outputs"
SCHEMA_PATH = f"{BASE_DIR}/data/schema/clinical_schema_v1.2.json"
VALUE_RANGES_PATH = f"{BASE_DIR}/data/schema/value_ranges.json"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------------------
# データ読み込み
# --------------------------------------------------------------
def load_csv(name):
    df = pd.read_csv(f"{INPUT_DIR}/{name}")
    print(f"✅ {name}: {len(df):,}件, 列: {list(df.columns)}")
    return df

gender_df = load_csv("gender.csv")
disease_df = load_csv("disease.csv")
drug_df = load_csv("drug.csv")


# --------------------------------------------------------------
# スキーマ読み込み
# --------------------------------------------------------------
with open(SCHEMA_PATH, "r") as f:
    schema = json.load(f)
print(f"✅ スキーマ読込完了: {SCHEMA_PATH}")

if os.path.exists(VALUE_RANGES_PATH):
    with open(VALUE_RANGES_PATH, "r") as f:
        value_ranges = json.load(f)
else:
    value_ranges = {}
    print("⚠️ value_ranges.json が見つかりません。AI生成で代用します。")


# --------------------------------------------------------------
# データ統合
# --------------------------------------------------------------
merged = (
    disease_df.merge(drug_df, on="patient_id", how="inner", suffixes=("_dis", "_drug"))
    .merge(gender_df, on="patient_id", how="left")
)
merged["disease_date"] = pd.to_datetime(merged["disease_date"], errors="coerce")
print(f"🧩 統合データ件数: {len(merged):,}")

# テストモードで制限
merged = merged.sample(n=200, random_state=42)
print(f"🔍 サンプルデータ件数: {len(merged):,}")


# --------------------------------------------------------------
# Gemini呼び出し（キャッシュ付き）
# --------------------------------------------------------------
@lru_cache(maxsize=None)
def get_tests_from_gemini_cached(disease, drug, gender="不明"):
    if not USE_GEMINI:
        return tuple()
    try:
        prompt = f"""
疾患「{disease}」と薬剤「{drug}」を使用している{gender}の患者に対して、
臨床的に関連性の高い検査項目を3つ挙げてください。
日本語または英語の検査名をカンマ区切りで短く返してください。
"""
        response = model.generate_content([prompt])
        text = response.text.strip().replace("\n", "").replace("、", ",")
        tests = [t.strip() for t in text.split(",") if len(t.strip()) > 0]
        return tuple(tests[:5])
    except Exception as e:
        print("⚠️ Gemini呼び出し失敗:", e)
        return tuple()


# --------------------------------------------------------------
# 検査値生成
# --------------------------------------------------------------
def generate_value_and_unit(test_name):
    if test_name in value_ranges:
        info = value_ranges[test_name]
        mean = info.get("mean", 1)
        sd = info.get("sd", 0.1)
        unit = info.get("unit", "")
        return round(np.random.normal(mean, sd), 2), unit
    return round(np.random.uniform(0.1, 10.0), 2), ""


# --------------------------------------------------------------
# 小バッチ処理
# --------------------------------------------------------------
BATCH_SIZE = 5
out_rows = []
total = len(merged)
now_str = datetime.now().strftime("%Y%m%d_%H%M")

for i in range(0, total, BATCH_SIZE):
    batch = merged.iloc[i:i+BATCH_SIZE]
    print(f"🚀 バッチ {i//BATCH_SIZE+1}: {len(batch)} 件処理中...")

    for _, row in batch.iterrows():
        disease = row.get("disease_name", "")
        drug = row.get("drug_name", "")
        gender = row.get("gender", "不明")
        patient_id = row["patient_id"]

        tests = get_tests_from_gemini_cached(disease, drug, gender)
        if len(tests) == 0:
            continue

        disease_date = row["disease_date"]
        if pd.isna(disease_date):
            continue

        inspection_date = disease_date + timedelta(days=int(np.random.choice([-1, 0, 1])))
        encounter_id = f"{patient_id}_{inspection_date.strftime('%Y%m%d')}"

        for test_name in tests:
            value, unit = generate_value_and_unit(test_name)
            out_rows.append({
                "patient_id": patient_id,
                "encounter_id": encounter_id,
                "disease_name": disease,
                "drug_name": drug,
                "inspection_name": test_name,
                "inspection_value": value,
                "unit": unit,
                "inspection_date": inspection_date.strftime("%Y-%m-%d"),
            })

    # バッチごとに保存
    if i % (BATCH_SIZE * 5) == 0 and len(out_rows) > 0:
        out_df = pd.DataFrame(out_rows)
        out_path = f"{OUTPUT_DIR}/inspection_partial_{now_str}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"💾 部分保存: {out_path} ({len(out_rows):,}件)")
        time.sleep(1)  # API負荷軽減

print("✅ 全処理完了！")
out_df = pd.DataFrame(out_rows)
out_path = f"{OUTPUT_DIR}/inspection_generated_final_{now_str}.csv"
out_df.to_csv(out_path, index=False)
print(f"🎉 出力完了: {out_path} / {len(out_df):,}件")
