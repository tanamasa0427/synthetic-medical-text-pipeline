# ==============================================================
# 04_generate_inspection_with_gemini.py
# 疾患・薬剤データからGemini APIで検査項目と値を生成するスクリプト
# ==============================================================

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Gemini API利用（Colab環境では pip install google-generativeai が必要）
USE_GEMINI = True
try:
    import google.generativeai as genai
except ImportError:
    USE_GEMINI = False

# --------------------------------------------------------------
# 設定
# --------------------------------------------------------------
BASE_DIR = "/content/synthetic-medical-text-pipeline"
INPUT_DIR = f"{BASE_DIR}/data/inputs"
OUTPUT_DIR = f"{INPUT_DIR}/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------------
# Gemini 初期化
# --------------------------------------------------------------
if USE_GEMINI:
    try:
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-pro")
        print("✅ Geminiモデル読み込み完了")
    except Exception as e:
        print("⚠️ Gemini初期化に失敗:", e)
        USE_GEMINI = False

# --------------------------------------------------------------
# データ読み込み
# --------------------------------------------------------------
def load_csv(name):
    path = f"{INPUT_DIR}/{name}"
    df = pd.read_csv(path)
    print(f"✅ {name}: {len(df):,}件, 列: {list(df.columns)}")
    return df

gender_df = load_csv("gender.csv")
disease_df = load_csv("disease.csv")
drug_df = load_csv("drug.csv")

# clinical schema（encounter構造）
with open(f"{INPUT_DIR}/clinical_schema_v1.2.json", "r") as f:
    schema = json.load(f)

# 検査値分布
value_ranges_path = f"{INPUT_DIR}/value_ranges.json"
if os.path.exists(value_ranges_path):
    with open(value_ranges_path, "r") as f:
        value_ranges = json.load(f)
else:
    value_ranges = {}

# --------------------------------------------------------------
# データ統合
# --------------------------------------------------------------
merged = (
    disease_df.merge(drug_df, on="patient_id", how="inner", suffixes=("_dis", "_drug"))
    .merge(gender_df, on="patient_id", how="left")
)
merged["disease_date"] = pd.to_datetime(merged["disease_date"], errors="coerce")
merged["key_date"] = pd.to_datetime(merged["key_date"], errors="coerce")
print(f"🧩 統合データ件数: {len(merged):,}")

# --------------------------------------------------------------
# Geminiで検査候補を推定
# --------------------------------------------------------------
def get_tests_from_gemini(disease, drug, gender="不明"):
    if not USE_GEMINI:
        return []
    try:
        prompt = f"""
疾患「{disease}」と薬剤「{drug}」を使用している{gender}の患者に対して、
臨床的に関連性の高い検査項目を3つ挙げてください。
一般的な検査名（例：HbA1c, eGFR, AST, LDL-Cなど）で出力してください。
日本語または英語の検査名をカンマ区切りで短く返してください。
"""
        response = model.generate_content(prompt)
        text = response.text.strip().replace("\n", "").replace("、", ",")
        tests = [t.strip() for t in text.split(",") if len(t.strip()) > 0]
        return tests[:5]
    except Exception as e:
        print("⚠️ Gemini呼び出し失敗:", e)
        return []

# --------------------------------------------------------------
# 検査値生成（AI or 既知分布）
# --------------------------------------------------------------
def generate_value_and_unit(test_name):
    if test_name in value_ranges:
        info = value_ranges[test_name]
        mean = info.get("mean", 1)
        sd = info.get("sd", 0.1)
        unit = info.get("unit", "")
        value = np.round(np.random.normal(mean, sd), 2)
        return value, unit

    if USE_GEMINI:
        try:
            prompt = f"検査「{test_name}」の正常範囲と単位を簡潔に1行で教えてください。"
            res = model.generate_content(prompt)
            text = res.text.strip()
            # 正規化（簡易抽出）
            value = np.random.normal(1, 0.1)
            unit = ""
            for token in ["mg/dL", "mmol/L", "U/L", "%", "mL/min", "g/dL"]:
                if token in text:
                    unit = token
            return round(float(abs(value)), 2), unit
        except:
            pass
    return np.round(np.random.uniform(0.1, 10.0), 2), ""

# --------------------------------------------------------------
# encounter構造に基づく出力生成
# --------------------------------------------------------------
out_rows = []
for i, row in merged.iterrows():
    disease = row.get("disease_name", "")
    drug = row.get("drug_name", "")
    gender = row.get("gender", "不明")
    patient_id = row["patient_id"]

    tests = get_tests_from_gemini(disease, drug, gender)
    if len(tests) == 0:
        continue

    disease_date = row["disease_date"]
    if pd.isna(disease_date):
        continue

    # 検査日は±1日
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

# --------------------------------------------------------------
# 出力保存
# --------------------------------------------------------------
out_df = pd.DataFrame(out_rows)
now_str = datetime.now().strftime("%Y%m%d_%H%M")
out_path = f"{OUTPUT_DIR}/inspection_generated_gemini_{now_str}.csv"
out_df.to_csv(out_path, index=False)
print(f"💾 出力完了: {out_path}")
print(f"🧾 生成件数: {len(out_df):,}")
