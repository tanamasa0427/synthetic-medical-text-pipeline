# ==============================================================
# 04_generate_inspection_with_gemini_v25.py
# 疾患・薬剤データから Gemini 2.5 API を用いて検査項目を生成するスクリプト
# ==============================================================

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --------------------------------------------------------------
# Gemini 初期化（v1対応）
# --------------------------------------------------------------
USE_GEMINI = True
try:
    import google.generativeai as genai

    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

    # ✅ 最新モデルを指定（精度重視 or 速度重視）
    model = genai.GenerativeModel(model_name="models/gemini-2.5-pro")
    # model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")  # ←高速モードに切り替える場合はこちら

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
# データ読み込み関数
# --------------------------------------------------------------
def load_csv(name):
    """CSVをロードして情報を表示"""
    path = f"{INPUT_DIR}/{name}"
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ {path} が存在しません。")
    df = pd.read_csv(path)
    print(f"✅ {name}: {len(df):,}件, 列: {list(df.columns)}")
    return df


# --------------------------------------------------------------
# 入力データの読み込み
# --------------------------------------------------------------
gender_df = load_csv("gender.csv")
disease_df = load_csv("disease.csv")
drug_df = load_csv("drug.csv")

# --------------------------------------------------------------
# スキーマ・検査値範囲ファイル読み込み
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
merged["key_date"] = pd.to_datetime(merged["key_date"], errors="coerce")

print(f"🧩 統合データ件数: {len(merged):,}")

# --------------------------------------------------------------
# Geminiで検査候補を推定
# --------------------------------------------------------------
def get_tests_from_gemini(disease, drug, gender="不明"):
    """疾患・薬剤・性別をもとに関連検査項目をGeminiで推定"""
    if not USE_GEMINI:
        return []

    try:
        prompt = f"""
疾患「{disease}」と薬剤「{drug}」を使用している{gender}の患者に対して、
臨床的に関連性の高い検査項目を3つ挙げてください。
一般的な検査名（例：HbA1c, eGFR, AST, LDL-Cなど）で出力してください。
日本語または英語の検査名をカンマ区切りで短く返してください。
"""
        response = model.generate_content([prompt])
        text = response.text.strip().replace("\n", "").replace("、", ",")
        tests = [t.strip() for t in text.split(",") if len(t.strip()) > 0]
        return tests[:5]
    except Exception as e:
        print("⚠️ Gemini呼び出し失敗:", e)
        return []

# --------------------------------------------------------------
# 検査値生成（既知分布 or AI推定）
# --------------------------------------------------------------
def generate_value_and_unit(test_name):
    """検査項目名に基づいて値と単位を生成"""
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
            res = model.generate_content([prompt])
            text = res.text.strip()
            unit = ""
            for token in ["mg/dL", "mmol/L", "U/L", "%", "mL/min", "g/dL"]:
                if token in text:
                    unit = token
            value = np.random.normal(1, 0.1)
            return round(float(abs(value)), 2), unit
        except:
            pass

    # フォールバック
    return np.round(np.random.uniform(0.1, 10.0), 2), ""

# --------------------------------------------------------------
# 検査データ生成
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
