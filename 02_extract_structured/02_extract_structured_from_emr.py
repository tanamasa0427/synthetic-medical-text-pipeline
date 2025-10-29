# ===============================================
# 02_extract_structured_from_emr_fast.py  v4.0
# -----------------------------------------------
# 並列処理対応・Gemini API最適化版
# ===============================================

import pandas as pd
import google.generativeai as genai
import os, json, re, time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===== Gemini 設定 =====
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL_NAME = "models/gemini-2.5-flash"
model = genai.GenerativeModel(MODEL_NAME)

# ===== 入出力設定 =====
INPUT_DIR = "../data/outputs"
os.makedirs(INPUT_DIR, exist_ok=True)

csv_files = [f for f in os.listdir(INPUT_DIR) if f.startswith("emr_combined_") and f.endswith(".csv")]
if not csv_files:
    raise FileNotFoundError(f"❌ emr_combined_*.csv が見つかりません。{os.path.abspath(INPUT_DIR)} を確認してください。")

latest_file = max(csv_files, key=lambda x: os.path.getmtime(os.path.join(INPUT_DIR, x)))
INPUT_FILE = os.path.join(INPUT_DIR, latest_file)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_JSON = os.path.join(INPUT_DIR, f"structured_from_emr_{timestamp}.json")
OUTPUT_DISEASE = os.path.join(INPUT_DIR, f"disease_{timestamp}.csv")
OUTPUT_INSPECTION = os.path.join(INPUT_DIR, f"inspection_{timestamp}.csv")
OUTPUT_DRUG = os.path.join(INPUT_DIR, f"drug_{timestamp}.csv")

df = pd.read_csv(INPUT_FILE)
text_column = [c for c in df.columns if "text" in c or "emr" in c][0]
print(f"✅ {len(df)} 件のカルテ文を読み込みました。")

# ===== プロンプトテンプレート =====
PROMPT_TEMPLATE = """
以下は医師が記載した電子カルテ（診療録）です。
この文章から、疾患（disease）、検査（inspection）、薬剤（drug）に関する情報を抽出してください。

# 出力要件
- 出力は必ず **純粋なJSON形式のみ**。
- JSON以外のコメント・説明は禁止。
- 各カテゴリが該当しない場合も空リストで出力。

# 出力例
{{
  "disease": [{{"disease_name": "2型糖尿病", "icd10_code": "E11", "disease_date": "{emr_date}", "department": "内科", "is_suspected": 0}}],
  "inspection": [{{"inspection_item": "HbA1c", "inspection_value": 7.6, "unit": "%", "inspection_date": "{emr_date}", "department": "内科"}}],
  "drug": [{{"drug_name": "メトグルコ錠250mg", "yj_code": "", "amount": 2, "unit": "錠", "days_count": 30, "remarks": "継続処方", "department": "内科"}}]
}}

# 入力カルテ文
{emr_text}
"""

# ===== 単一カルテ処理関数 =====
def process_record(i, row):
    emr_text = str(row[text_column]).strip()
    if not emr_text:
        return None

    emr_date = row.get("dispense_date") or row.get("date") or ""
    pid = row.get("patient_id") or row.get("id") or f"patient_{i+1}"
    eid = row.get("encounter_id") or f"enc_{i+1}"

    prompt = PROMPT_TEMPLATE.format(emr_text=emr_text, emr_date=emr_date)

    for attempt in range(3):  # 最大3回リトライ
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            json_text = re.search(r"\{[\s\S]*\}", text)
            if not json_text:
                raise ValueError("JSON検出不可")

            data = json.loads(json_text.group(0))
            return {
                "patient_id": pid,
                "encounter_id": eid,
                "emr_date": emr_date,
                "output": data
            }
        except Exception as e:
            if attempt == 2:
                print(f"⚠️ ({i+1}行目) 失敗: {e}")
            else:
                time.sleep(2)  # リトライ前待機

# ===== 並列実行 =====
MAX_WORKERS = min(10, len(df))  # 最大10スレッド
records = []

print("🚀 並列処理を開始します...")
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process_record, i, row): i for i, row in df.iterrows()}
    for future in as_completed(futures):
        result = future.result()
        if result:
            records.append(result)
            print(f"✅ {len(records)} 件完了 ({futures[future]+1}行目)")

# ===== JSON保存 =====
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)
print(f"💾 構造化データを保存: {OUTPUT_JSON}")

# ===== CSV分割保存 =====
disease_rows, inspection_rows, drug_rows = [], [], []
for rec in records:
    pid, eid, date, data = rec["patient_id"], rec["encounter_id"], rec["emr_date"], rec["output"]
    for d in data.get("disease", []):
        disease_rows.append({**d, "patient_id": pid, "encounter_id": eid, "source_date": date})
    for i in data.get("inspection", []):
        inspection_rows.append({**i, "patient_id": pid, "encounter_id": eid, "source_date": date})
    for dr in data.get("drug", []):
        drug_rows.append({**dr, "patient_id": pid, "encounter_id": eid, "source_date": date})

if disease_rows:
    pd.DataFrame(disease_rows).to_csv(OUTPUT_DISEASE, index=False, encoding="utf-8-sig")
if inspection_rows:
    pd.DataFrame(inspection_rows).to_csv(OUTPUT_INSPECTION, index=False, encoding="utf-8-sig")
if drug_rows:
    pd.DataFrame(drug_rows).to_csv(OUTPUT_DRUG, index=False, encoding="utf-8-sig")

print(f"🎉 抽出完了: 疾患 {len(disease_rows)}件 / 検査 {len(inspection_rows)}件 / 薬剤 {len(drug_rows)}件")
