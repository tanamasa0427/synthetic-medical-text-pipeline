# ===============================================
# 02_extract_structured_from_emr.py  v3.1 (Colab対応)
# -----------------------------------------------
# 最新の emr_combined_*.csv を自動検出し、
# disease / inspection / drug を抽出する。
# Google Colab / Drive 両対応
# ===============================================

import pandas as pd
import google.generativeai as genai
import os, json, re
from datetime import datetime

# ===== Gemini 設定 =====
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL = "models/gemini-2.5-flash"

# ===== Colab環境対応: ディレクトリ作成 =====
# （このスクリプトを /content/ または /content/drive/... 下で動作させる）
os.makedirs("../data/outputs", exist_ok=True)

# ===== 入出力パス =====
INPUT_DIR = "../data/outputs"

# 「emr_combined_*.csv」を自動検出
csv_files = [f for f in os.listdir(INPUT_DIR) if f.startswith("emr_combined_") and f.endswith(".csv")]
if not csv_files:
    raise FileNotFoundError(f"❌ emr_combined_*.csv が見つかりません。{os.path.abspath(INPUT_DIR)} を確認してください。")

latest_file = max(csv_files, key=lambda x: os.path.getmtime(os.path.join(INPUT_DIR, x)))
INPUT_FILE = os.path.join(INPUT_DIR, latest_file)
print(f"📄 入力ファイル: {INPUT_FILE}")

# 出力ファイルを日時付きで同じフォルダに保存
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_JSON = os.path.join(INPUT_DIR, f"structured_from_emr_{timestamp}.json")
OUTPUT_DISEASE = os.path.join(INPUT_DIR, f"disease_{timestamp}.csv")
OUTPUT_INSPECTION = os.path.join(INPUT_DIR, f"inspection_{timestamp}.csv")
OUTPUT_DRUG = os.path.join(INPUT_DIR, f"drug_{timestamp}.csv")

# ===== データ読み込み =====
df = pd.read_csv(INPUT_FILE)
text_column = [c for c in df.columns if "text" in c or "emr" in c][0]
print(f"✅ {len(df)} 件のカルテ文を読み込みました。")

# ===== プロンプト =====
PROMPT_TEMPLATE = """
以下は医師が記載した電子カルテ（診療録）です。
この文章から、疾患（disease）、検査（inspection）、薬剤（drug）に関する情報を抽出してください。

# 出力要件
- 出力は必ず **純粋なJSON形式のみ** で返す。
- JSON以外のコメント・文章・説明は絶対に出力しない。
- 各カテゴリが該当しない場合も、必ず空リストとして出力する。

# カルテ文体について
- 医師の診療録として、診断・検査結果・処方・経過を記載した文体です。
- 「患者は～」「前回と比較して～」「処方を継続する」などの文から文脈を判断。
- 一人称ではなく、医師の診療記録として正確に解析する。

# 抽出条件
- 疾患（disease）:
  - 病名・診断名・合併症・疑いなどを抽出。
  - ICD10コードが不明な場合は空欄。
- 検査（inspection）:
  - 数値・単位の有無に関わらず、検査項目を抽出。
  - 例: 「HbA1c 7.8」「AST 35」「LDL 110」「血糖値 120」など。
  - 単位がなければ空欄で良い。
- 薬剤（drug）:
  - 商品名または一般名を抽出。
  - 継続処方・中止・変更などは remarks に記載。
  - 数量や日数が不明な場合は空欄可。

# 出力例
{{
  "disease": [
    {{"disease_name": "2型糖尿病", "icd10_code": "E11", "disease_date": "{emr_date}", "department": "内科", "is_suspected": 0}}
  ],
  "inspection": [
    {{"inspection_item": "HbA1c", "inspection_value": 7.6, "unit": "%", "inspection_date": "{emr_date}", "department": "内科"}}
  ],
  "drug": [
    {{"drug_name": "メトグルコ錠250mg", "yj_code": "", "amount": 2, "unit": "錠", "days_count": 30, "remarks": "継続処方", "department": "内科"}}
  ]
}}

# 入力カルテ文
{emr_text}
"""

# ===== 出力格納用 =====
records = []

# ===== メイン処理ループ =====
for i, row in df.iterrows():
    emr_text = str(row[text_column])
    emr_date = row.get("dispense_date") or row.get("date") or ""
    pid = row.get("patient_id") or row.get("id") or f"patient_{i+1}"
    eid = row.get("encounter_id") or f"enc_{i+1}"

    if not emr_text.strip():
        continue

    prompt = PROMPT_TEMPLATE.format(emr_text=emr_text, emr_date=emr_date)
    try:
        model = genai.GenerativeModel(MODEL)
        response = model.generate_content(prompt)
        text = response.text.strip()

        json_text = re.search(r"\{[\s\S]*\}", text)
        if not json_text:
            print(f"⚠️ JSONが検出されませんでした ({i+1}行目)")
            continue

        data = json.loads(json_text.group(0))
        records.append({
            "patient_id": pid,
            "encounter_id": eid,
            "emr_date": emr_date,
            "output": data
        })
        print(f"✅ {i+1}件目抽出完了")

    except Exception as e:
        print(f"⚠️ 抽出エラー（{i+1}行目）: {e}")

# ===== JSON保存 =====
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)
print(f"💾 構造化データを保存: {OUTPUT_JSON}")

# ===== 各構造別CSV保存 =====
disease_rows, inspection_rows, drug_rows = [], [], []
for rec in records:
    pid = rec["patient_id"]
    eid = rec["encounter_id"]
    date = rec["emr_date"]
    data = rec["output"]

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
