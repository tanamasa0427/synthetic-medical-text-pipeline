# ===============================================
# generate_from_prescription.py (v4.4)
# ===============================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # TensorFlow警告OFF
os.environ["GRPC_VERBOSITY"] = "ERROR"    # gRPCログOFF
os.environ["GLOG_minloglevel"] = "3"      # Gemini内部ログOFF

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import google.generativeai as genai
import time
import re
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========= 設定 =========
MODEL = "models/gemini-2.5-pro"   # 有料APIならこのモデル。無料なら gemini-1.5-flash
PARALLEL = 10                     # 並列スレッド数
BATCH_SLEEP = 2.0                 # バッチ間スリープ
RETRY_LIMIT = 2                   # 再試行回数

INPUT_PRESCRIPTION = "../data/inputs/prescription.csv"
INPUT_EMR_TEMPLATE = "../data/inputs/emr.csv"
OUTPUT_PATH = "../data/outputs"

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ========= 文体テンプレート構築 =========
def build_emr_style_templates(path):
    df = pd.read_csv(path)
    templates = {}
    for dept, sub in df.groupby("department"):
        texts = " ".join(sub["emr_text"].astype(str).tolist())
        texts = re.sub(r"\s+", " ", texts)
        templates[dept] = texts[:1500]  # 診療科別で1500文字まで参照
    return templates

# ========= Gemini呼び出し =========
def call_gemini(prompt, model=MODEL):
    for i in range(RETRY_LIMIT):
        try:
            response = genai.GenerativeModel(model).generate_content(prompt)
            if response and hasattr(response, "text"):
                return response.text.strip()
        except Exception as e:
            msg = str(e)
            if "429" in msg or "quota" in msg:
                wait = 18 * (i + 1)
                print(f"⚠️ レート制限: {wait}s 待機して再試行 ({i+1}/{RETRY_LIMIT})")
                time.sleep(wait)
                continue
            else:
                print(f"⚠️ Gemini出力エラー: {msg}")
                break
    return ""

# ========= プロンプト生成 =========
def build_prompt(pid, date, df, dept, style_text):
    meds = ", ".join(sorted(df["drug_name"].dropna().unique()))
    prompt = f"""
あなたは臨床医です。以下の患者情報と処方履歴に基づき、診療録（カルテ）を日本語で作成してください。

---患者情報---
患者ID: {pid}
診療科: {dept}
診療日: {date}

---当日の処方---
{meds}

---文体ルール---
- emr.csvの文体を再現し、臨床医が記録するカルテ形式とする。
- 「〜と思われる」「〜と考える」「〜と評価する」などの推測語は使用しない。
- 判断は明示する（例：「高血圧症の診断」「感冒に伴う発熱を認める」など）。
- 一人称は避け、主観的・感情的表現は使わない。
- S/O/A/Pの各要素を明確に記載し、簡潔にまとめる。
- 初診では主要な診断名・方針を明示し、再診では経過・変更を中心に記載。
- 薬剤の継続は「現行処方継続」、中止は「○○中止」とする。

---診療科の文体参考---
{style_text[:800]}

# 出力形式
S: （患者申告・自覚症状）
O: （身体所見・検査結果）
A: （診断・臨床的判断）
P: （治療方針・処方・次回予定）
"""
    return prompt.strip()

# ========= メイン処理 =========
def main():
    print(f"📚 文体テンプレート構築: ", end="")
    emr_styles = build_emr_style_templates(INPUT_EMR_TEMPLATE)
    print(f"{len(emr_styles)}診療科")

    df = pd.read_csv(INPUT_PRESCRIPTION)
    df["dispense_date"] = pd.to_datetime(df["dispense_date"])
    df.sort_values(["patient_id", "dispense_date"], inplace=True)

    total_patients = df["patient_id"].nunique()
    print(f"\n📄 処方レコード: {len(df)}")
    print(f"👩‍⚕️ 対象患者数: {total_patients}")
    print(f"🧠 モデル: {MODEL} | 並列: {PARALLEL} | SLEEP: {BATCH_SLEEP}s | RETRY: {RETRY_LIMIT}\n")

    results = []

    patients = df["patient_id"].unique()
    with tqdm(total=len(patients), desc="全体進捗（患者単位）") as pbar:
        for pid in patients:
            pdf = df[df["patient_id"] == pid].sort_values("dispense_date")
            dept = pdf["department"].iloc[0] if "department" in pdf.columns else "一般"
            style_text = emr_styles.get(dept, "")
            records = []

            with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
                futures = []
                for date, day_df in pdf.groupby("dispense_date"):
                    prompt = build_prompt(pid, date.date(), day_df, dept, style_text)
                    futures.append(ex.submit(call_gemini, prompt))

                for future in as_completed(futures):
                    text = future.result()
                    if text:
                        records.append(text)

            for (date, _), text in zip(pdf.groupby("dispense_date"), records):
                results.append({
                    "patient_id": pid,
                    "age_group": pdf["age_group"].iloc[0] if "age_group" in pdf.columns else "",
                    "sex": pdf["sex"].iloc[0] if "sex" in pdf.columns else "",
                    "department": dept,
                    "dispense_date": date.date(),
                    "note_type": "診療録(医師)",
                    "emr_text": text,
                    "visit_type": "外来"
                })

            pbar.update(1)
            time.sleep(BATCH_SLEEP)

    # 出力
    out_name = f"emr_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    out_path = os.path.join(OUTPUT_PATH, out_name)
    pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n📦 最終出力: {out_path}（{len(results)}件）")
    print("🎉 完了")

if __name__ == "__main__":
    main()
