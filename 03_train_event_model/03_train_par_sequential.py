# ==========================================
# ✅ PARSynthesizer (時系列) + 軽量版
# (Kaggle /working/ ディレクトリ対応)
# [2025-10-23 最終修正版：学習パラメータを .fit() に移動]
# ==========================================
import os
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm
from sdv.sequential import PARSynthesizer
from sdv.metadata import SingleTableMetadata

# ------------------------------------------
# ⚙️ 実行環境に合わせて、以下のパスを設定
# ------------------------------------------
ENVIRONMENT = "kaggle_working" 

if ENVIRONMENT == "kaggle_working":
    BASE_DIR = "/kaggle/working/synthetic-medical-text-pipeline"
    INPUT_DIR = os.path.join(BASE_DIR, "data/inputs")
    OUTPUT_DIR = os.path.join(BASE_DIR, "data/outputs")
elif ENVIRONMENT == "colab":
    BASE_DIR = "/content/synthetic-medical-text-pipeline"
    INPUT_DIR = os.path.join(BASE_DIR, "data/inputs")
    OUTPUT_DIR = os.path.join(BASE_DIR, "data/outputs")
else:
    INPUT_DIR = "./data/inputs"
    OUTPUT_DIR = "./data/outputs"

# ------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"📦 SDV PARSynthesizer (時系列) パイプライン | {datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"🔩 環境: {ENVIRONMENT}")
print(f"📁 入力: {INPUT_DIR}")
print(f"💾 出力: {OUTPUT_DIR}")

def read_csv_safe(path):
    try:
        df = pd.read_csv(path)
        print(f"✅ {os.path.basename(path)}: {len(df):,} 件, 列: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"⚠️ {os.path.basename(path)} の読込失敗: {e}")
        return pd.DataFrame()

# 1. データの読み込み
gender_df = read_csv_safe(os.path.join(INPUT_DIR, "gender.csv")) 
disease_df = read_csv_safe(os.path.join(INPUT_DIR, "disease.csv"))
inspection_df = read_csv_safe(os.path.join(INPUT_DIR, "inspection.csv"))
drug_df = read_csv_safe(os.path.join(INPUT_DIR, "drug.csv"))
emr_df = read_csv_safe(os.path.join(INPUT_DIR, "emr.csv"))

def normalize(df, date_col, type_label):
    df = df.copy()
    if date_col in df.columns:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df["event_type"] = type_label
    return df

disease_df = normalize(disease_df, "disease_date", "disease")
inspection_df = normalize(inspection_df, "inspection_date", "inspection")
drug_df = normalize(drug_df, "key_date", "drug")
emr_df = normalize(emr_df, "emr_date", "emr")

# 2. データの前処理 (PARSynthesizer 用)
if "emr_text" in emr_df.columns:
    emr_df = emr_df.drop(columns=["emr_text"])

merged_df = pd.concat([disease_df, inspection_df, drug_df, emr_df], ignore_index=True)

drop_cols = [
    "inspection_value", "drug_name", "icd10_code", "disease_name",
    "yj_code", "inspection_name", "amount", "days_count",
    "extracted_number", "daily_dosage"
]
print(f"🗑️ 以下の高コスト列を除外します: {drop_cols}")
merged_df = merged_df.drop(columns=[c for c in drop_cols if c in merged_df.columns], errors='ignore')

merged_df["date"] = pd.to_datetime(merged_df["date"], errors="coerce")
original_count = len(merged_df)
merged_df = merged_df.dropna(subset=['date'])
print(f"🗓️ 日付(date)が NaT の {original_count - len(merged_df):,} 件を除外しました。")

date_cols = ["disease_date", "inspection_date", "key_date", "emr_date"]
merged_df = merged_df.drop(columns=[c for c in date_cols if c in merged_df.columns], errors='ignore')

if 'age' in merged_df.columns:
    merged_df['age'] = pd.
