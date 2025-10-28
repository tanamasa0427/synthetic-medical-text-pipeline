# ← ここから下に、修正版のスクリプト（全コード）を貼り付け
# ==========================================
# ✅ Colab用：datetime欠損対応版（emr_text除外済み）
# ファイル名: 03_train_event_model_no_emrtext_v3.py
# ==========================================

import os
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata

# ==========================================
# 初期設定
# ==========================================
BASE_DIR = "/content/synthetic-medical-text-pipeline"
INPUT_DIR = os.path.join(BASE_DIR, "data/inputs")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"📦 SDV CTGAN パイプライン | {datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"📁 入力: {INPUT_DIR}")
print(f"💾 出力: {OUTPUT_DIR}")

# ==========================================
# データ読込関数
# ==========================================
def read_csv_safe(path):
    try:
        df = pd.read_csv(path)
        print(f"✅ {os.path.basename(path)}: {len(df):,} 件, 列: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"⚠️ {os.path.basename(path)} の読込失敗: {e}")
        return pd.DataFrame()

# ==========================================
# データ読込
# ==========================================
gender_df = read_csv_safe(os.path.join(INPUT_DIR, "gender.csv"))
disease_df = read_csv_safe(os.path.join(INPUT_DIR, "disease.csv"))
inspection_df = read_csv_safe(os.path.join(INPUT_DIR, "inspection.csv"))
drug_df = read_csv_safe(os.path.join(INPUT_DIR, "drug.csv"))
emr_df = read_csv_safe(os.path.join(INPUT_DIR, "emr.csv"))  # emr_textは除外対象

# ==========================================
# 統合と標準化
# ==========================================
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

# ❌ emr_textを除外
if "emr_text" in emr_df.columns:
    emr_df = emr_df.drop(columns=["emr_text"])

# 統合
merged_df = pd.concat([disease_df, inspection_df, drug_df, emr_df], ignore_index=True)

# ==========================================
# 日付の欠損処理 & 文字列化
# ==========================================
date_cols = ["disease_date", "inspection_date", "key_date", "emr_date", "date"]
for col in date_cols:
    if col in merged_df.columns:
        merged_df[col] = merged_df[col].fillna("1970-01-01")
        merged_df[col] = merged_df[col].astype(str)

# ==========================================
# 型変換：CTGAN互換対応
# ==========================================
for col in merged_df.columns:
    if merged_df[col].dtype.name == "category":
        merged_df[col] = merged_df[col].astype(str)
    elif merged_df[col].dtype == "object":
        merged_df[col] = merged_df[col].astype(str)

training_data = merged_df.copy()
print(f"✅ 統合イベント数: {len(training_data):,}")

# ==========================================
# データ保存
# ==========================================
training_path = os.path.join(OUTPUT_DIR, f"event_training_data_no_emrtext_{timestamp}.csv")
training_data.to_csv(training_path, index=False)
print(f"💾 学習データ保存: {training_path}")

# ==========================================
# メタデータ作成
# ==========================================
print("🧠 メタデータ作成中...")
metadata = Metadata()
metadata.detect_table_from_dataframe(
    table_name="medical_events",
    data=training_data
)

# ==========================================
# デバイス設定
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"💡 使用デバイス: {device}")

# ==========================================
# CTGAN 学習
# ==========================================
print("🤖 CTGAN 学習開始（EPOCHS=25, device=cuda対応）...")
try:
    model = CTGANSynthesizer(metadata)
    model.fit(training_data)
    model_path = os.path.join(OUTPUT_DIR, f"ctgan_model_no_emrtext_{timestamp}.pkl")
    model.save(model_path)
    print(f"✅ モデル保存完了: {model_path}")
except Exception as e:
    print(f"❌ 学習エラー: {e}")
finally:
    print("🎉 学習完了！")
