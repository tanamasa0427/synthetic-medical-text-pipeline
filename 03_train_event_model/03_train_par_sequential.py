# ==========================================
# ✅ PARSynthesizer (時系列) + 最終安定版
# 対応: SDV 1.0.0 / Python 3.10 / Kaggle環境
# [修正: epochs, batch_size を __init__ に渡す (SDV 1.x 仕様)]
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
# ⚙️ 環境設定
# ------------------------------------------
ENVIRONMENT = "kaggle_working"

if ENVIRONMENT == "kaggle_working":
    BASE_DIR = "/kaggle/working/synthetic-medical-text-pipeline"
elif ENVIRONMENT == "colab":
    BASE_DIR = "/content/synthetic-medical-text-pipeline"
else:
    BASE_DIR = "./synthetic-medical-text-pipeline"

INPUT_DIR = os.path.join(BASE_DIR, "data/inputs")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"📦 SDV PARSynthesizer (時系列) パイプライン開始: {datetime.now():%Y-%m-%d %H:%M:%S}")

# ------------------------------------------
# 1. 安全なCSV読込関数
# ------------------------------------------
def read_csv_safe(path):
    try:
        df = pd.read_csv(path)
        print(f"✅ {os.path.basename(path)}: {len(df):,} 件, 列: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"⚠️ {os.path.basename(path)} 読込失敗: {e}")
        return pd.DataFrame()

# ------------------------------------------
# 2. データ読込
# ------------------------------------------
gender_df = read_csv_safe(os.path.join(INPUT_DIR, "gender.csv"))
disease_df = read_csv_safe(os.path.join(INPUT_DIR, "disease.csv"))
inspection_df = read_csv_safe(os.path.join(INPUT_DIR, "inspection.csv"))
drug_df = read_csv_safe(os.path.join(INPUT_DIR, "drug.csv"))
emr_df = read_csv_safe(os.path.join(INPUT_DIR, "emr.csv"))

# ------------------------------------------
# 3. 日付列とevent_type列を統一
# ------------------------------------------
def normalize(df, date_col, type_label):
    if df.empty:
        return df
    df = df.copy()
    if date_col in df.columns:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        df["date"] = pd.NaT
    df["event_type"] = type_label
    return df

disease_df = normalize(disease_df, "disease_date", "disease")
inspection_df = normalize(inspection_df, "inspection_date", "inspection")
drug_df = normalize(drug_df, "key_date", "drug")
emr_df = normalize(emr_df, "emr_date", "emr")

if "emr_text" in emr_df.columns:
    emr_df = emr_df.drop(columns=["emr_text"], errors="ignore")

# ------------------------------------------
# 4. 結合と前処理
# ------------------------------------------
merged_df = pd.concat([disease_df, inspection_df, drug_df, emr_df], ignore_index=True)

drop_cols = [
    "inspection_value", "drug_name", "icd10_code", "disease_name",
    "yj_code", "inspection_name", "amount", "days_count",
    "extracted_number", "daily_dosage"
]
merged_df = merged_df.drop(columns=[c for c in drop_cols if c in merged_df.columns], errors="ignore")

# 日付NaT除外
merged_df["date"] = pd.to_datetime(merged_df["date"], errors="coerce")
before = len(merged_df)
merged_df = merged_df.dropna(subset=["date"])
print(f"🗓️ NaT除外: {before - len(merged_df)} 件削除")

# 不要列除外
for c in ["disease_date", "inspection_date", "key_date", "emr_date"]:
    if c in merged_df.columns:
        merged_df = merged_df.drop(columns=[c], errors="ignore")

# 年齢列処理
if "age" in merged_df.columns:
    merged_df["age"] = pd.to_numeric(merged_df["age"], errors="coerce").fillna(0)

# カテゴリ列の正規化
cat_cols = ['event_type', 'is_suspected', 'admission_status', 'department', 'unit', '採否', 'emr_type', 'hospital_id']
for col in cat_cols:
    if col in merged_df.columns:
        merged_df[col] = merged_df[col].fillna("Unknown").astype("category")

# patient_id 確認
if "patient_id" not in merged_df.columns:
    raise ValueError("❌ 必須列 patient_id が存在しません。")

merged_df["patient_id"] = merged_df["patient_id"].astype(str)

# ------------------------------------------
# 5. シーケンス列作成
# ------------------------------------------
merged_df = merged_df.sort_values(by=["patient_id", "date"])
merged_df["sequence_order"] = merged_df.groupby("patient_id").cumcount() + 1

training_data = merged_df.reset_index(drop=True)
training_path = os.path.join(OUTPUT_DIR, f"event_training_data_sequential_{timestamp}.csv")
training_data.to_csv(training_path, index=False)
print(f"✅ 学習データ保存完了: {training_path}")

# ------------------------------------------
# 6. メタデータ作成 (✅ 最終安定版)
# ------------------------------------------
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(training_data)

metadata.update_column("patient_id", sdtype="id")
metadata.set_primary_key("patient_id")

metadata.update_column("sequence_order", sdtype="id")
metadata.set_sequence_key("sequence_order")

metadata.update_column("date", sdtype="datetime")

print("✅ メタデータ作成完了")

# ------------------------------------------
# 7. PARSynthesizer 学習 (SDV 1.0.0 仕様)
# ------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"💡 使用デバイス: {device}")

try:
    print("🤖 PARSynthesizer 学習開始 (EPOCHS=25)")
    
    # ------------------------------------------
    # ⚠️ 最終修正点： (ご提示いただいた通り)
    # epochs, batch_size を __init__ に渡す
    # ------------------------------------------
    model = PARSynthesizer(
        metadata=metadata,
        cuda=(device == "cuda"),
        epochs=25,
        batch_size=500
    )
    
    # .fit() には学習パラメータを渡さない
    model.fit(
        data=training_data
    )
    
    model_path = os.path.join(OUTPUT_DIR, f"par_model_light_{timestamp}.pkl")
    model.save(model_path)
    print(f"✅ モデル保存完了: {model_path}")

except Exception as e:
    print(f"❌ 学習中にエラーが発生しました: {e}")

print("🎉 パイプライン完了！")
