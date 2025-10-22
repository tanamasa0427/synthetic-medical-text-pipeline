# ==========================================
# 03_train_event_model.py (Colab対応最終安定版)
# Synthetic Medical Event Model Training Pipeline
# ==========================================

import os
import pandas as pd
import numpy as np
import torch
import joblib
from datetime import datetime
from tqdm import tqdm
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata

# 並列処理の安定化
joblib.parallel_backend('loky', n_jobs=1)

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
        print(f"⚠️ 読込エラー {path}: {e}")
        return pd.DataFrame()

# ファイル読込
gender_df = read_csv_safe(os.path.join(INPUT_DIR, "gender.csv"))
disease_df = read_csv_safe(os.path.join(INPUT_DIR, "disease.csv"))
inspection_df = read_csv_safe(os.path.join(INPUT_DIR, "inspection.csv"))
drug_df = read_csv_safe(os.path.join(INPUT_DIR, "drug.csv"))
emr_df = read_csv_safe(os.path.join(INPUT_DIR, "emr.csv"))

# ==========================================
# データ整形・統合
# ==========================================
print("🧩 データ統合中...")

rename_map = {
    "disease_date": "date",
    "inspection_date": "date",
    "key_date": "date",
    "emr_date": "date"
}
for df in [disease_df, inspection_df, drug_df, emr_df]:
    df.rename(columns={c: rename_map[c] for c in rename_map if c in df.columns}, inplace=True)

disease_df["type"] = "disease"
inspection_df["type"] = "inspection"
drug_df["type"] = "drug"
emr_df["type"] = "emr"

cols = [
    "hospital_id", "patient_id", "gender", "age", "date",
    "type", "disease_name", "icd10_code", "inspection_name",
    "inspection_value", "drug_name", "amount", "unit",
    "emr_text", "department", "admission_status"
]

def standardize(df, cols):
    return df.reindex(columns=cols, fill_value=np.nan)

dfs = [standardize(df, cols) for df in [disease_df, inspection_df, drug_df, emr_df]]
merged_df = pd.concat(dfs, ignore_index=True)

merged_df = merged_df.merge(gender_df, on=["hospital_id", "patient_id"], how="left", suffixes=("", "_gender"))
merged_df["gender"] = merged_df["gender"].fillna(merged_df["gender_gender"])

merged_df["date"] = pd.to_datetime(merged_df["date"], errors="coerce")
merged_df = merged_df.dropna(subset=["date"])
merged_df["month"] = merged_df["date"].dt.month
merged_df["days_since_first"] = (
    merged_df.groupby("patient_id")["date"].transform(lambda x: (x - x.min()).dt.days)
)

print(f"✅ 統合イベント数: {len(merged_df):,}")

output_path = os.path.join(OUTPUT_DIR, f"event_training_data_{timestamp}.csv")
merged_df.to_csv(output_path, index=False)
print(f"💾 学習データ保存: {output_path}")

# ==========================================
# SDV モデル学習
# ==========================================
print("🧠 メタデータ作成中...")

metadata = Metadata()
metadata.detect_table_from_dataframe(table_name="medical_events", data=training_data)

training_data = merged_df.sample(frac=0.1, random_state=42).reset_index(drop=True)
print(f"💡 サンプリング後: {len(training_data):,} 件（元の10%）")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🤖 CTGAN 学習開始（EPOCHS=25, device={device}）...")

try:
    ctgan = CTGANSynthesizer(metadata, epochs=25, verbose=True)
    ctgan.fit(training_data)
    model_path = os.path.join(OUTPUT_DIR, f"ctgan_model_{timestamp}.pkl")
    joblib.dump(ctgan, model_path)
    print(f"✅ モデル保存: {model_path}")
except Exception as e:
    print(f"❌ 学習エラー: {e}")

print("🎉 学習完了！")
