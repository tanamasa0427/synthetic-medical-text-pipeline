# ==========================================
# ✅ PARSynthesizer (時系列) + 軽量版
# (Kaggle /working/ ディレクトリ対応)
# [2025-10-23 再々修正版：シーケンス順序番号列を追加]
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
    merged_df['age'] = pd.to_numeric(merged_df['age'], errors='coerce').fillna(0)

cat_cols = ['event_type', 'is_suspected', 'admission_status', 'department', 'unit', '採否', 'emr_type', 'hospital_id']
for col in cat_cols:
    if col in merged_df.columns:
        merged_df[col] = merged_df[col].fillna('Unknown').astype('category')
            
if 'patient_id' in merged_df.columns:
    merged_df['patient_id'] = merged_df['patient_id'].astype(str)
else:
    print("❌ 警告: patient_id が見つかりません。PARSynthesizer は失敗します。")

# -----------------------------------------------------------------
# ⚠️ 修正点：シーケンス番号（順序）の列を作成
# -----------------------------------------------------------------
print("🗓️ シーケンス順序番号の列 (sequence_order) を作成中...")
# 1. IDと日付でソート
training_data = merged_df.sort_values(by=['patient_id', 'date']).copy()
# 2. ID ごとに 1 から始まる連番を振る
training_data['sequence_order'] = training_data.groupby('patient_id').cumcount() + 1
print("✅ シーケンス順序番号の列を作成しました。")
# -----------------------------------------------------------------


print(f"✅ 統合イベント数: {len(training_data):,}")
print("--- データ型確認 (info) ---")
print(training_data.info()) # 'sequence_order' (int64) が追加されていることを確認
print("--------------------------")

training_path = os.path.join(OUTPUT_DIR, f"event_training_data_sequential_{timestamp}.csv")
training_data.to_csv(training_path, index=False)
print(f"💾 学習データ保存: {training_path}")

# -----------------------------------------------------------------
# 3. PARSynthesizer (時系列) モデルの学習
# -----------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"💡 使用デバイス: {device}")
print("🧠 PARSynthesizer 用のメタデータを作成中...")

try:
    # -------------------------------------------------
    # ✅ 再々修正点： 正しいメソッド名と順序キーを使用
    # -------------------------------------------------
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=training_data)
    
    # 1. エンティティ (誰のシーケンスか)
    metadata.update_column(
        column_name='patient_id',
        sdtype='id' 
    )
    metadata.set_primary_key(column_name='patient_id') 

    # 2. シーケンスインデックス (何順か)
    # ⚠️ 'date' ではなく 'sequence_order' を指定
    metadata.update_column(
        column_name='sequence_order', 
        sdtype='numerical', # 数値型 (int64)
        computer_representation='Int64'
    )
    metadata.set_sequence_key(column_name='sequence_order') # 👈 修正
    
    # 3. 'date' 列は通常の datetime として扱う
    metadata.update_column(
        column_name='date',
        sdtype='datetime'
    )
    
    print("✅ メタデータ設定完了。")
    # print(metadata.to_dict()) # デバッグ用に設定内容を表示

    # -------------------------------------------------
    
    print("🤖 PARSynthesizer 学習開始（シーケンス版, EPOCHS=25）...")
    model = PARSynthesizer(
        metadata, # メタデータを渡す
        epochs=25,
        batch_size=500,
        verbose=True,
        device_name=device
    )
    
    model.fit(training_data)
    
    model_path = os.path.join(OUTPUT_DIR, f"par_model_light_{timestamp}.pkl")
    model.save(model_path)
    print(f"✅ モデル保存完了: {model_path}")

except Exception as e:
    print(f"❌ 学習エラー: {e}")
finally:
    print("🎉 学習完了！")
