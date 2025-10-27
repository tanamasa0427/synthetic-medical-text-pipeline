import os
import re
import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.transformers import FrequencyEncoder, LabelEncoder, UnixTimestampEncoder
from sdv.evaluation.single_table import evaluate_quality

# ===============================================================
# 1️⃣ 基本設定
# ===============================================================
INPUT_PATH = '/kaggle/working/synthetic-medical-text-pipeline/data/inputs'
OUTPUT_PATH = '/kaggle/working/synthetic-medical-text-pipeline/data/outputs'
MODEL_NAME = 'ctgan_model_light'
EPOCHS = 50  # 改良案: 安定化のため少し増加

# ===============================================================
# 2️⃣ データ読込
# ===============================================================
def load_data():
    files = {
        'gender': 'gender.csv',
        'disease': 'disease.csv',
        'inspection': 'inspection.csv',
        'drug': 'drug.csv',
        'emr': 'emr.csv'
    }
    dfs = {}
    for key, file in files.items():
        path = os.path.join(INPUT_PATH, file)
        if os.path.exists(path):
            dfs[key] = pd.read_csv(path)
            print(f"✅ Loaded {file}: {len(dfs[key])} rows")
        else:
            print(f"⚠️ File not found: {file}")
    return dfs

dfs = load_data()

# ===============================================================
# 3️⃣ 製品名の軽い正規化
# ===============================================================
def normalize_product_name(name: str) -> str:
    """製品名を軽く正規化（成分変換は行わない）"""
    if pd.isna(name):
        return name
    name = name.strip().lower()
    name = name.translate(str.maketrans({'　': ' ', '（': '(', '）': ')'}))
    name = re.sub(r'\s+', ' ', name)
    return name

if 'drug' in dfs:
    dfs['drug']['drug_name_norm'] = dfs['drug']['drug_name'].apply(normalize_product_name)

    # ===========================================================
    # Optional改良①：低頻度カテゴリの統合
    # ===========================================================
    min_count = 10  # 10件未満の製品を「その他」に統合
    rare = dfs['drug']['drug_name_norm'].value_counts()[lambda x: x < min_count].index
    dfs['drug']['drug_name_norm'] = dfs['drug']['drug_name_norm'].replace(rare, 'その他')
    print(f"✅ Reduced rare products to 'その他' ({len(rare)} rare items grouped)")

# ===============================================================
# 4️⃣ 学習データ統合
# ===============================================================
def merge_all(dfs):
    merged = []
    for name, df in dfs.items():
        df['event_type'] = name
        merged.append(df)
    df_all = pd.concat(merged, ignore_index=True)
    print(f"✅ Total merged events: {len(df_all)}")
    return df_all

df_all = merge_all(dfs)

# ===============================================================
# 5️⃣ Metadata 定義 & トランスフォーマ設定
# ===============================================================
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_all)

metadata.update_transformers({
    # --- ID・基本情報 ---
    'hospital_id': None,
    'patient_id': None,
    'age': None,
    'gender': LabelEncoder(),

    # --- 疾患関連 ---
    'disease_date': UnixTimestampEncoder(),
    'icd10_code': FrequencyEncoder(),
    'disease_name': FrequencyEncoder(),
    'is_suspected': LabelEncoder(),
    'admission_status': LabelEncoder(),
    'department': LabelEncoder(),

    # --- 検査 ---
    'inspection_date': UnixTimestampEncoder(),
    'inspection_name': FrequencyEncoder(),
    'inspection_value': None,
    'unit': LabelEncoder(),
    '採否': LabelEncoder(),

    # --- 薬剤 ---
    'key_date': UnixTimestampEncoder(),
    'yj_code': LabelEncoder(),
    'drug_name_norm': FrequencyEncoder(),  # 製品名そのまま
    'amount': None,
    'days_count': None,
    'extracted_number': None,
    'daily_dosage': None,

    # --- EMR ---
    'emr_date': UnixTimestampEncoder(),
    'emr_type': LabelEncoder(),
    'emr_text': None
})

# ===============================================================
# 6️⃣ モデル学習
# ===============================================================
print("\n🤖 CTGAN training start...")
synthesizer = CTGANSynthesizer(metadata, epochs=EPOCHS)
synthesizer.fit(df_all)
print("✅ CTGAN training complete.")

# ===============================================================
# 7️⃣ モデル保存・合成データ生成
# ===============================================================
model_path = os.path.join(OUTPUT_PATH, f"{MODEL_NAME}.pkl")
synthesizer.save(model_path)
print(f"✅ Model saved: {model_path}")

synthetic_data = synthesizer.sample(500)  # サンプルを500件生成
synthetic_path = os.path.join(OUTPUT_PATH, 'synthetic_sample.csv')
synthetic_data.to_csv(synthetic_path, index=False, encoding='utf-8')
print(f"🎉 Synthetic sample saved: {synthetic_path}")
print(synthetic_data.head())

# ===============================================================
# 8️⃣ Optional改良②：品質評価レポート生成
# ===============================================================
print("\n📊 Evaluating synthetic data quality...")
quality_report = evaluate_quality(
    real_data=df_all.sample(min(1000, len(df_all))),  # 一部を評価用に使用
    synthetic_data=synthetic_data,
    metadata=metadata
)
report_path = os.path.join(OUTPUT_PATH, 'quality_report.json')
quality_report.save(report_path)
print(f"✅ Quality report saved: {report_path}")

summary = quality_report.get_summary()
print("\n🔍 Quality Summary:")
print(summary)
