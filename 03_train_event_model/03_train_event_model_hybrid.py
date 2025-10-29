# ==============================================
# 💎 03_colab_highprecision_plus_eval.py
#    - T4 GPU最適化 + 製品名ベース薬剤処理 + 品質評価付き最終版
# ==============================================

import os
import re
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
from sdv.transformers import LabelEncoder, FrequencyEncoder, UnixTimestampEncoder

import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------
# 引数設定
# ----------------------------------------------
parser = argparse.ArgumentParser(description="CTGAN学習（Colab向け・高精度＋評価付き）")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--base_dir", type=str, default=None)
parser.add_argument("--drop_emr_text", action="store_true")
parser.add_argument("--save_latest_alias", action="store_true")
parser.add_argument("--sample_n", type=int, default=0)
args = parser.parse_args() if __name__ == "__main__" else parser.parse_args("")

# ----------------------------------------------
# ベースパス設定
# ----------------------------------------------
COLAB_BASE = "/content/synthetic-medical-text-pipeline"
BASE_DIR = Path(args.base_dir) if args.base_dir else (Path(COLAB_BASE) if Path(COLAB_BASE).exists() else Path(".."))
INPUT_DIR = BASE_DIR / "data" / "inputs"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TS = datetime.now().strftime("%Y%m%d_%H%M%S")

# ----------------------------------------------
# GPU/CPU自動判定 + OOM対策
# ----------------------------------------------
if torch.cuda.is_available():
    try:
        torch.zeros((1024, 1024)).cuda()
        DEVICE = "cuda"
        print("🟢 GPU (T4) 利用: CUDA有効")
    except RuntimeError:
        DEVICE = "cpu"
        print("⚠️ GPUメモリ不足 → CPUで実行")
else:
    DEVICE = "cpu"
    print("💻 CPUモードで実行")

# ----------------------------------------------
# CSV読込関数
# ----------------------------------------------
def load_csv(name: str):
    path = INPUT_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"❌ {path} が見つかりません")
    df = pd.read_csv(path)
    print(f"✅ {name}.csv: {len(df):,}件")
    return df

def to_date(df, col):
    if col in df.columns:
        df["date"] = pd.to_datetime(df[col], errors="coerce")
    return df

# ----------------------------------------------
# データ読込
# ----------------------------------------------
print("📥 実データ読込中...")
gender_df = load_csv("gender")
disease_df = load_csv("disease")
inspection_df = load_csv("inspection")
drug_df = load_csv("drug")
emr_df = load_csv("emr")

if args.drop_emr_text and "emr_text" in emr_df.columns:
    emr_df = emr_df.drop(columns=["emr_text"])

# ----------------------------------------------
# 日付変換
# ----------------------------------------------
disease_df = to_date(disease_df, "disease_date")
inspection_df = to_date(inspection_df, "inspection_date")
drug_df = to_date(drug_df, "drug_date" if "drug_date" in drug_df.columns else "key_date")
emr_df = to_date(emr_df, "emr_date")

# ----------------------------------------------
# gender 正規化
# ----------------------------------------------
if "gender" in gender_df.columns:
    gender_df["gender"] = gender_df["gender"].replace({"M":"男性","F":"女性","male":"男性","female":"女性"})

# ----------------------------------------------
# 製品名ベースの薬剤名処理
# ----------------------------------------------
def normalize_drug_name(name: str):
    if pd.isna(name):
        return name
    s = str(name)
    s = s.replace("　", " ").replace("ｍｇ", "mg").replace("ＭＧ", "mg")
    s = re.sub(r"（.*?）|\(.*?\)", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

if "drug_name" in drug_df.columns:
    drug_df["drug_name"] = drug_df["drug_name"].map(normalize_drug_name)
    # 低頻度薬剤を「その他」に統合
    value_counts = drug_df["drug_name"].value_counts()
    rare_names = value_counts[value_counts < 10].index
    drug_df["drug_name"] = drug_df["drug_name"].replace(rare_names, "その他")
    # 上位300薬剤を保持
    top_drugs = drug_df["drug_name"].value_counts().head(300).index
    drug_df["drug_name"] = drug_df["drug_name"].apply(lambda x: x if x in top_drugs else "OTHER_DRUG")

# ----------------------------------------------
# データ統合
# ----------------------------------------------
print("🧩 データ統合中...")
for df, t in [(disease_df, "disease"), (inspection_df, "inspection"), (drug_df, "drug"), (emr_df, "emr")]:
    df["event_type"] = t

merged = pd.concat([disease_df, inspection_df, drug_df, emr_df], ignore_index=True)
merged = merged.merge(gender_df, on="patient_id", how="left")
merged = merged.sort_values(["patient_id", "date"]).reset_index(drop=True)

# ----------------------------------------------
# 学習用データ作成
# ----------------------------------------------
train_cols = [c for c in ["patient_id","date","gender","disease_name","icd10_code","drug_name","inspection_name","inspection_value","department"] if c in merged.columns]
train = merged[train_cols].copy()
train["days_since_first"] = train.groupby("patient_id")["date"].transform(lambda x:(x-x.min()).dt.days)
train["month"] = train["date"].dt.month
if {"drug_name","department"}.issubset(train.columns):
    train["drug_name_department"] = train["drug_name"] + "_" + train["department"]
train = train.drop(columns=["patient_id","date"], errors="ignore")

# ----------------------------------------------
# メタデータ定義 + トランスフォーマー指定
# ----------------------------------------------
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(train)
metadata.update_transformers({
    "gender": LabelEncoder(),
    "department": LabelEncoder(),
    "disease_name": FrequencyEncoder(),
    "drug_name": FrequencyEncoder(),
    "inspection_name": FrequencyEncoder(),
    "inspection_value": None,
    "days_since_first": None,
    "month": LabelEncoder(),
    "drug_name_department": FrequencyEncoder(),
})

meta_path = OUTPUT_DIR / f"ctgan_metadata_{TS}.json"
metadata.save_to_json(str(meta_path))
print(f"💾 スキーマ保存: {meta_path}")

# ----------------------------------------------
# CTGAN学習 (安定化 + batch_size指定)
# ----------------------------------------------
print("🚀 CTGAN学習開始...")
ctgan = CTGANSynthesizer(
    metadata,
    epochs=args.epochs,
    batch_size=500,
    cuda=(DEVICE == "cuda"),
)

try:
    ctgan.fit(train)
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("⚠️ GPUメモリ不足 → CPUで再学習")
        torch.cuda.empty_cache()
        ctgan = CTGANSynthesizer(metadata, epochs=args.epochs, batch_size=500, cuda=False)
        ctgan.fit(train)
    else:
        raise e

model_path = OUTPUT_DIR / f"ctgan_model_no_emrtext_{TS}.pkl"
ctgan.save(str(model_path))
print(f"💾 モデル保存: {model_path}")

# ----------------------------------------------
# 学習データ保存
# ----------------------------------------------
train_path = OUTPUT_DIR / f"event_training_data_no_emrtext_{TS}.csv"
train.to_csv(train_path, index=False)
print(f"💾 学習データ保存: {train_path}")

if args.save_latest_alias:
    alias_path = OUTPUT_DIR / "event_training_data_no_emrtext_latest.csv"
    train.to_csv(alias_path, index=False)
    print(f"🔖 最新版保存: {alias_path}")

# ----------------------------------------------
# 品質評価
# ----------------------------------------------
print("📈 合成データ品質評価中...")
try:
    synthetic_data = ctgan.sample(min(500, len(train)))
    quality = evaluate_quality(real_data=train.sample(min(1000, len(train))), synthetic_data=synthetic_data, metadata=metadata)
    report_path = OUTPUT_DIR / f"quality_report_{TS}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Quality Score: {quality.get_score():.3f}\n")
        f.write(str(quality.get_details()))
    print(f"🧾 品質スコア: {quality.get_score():.3f} → 詳細: {report_path}")
except Exception as e:
    print(f"⚠️ 品質評価スキップ: {e}")

# ----------------------------------------------
# サンプル生成（任意）
# ----------------------------------------------
if args.sample_n>0:
    synth = ctgan.sample(args.sample_n)
    synth_out = OUTPUT_DIR / f"synthetic_events_{TS}.csv"
    synth.to_csv(synth_out, index=False)
    print(f"🧬 サンプル生成: {synth_out}")

print("\n🎉 Step03 完了 — Colab高精度＋評価版実行済\n")
