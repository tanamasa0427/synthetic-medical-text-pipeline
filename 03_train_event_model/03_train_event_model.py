import os
import re
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import torch
import multiprocessing
import joblib

# =========================
# Windows向け安定化設定
# =========================
if os.name == "nt":
    # Windowsでは 'threading' は非対応なので 'spawn' を明示
    multiprocessing.set_start_method("spawn", force=True)
    joblib.parallel_config(n_jobs=1)
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"

from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

# =========================
# 設定
# =========================
DATA_DIR = Path("../data/inputs")
OUTPUT_DIR = Path("../data/outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

EPOCHS = 25                  # CPU実行でも現実的な学習時間
SAMPLE_N = 20                # 生成サンプル件数
TOPK_DISEASE = 200           # 疾患カテゴリ上位
TOPK_DRUG = 200              # 薬剤カテゴリ上位
TOPK_INSPECTION = 80         # 検査名カテゴリ上位
TOPK_VALUE = 800             # 検査値カテゴリ上位
SAVE_PREFIX = datetime.now().strftime("%Y%m%d_%H%M%S")

# =========================
# ユーティリティ関数
# =========================
def load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} が見つかりません")
    df = pd.read_csv(path)
    print(f"✅ {name}: {len(df):,} 件, 列: {list(df.columns)}")
    return df

def standardize_date_column(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    if date_col not in df.columns:
        return df
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    return df.dropna(subset=["date"])

def normalize_gender(val):
    if pd.isna(val):
        return "不明"
    s = str(val).strip()
    if s in ["男", "男性", "M", "male", "Male"]:
        return "男性"
    if s in ["女", "女性", "F", "female", "Female"]:
        return "女性"
    return s or "不明"

def normalize_qual_sign(val: str) -> str:
    s = str(val).replace("＋", "+").replace("－", "-").strip()
    if s in ["陰性", "negative", "neg", "(-)"]:
        return "NEG"
    if s in ["陽性", "positive", "(+)"]:
        return "POS1"
    if s in ["++", "(++)"]:
        return "POS2"
    if s in ["+++", "(+++)"]:
        return "POS3"
    if s in ["+", "+ ", " +"]:
        return "POS1"
    if s in ["-", "- ", " -"]:
        return "NEG"
    if s in ["±", "+-"]:
        return "BORDERLINE"
    return s

def preprocess_inspection_value(val):
    """定量→float化、定性→コード化"""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    m = re.search(r"([-+]?\d*\.?\d+)", s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    return normalize_qual_sign(s)

def add_inspection_type(series: pd.Series) -> pd.Series:
    return series.apply(lambda v: "quantitative" if isinstance(v, float) else ("qualitative" if pd.notna(v) else "unknown"))

def cap_categories(series: pd.Series, top_k: int, other_token: str) -> pd.Series:
    vc = series.fillna("").astype(str).value_counts()
    keep = set(vc.head(top_k).index)
    return series.fillna("").astype(str).apply(lambda x: x if x in keep and x != "" else other_token)

# =========================
# メイン処理
# =========================
def main():
    print(f"📦 SDV CTGAN パイプライン | {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"📁 入力: {DATA_DIR.resolve()}")
    print(f"💾 出力: {OUTPUT_DIR.resolve()}")
    print("📊 データ読み込み中...")

    # --- データ読み込み ---
    gender_df     = load_csv("gender.csv")
    disease_df    = load_csv("disease.csv")
    inspection_df = load_csv("inspection.csv")
    drug_df       = load_csv("drug.csv")
    emr_df        = load_csv("emr.csv")

    # --- 日付列の正規化 ---
    disease_df    = standardize_date_column(disease_df,    "disease_date")
    inspection_df = standardize_date_column(inspection_df, "inspection_date")
    drug_df       = standardize_date_column(drug_df,       "key_date")
    emr_df        = standardize_date_column(emr_df,        "emr_date")

    # --- 各種正規化 ---
    gender_df["gender"] = gender_df["gender"].map(normalize_gender)
    inspection_df["inspection_value"] = inspection_df["inspection_value"].map(preprocess_inspection_value)
    inspection_df["inspection_type"]  = add_inspection_type(inspection_df["inspection_value"])

    # --- 統合 ---
    print("🧩 データ統合中...")
    merged = (
        disease_df.merge(drug_df, on=["patient_id", "date"], how="outer", suffixes=("_disease", "_drug"))
        .merge(inspection_df, on=["patient_id", "date"], how="outer")
        .merge(emr_df, on=["patient_id", "date"], how="outer")
        .merge(gender_df, on="patient_id", how="left")
    )
    merged = merged.sort_values(["patient_id", "date"]).reset_index(drop=True)
    print(f"✅ 統合イベント数: {len(merged):,}")

    # --- 学習データ構築 ---
    train = merged[[
        "patient_id", "date", "gender",
        "disease_name", "drug_name",
        "inspection_name", "inspection_value", "inspection_type"
    ]].copy()

    train["days_since_first"] = train.groupby("patient_id")["date"].transform(lambda x: (x - x.min()).dt.days)
    train["month"] = train["date"].dt.month
    train = train.drop(columns=["date", "patient_id"])

    # --- カテゴリ圧縮 ---
    train["disease_name"]    = cap_categories(train["disease_name"],    TOPK_DISEASE,    "OTHER_DISEASE")
    train["drug_name"]       = cap_categories(train["drug_name"],       TOPK_DRUG,       "OTHER_DRUG")
    train["inspection_name"] = cap_categories(train["inspection_name"], TOPK_INSPECTION, "OTHER_INSPECTION")
    train["inspection_value"] = cap_categories(train["inspection_value"], TOPK_VALUE, "OTHER_VALUE")

    train["gender"] = train["gender"].fillna("不明")
    train["inspection_type"] = train["inspection_type"].fillna("unknown")

    train_out = OUTPUT_DIR / f"event_training_data_{SAVE_PREFIX}.csv"
    train.to_csv(train_out, index=False, encoding="utf-8-sig")
    print(f"💾 学習データ保存: {train_out}")

    # --- メタデータ作成 ---
    print("🧠 メタデータ作成中...")
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train)

    metadata.update_column("inspection_value", sdtype="categorical")
    metadata.update_column("inspection_type",  sdtype="categorical")
    metadata.update_column("gender",           sdtype="categorical")
    metadata.update_column("disease_name",     sdtype="categorical")
    metadata.update_column("drug_name",        sdtype="categorical")
    metadata.update_column("inspection_name",  sdtype="categorical")
    metadata.update_column("month",            sdtype="categorical")
    metadata.update_column("days_since_first", sdtype="numerical")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🤖 CTGAN 学習開始（EPOCHS={EPOCHS}, device={device}）...")
    ctgan = CTGANSynthesizer(
        metadata,
        epochs=EPOCHS,
        cuda=torch.cuda.is_available()
    )

    # --- 学習実行 ---
    ctgan.fit(train)
    print("✅ 学習完了")

    # --- モデル保存 ---
    model_path = OUTPUT_DIR / f"ctgan_model_{SAVE_PREFIX}.pkl"
    ctgan.save(str(model_path))
    print(f"💾 モデル保存: {model_path}")

    # --- メタデータ保存 ---
    meta_json_path = OUTPUT_DIR / f"ctgan_metadata_{SAVE_PREFIX}.json"
    try:
        metadata.save_to_json(str(meta_json_path))
        print(f"💾 メタデータ保存: {meta_json_path}")
    except Exception as e:
        print(f"⚠️ メタデータ保存時の警告: {e}")

    # --- 疑似イベント生成 ---
    print(f"🧬 疑似イベントを {SAMPLE_N} 件生成中...")
    synth = ctgan.sample(SAMPLE_N)
    synth_out = OUTPUT_DIR / f"synthetic_events_{SAVE_PREFIX}.csv"
    synth.to_csv(synth_out, index=False, encoding="utf-8-sig")

    print("\n=== 生成プレビュー ===")
    print(synth.head(min(10, SAMPLE_N)))

    print(f"\n💾 疑似イベント出力: {synth_out}")
    print("🎉 Step 03 完了 — 定性/定量対応・Windows安定化済みバージョン。")


if __name__ == "__main__":
    main()
