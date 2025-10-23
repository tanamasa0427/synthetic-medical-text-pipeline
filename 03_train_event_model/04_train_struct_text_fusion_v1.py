# ==========================================
# ✅ 構造＋テキスト統合 疑似データ生成パイプライン (SDV 1.x 最終安定版)
# 対応: Python 3.11 / Kaggle環境 / GitHub連携対応
# ==========================================

import os
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
from sdv.sequential import PARSynthesizer
from sdv.metadata import SingleTableMetadata

# ==========================================
# 初期設定
# ==========================================
BASE_DIR = "/kaggle/working/synthetic-medical-text-pipeline"
INPUT_DIR = os.path.join(BASE_DIR, "data/inputs")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"📦 構造＋文脈 パイプライン開始 | {datetime.now():%Y-%m-%d %H:%M:%S}")

# ==========================================
# データ読込関数
# ==========================================
def read_csv_safe(name):
    path = os.path.join(INPUT_DIR, f"{name}.csv")
    df = pd.read_csv(path)
    print(f"✅ {name}: {len(df):,}件, 列={df.columns.tolist()}")
    return df

gender_df = read_csv_safe("gender")
disease_df = read_csv_safe("disease")
inspection_df = read_csv_safe("inspection")
drug_df = read_csv_safe("drug")
emr_df = read_csv_safe("emr")

# ==========================================
# 正規化 & 統合（構造データ）
# ==========================================
def normalize(df, date_col, event_type):
    df = df.copy()
    df["event_type"] = event_type
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    return df

disease_df = normalize(disease_df, "disease_date", "disease")
inspection_df = normalize(inspection_df, "inspection_date", "inspection")
drug_df = normalize(drug_df, "key_date", "drug")

struct_df = pd.concat([disease_df, inspection_df, drug_df], ignore_index=True)
struct_df = struct_df.merge(gender_df, on=["hospital_id", "patient_id"], how="left")
struct_df = struct_df.sort_values(["patient_id", "date"])
print(f"🧩 構造データ統合完了: {len(struct_df):,}行")

# ==========================================
# EMRテキスト埋め込み
# ==========================================
print("💬 emr_text埋め込み中...")
emr_df["date"] = pd.to_datetime(emr_df["emr_date"], errors="coerce")
embed_model = SentenceTransformer("sonoisa/sentence-bert-base-ja-mean-tokens")

emr_df["embedding"] = emr_df["emr_text"].apply(lambda x: embed_model.encode(str(x), convert_to_numpy=True))

emr_embed_df = emr_df.groupby(["patient_id", "date"])["embedding"].apply(
    lambda v: np.mean(np.stack(v), axis=0)
).reset_index()

# 構造データに近似マージ（日付差±3日）
def nearest_merge(struct_df, emr_embed_df, max_days=3):
    out = []
    for pid, group in tqdm(struct_df.groupby("patient_id")):
        emb_sub = emr_embed_df[emr_embed_df["patient_id"] == pid]
        if len(emb_sub) == 0:
            group["embedding"] = None
            out.append(group)
            continue
        merged = []
        for _, row in group.iterrows():
            diffs = abs((emb_sub["date"] - row["date"]).dt.days)
            idx = diffs.idxmin() if len(diffs) > 0 else None
            if idx is not None and diffs[idx] <= max_days:
                row["embedding"] = emb_sub.loc[idx, "embedding"]
            else:
                row["embedding"] = None
            merged.append(row)
        out.append(pd.DataFrame(merged))
    return pd.concat(out, ignore_index=True)

merged_df = nearest_merge(struct_df, emr_embed_df)
print(f"✅ 構造＋文脈統合完了: {len(merged_df):,}行")

# nan埋め込みをゼロベクトルに
embed_dim = len(emr_embed_df["embedding"].iloc[0])
merged_df["embedding"] = merged_df["embedding"].apply(lambda x: np.zeros(embed_dim) if x is None else x)

# ベクトル展開
emb_cols = [f"emr_emb_{i}" for i in range(embed_dim)]
emb_matrix = np.vstack(merged_df["embedding"].to_numpy())
emb_df = pd.DataFrame(emb_matrix, columns=emb_cols)
merged_df = pd.concat([merged_df.drop(columns=["embedding"]), emb_df], axis=1)

# ==========================================
# メタデータ定義 (SDV 1.x 構成)
# ==========================================
print("🧠 メタデータ作成中...")
merged_df["event_id"] = range(1, len(merged_df) + 1)

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(merged_df)
metadata.update_column("event_id", sdtype="id")
metadata.set_primary_key("event_id")
metadata.update_column("patient_id", sdtype="id")
metadata.set_sequence_key("patient_id")
metadata.update_column("date", sdtype="datetime")
print("✅ メタデータ作成完了 (PK=event_id, SK=patient_id)")

# ==========================================
# PARSynthesizer学習
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"💡 使用デバイス: {device}")
print("🤖 時系列学習開始 (EPOCHS=300)...")

model = PARSynthesizer(
    metadata=metadata,
    cuda=(device == "cuda"),
    epochs=300
)
model.fit(merged_df)

model_path = os.path.join(OUTPUT_DIR, f"par_model_struct_text_{timestamp}.pkl")
model.save(model_path)
print(f"✅ モデル保存完了: {model_path}")

# ==========================================
# 生成
# ==========================================
print("🧬 疑似データ生成中...")
synthetic_data = model.sample(num_sequences=100)
synthetic_data.to_csv(os.path.join(OUTPUT_DIR, f"synthetic_struct_text_{timestamp}.csv"), index=False)
print("🎉 疑似データ生成完了！")
