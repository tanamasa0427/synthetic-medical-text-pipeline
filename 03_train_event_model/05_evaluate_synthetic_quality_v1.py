# ==========================================
# ✅ 05_evaluate_synthetic_quality_v1.py
# ==========================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from sdv.evaluation.single_table import evaluate_quality
from datetime import datetime

BASE_DIR = "/content/synthetic-medical-text-pipeline"
OUTPUT_DIR = os.path.join(BASE_DIR, "data/outputs")
REAL_PATH = os.path.join(OUTPUT_DIR, "event_training_data_no_emrtext_latest.csv")  # 元データ
SYNTH_PATH = sorted(
    [f for f in os.listdir(OUTPUT_DIR) if f.startswith("synthetic_struct_text_")],
    key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x))
)[-1]
SYNTH_PATH = os.path.join(OUTPUT_DIR, SYNTH_PATH)

print(f"📊 評価対象: {SYNTH_PATH}")

real_data = pd.read_csv(REAL_PATH)
synthetic_data = pd.read_csv(SYNTH_PATH)

# ==========================================
# SDV品質スコア
# ==========================================
from sdv.metadata import Metadata
metadata = Metadata()
metadata.detect_table_from_dataframe(table_name="medical_events", data=real_data)

score = evaluate_quality(real_data=real_data, synthetic_data=synthetic_data, metadata=metadata)
print(f"✅ Quality score: {score.get_score():.3f}")

# ==========================================
# 分布比較
# ==========================================
numeric_cols = [c for c in real_data.columns if real_data[c].dtype != "object" and real_data[c].dtype != "datetime64[ns]"]
plot_dir = os.path.join(OUTPUT_DIR, f"eval_plots_{datetime.now():%Y%m%d_%H%M%S}")
os.makedirs(plot_dir, exist_ok=True)

for col in numeric_cols[:10]:  # 上位10カラムだけ
    plt.figure()
    plt.hist(real_data[col].dropna(), bins=50, alpha=0.5, label="Real")
    plt.hist(synthetic_data[col].dropna(), bins=50, alpha=0.5, label="Synthetic")
    plt.legend()
    plt.title(f"Distribution: {col}")
    plt.savefig(os.path.join(plot_dir, f"{col}.png"))
    plt.close()

print(f"📈 ヒストグラム保存完了: {plot_dir}")

# ==========================================
# 相関比較
# ==========================================
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(real_data[numeric_cols].corr(), cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Real Data Correlation")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(synthetic_data[numeric_cols].corr(), cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Synthetic Data Correlation")
plt.colorbar()

corr_path = os.path.join(plot_dir, "correlation_comparison.png")
plt.savefig(corr_path)
plt.close()
print(f"🔍 相関比較保存完了: {corr_path}")
print("🎉 評価完了！")
