import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from updated_models import (
    TemporalAttention,
    TransformerBlock,
    ReduceMeanLayer,
    ReduceSumLayer,
)
from data_loader import load_fold_data

# =========================================================
# Paths
# =========================================================
fold_id = 1

data_root = "/home/Students/stg60/RL/RL_Pole/smfallData/prepared_smartfall_meta_wrist"
fold_data_dir = os.path.join(data_root, f"fold_{fold_id}")

rl_dir = "/home/Students/stg60/RL/RL_Pole/smfallData/results/rl_bilstm_results/fold_1"

backbone_path = os.path.join(rl_dir, "backbone.keras")
policy_head_path = os.path.join(rl_dir, "policy_head.keras")
classifier_head_path = os.path.join(rl_dir, "classifier_head.keras")
norm_path = os.path.join(rl_dir, "normalization_stats.npz")

# =========================================================
# Load data
# =========================================================
X_train, y_train, X_val, y_val, X_test, y_test = load_fold_data(fold_data_dir)
y_test = y_test.ravel()

norm_stats = np.load(norm_path)
mean = norm_stats["mean"]
std = norm_stats["std"]
X_test = (X_test - mean) / (std + 1e-8)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# =========================================================
# Load models
# =========================================================
custom_objects = {
    "TemporalAttention": TemporalAttention,
    "TransformerBlock": TransformerBlock,
    "ReduceMeanLayer": ReduceMeanLayer,
    "ReduceSumLayer": ReduceSumLayer,
}

backbone = tf.keras.models.load_model(
    backbone_path,
    custom_objects=custom_objects,
    compile=False,
)

policy_head = tf.keras.models.load_model(
    policy_head_path,
    compile=False,
)

classifier_head = tf.keras.models.load_model(
    classifier_head_path,
    compile=False,
)

print("Models loaded successfully.")

# =========================================================
# DEBUG: check shapes on one sample
# =========================================================
sample = X_test[0:1]

hidden_seq, base_context, base_attn = backbone(sample, training=False)
rl_attn = policy_head(hidden_seq, training=False)

print("hidden_seq shape:", hidden_seq.shape)
print("base_attn shape:", base_attn.shape)
print("rl_attn shape:", rl_attn.shape)

# =========================================================
# Helper
# =========================================================
def get_sample_outputs(sample_batch):
    hidden_seq, base_context, base_attn = backbone(sample_batch, training=False)
    rl_attn = policy_head(hidden_seq, training=False)
    rl_context = tf.reduce_sum(rl_attn * hidden_seq, axis=1)

    base_pred = classifier_head(base_context, training=False).numpy().ravel()[0]
    rl_pred = classifier_head(rl_context, training=False).numpy().ravel()[0]

    base_attn = base_attn.numpy()
    rl_attn = rl_attn.numpy()

    if base_attn.ndim == 3:
        base_attn_1d = base_attn[0, :, 0]
    elif base_attn.ndim == 2:
        base_attn_1d = base_attn[0]
    else:
        raise ValueError(f"Unexpected base_attn shape: {base_attn.shape}")

    if rl_attn.ndim == 3:
        rl_attn_1d = rl_attn[0, :, 0]
    elif rl_attn.ndim == 2:
        rl_attn_1d = rl_attn[0]
    else:
        raise ValueError(f"Unexpected rl_attn shape: {rl_attn.shape}")

    return base_pred, rl_pred, base_attn_1d, rl_attn_1d

# =========================================================
# Predict on all test data using RL
# =========================================================
rl_probs = []
batch_size = 256

for i in range(0, len(X_test), batch_size):
    xb = X_test[i:i+batch_size]

    hidden_seq, base_context, base_attn = backbone(xb, training=False)
    rl_attn = policy_head(hidden_seq, training=False)
    rl_context = tf.reduce_sum(rl_attn * hidden_seq, axis=1)
    rl_pred = classifier_head(rl_context, training=False).numpy().ravel()

    rl_probs.append(rl_pred)

rl_probs = np.concatenate(rl_probs, axis=0)
rl_preds = (rl_probs >= 0.5).astype(int)

print("Computed RL predictions on test set.")

# =========================================================
# Select one correct positive and one wrong sample
# =========================================================
correct_pos_indices = np.where((y_test == 1) & (rl_preds == 1))[0]
wrong_indices = np.where(y_test != rl_preds)[0]

correct_pos_idx = int(correct_pos_indices[0]) if len(correct_pos_indices) > 0 else None
wrong_idx = int(wrong_indices[0]) if len(wrong_indices) > 0 else None

print("Correct positive index:", correct_pos_idx)
print("Misclassified index:", wrong_idx)

# =========================================================
# Publication-style plot
# =========================================================
def plot_publication_style(sample_idx, feature_idx=0, highlight_region=None, save_path=None):
    sample = X_test[sample_idx:sample_idx+1]
    true_label = int(y_test[sample_idx])

    base_pred, rl_pred, base_attn_1d, rl_attn_1d = get_sample_outputs(sample)
    ts = sample[0, :, feature_idx]
    t = np.arange(len(ts))

    fig, axes = plt.subplots(
        3, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.2, 1.2]}
    )

    axes[0].plot(t, ts, linewidth=1.5)
    axes[0].set_ylabel("Signal", fontsize=11)
    axes[0].set_title(
        f"Sample {sample_idx} | True={true_label} | "
        f"Baseline Pred={base_pred:.3f} | RL Pred={rl_pred:.3f}",
        fontsize=12
    )
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, base_attn_1d, linewidth=1.5)
    axes[1].set_ylabel("Attention", fontsize=11)
    axes[1].set_title("Before RL", fontsize=11)
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, rl_attn_1d, linewidth=1.5)
    axes[2].set_ylabel("Attention", fontsize=11)
    axes[2].set_title("After RL", fontsize=11)
    axes[2].set_xlabel("Timestep", fontsize=11)
    axes[2].grid(alpha=0.3)

    if highlight_region is not None:
        start_idx, end_idx = highlight_region
        for ax in axes:
            ax.axvspan(start_idx, end_idx, alpha=0.15)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print("Saved figure to:", save_path)

    plt.show()

# =========================================================
# Plot selected samples
# =========================================================
if correct_pos_idx is not None:
    plot_publication_style(
        sample_idx=correct_pos_idx,
        feature_idx=0,
        highlight_region=(96, 127),
        save_path="correct_positive_attention_plot.png"
    )

if wrong_idx is not None:
    plot_publication_style(
        sample_idx=wrong_idx,
        feature_idx=0,
        highlight_region=(96, 127),
        save_path="wrong_sample_attention_plot.png"
    )
