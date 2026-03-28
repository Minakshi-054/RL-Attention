import os
import json
import random
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from config import (
    BASE_DIR,
    RANDOM_STATE,
    BATCH_SIZE,
    LEARNING_RATE,
    get_model_output_dir,
)
from data_loader import load_fold_data, normalize_data
from rl_models import (
    build_rl_bilstm_modules,
    build_rl_transformer_modules,
    sample_one_hot_action,
)


# =========================================================
# Helpers
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_json(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, zero_division=0, output_dict=True
        ),
    }
    return metrics


# =========================================================
# RL Trainer
# =========================================================
class RLTrainer:
    def __init__(self, backbone, policy_head, classifier_head, learning_rate=1e-4):
        self.backbone = backbone
        self.policy_head = policy_head
        self.classifier_head = classifier_head

        # Freeze supervised parts
        self.backbone.trainable = False
        self.classifier_head.trainable = False
        self.policy_head.trainable = True

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.bce = tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE
        )

    def train_step(self, x_batch, y_batch):
        y_batch = tf.cast(tf.reshape(y_batch, (-1, 1)), tf.float32)

        with tf.GradientTape() as tape:
            # Backbone outputs
            hidden_seq, base_context, base_attn = self.backbone(x_batch, training=False)

            # Baseline prediction using supervised context
            base_pred = self.classifier_head(base_context, training=False)
            base_loss = self.bce(y_batch, base_pred)   # shape (B,)

            # Policy chooses attention over timesteps
            policy_probs = self.policy_head(hidden_seq, training=True)   # (B, T, 1)

            # Sample one timestep action
            one_hot_action, sampled_idx = sample_one_hot_action(policy_probs)   # (B, T, 1)

            # RL context from selected timestep
            rl_context = tf.reduce_sum(one_hot_action * hidden_seq, axis=1)     # (B, H)

            # Prediction using RL-selected context
            rl_pred = self.classifier_head(rl_context, training=False)
            rl_loss = self.bce(y_batch, rl_pred)   # shape (B,)

            # Reward: positive if RL improves loss
            reward = tf.stop_gradient(base_loss - rl_loss)   # shape (B,)

            # log prob of sampled action
            probs_2d = tf.squeeze(policy_probs, axis=-1)                     # (B, T)
            log_probs = tf.math.log(probs_2d + 1e-8)                         # (B, T)
            action_mask = tf.squeeze(one_hot_action, axis=-1)                # (B, T)
            selected_log_prob = tf.reduce_sum(action_mask * log_probs, axis=1)  # (B,)

            # REINFORCE loss
            policy_loss = -tf.reduce_mean(selected_log_prob * reward)

        grads = tape.gradient(policy_loss, self.policy_head.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy_head.trainable_variables))

        return {
            "policy_loss": float(policy_loss.numpy()),
            "reward_mean": float(tf.reduce_mean(reward).numpy()),
            "base_loss_mean": float(tf.reduce_mean(base_loss).numpy()),
            "rl_loss_mean": float(tf.reduce_mean(rl_loss).numpy()),
        }

    def predict(self, X, use_rl=True, batch_size=256):
        all_probs = []

        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size]
            hidden_seq, base_context, base_attn = self.backbone(xb, training=False)

            if use_rl:
                policy_probs = self.policy_head(hidden_seq, training=False)   # soft attention
                context = tf.reduce_sum(policy_probs * hidden_seq, axis=1)
            else:
                context = base_context

            pred = self.classifier_head(context, training=False)
            all_probs.append(pred.numpy())

        return np.concatenate(all_probs, axis=0).ravel()


# =========================================================
# Build RL modules by model name
# =========================================================
def build_rl_modules(model_name, input_shape):
    if model_name == "rl_bilstm":
        return build_rl_bilstm_modules(input_shape=input_shape)

    if model_name == "rl_transformer":
        return build_rl_transformer_modules(input_shape=input_shape)

    raise ValueError(f"Unsupported RL model: {model_name}")


# =========================================================
# Train one RL fold
# =========================================================
def train_rl_one_fold(fold_id, model_name, rl_epochs=2):
    set_seed(RANDOM_STATE)

    print("\n" + "=" * 60)
    print(f"Training {model_name} - Fold {fold_id}")
    print("=" * 60)

    fold_dir = os.path.join(BASE_DIR, f"fold_{fold_id}")
    output_root = get_model_output_dir(model_name)
    output_dir = os.path.join(output_root, f"fold_{fold_id}")
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_fold_data(fold_dir)
    print("Train:", X_train.shape, y_train.shape)
    print("Val  :", X_val.shape, y_val.shape)
    print("Test :", X_test.shape, y_test.shape)

    # Normalize
    X_train, X_val, X_test, mean, std = normalize_data(X_train, X_val, X_test)
    np.savez_compressed(
        os.path.join(output_dir, "normalization_stats.npz"),
        mean=mean,
        std=std,
    )

    # Build modules
    backbone, policy_head, classifier_head = build_rl_modules(model_name, X_train.shape[1:])

    # Build once with dummy input so shapes are initialized
    dummy_x = tf.convert_to_tensor(X_train[:2], dtype=tf.float32)
    hidden_seq, base_context, base_attn = backbone(dummy_x, training=False)
    _ = policy_head(hidden_seq, training=False)
    _ = classifier_head(base_context, training=False)

    print("Hidden sequence shape :", hidden_seq.shape)
    print("Base context shape    :", base_context.shape)
    print("Base attention shape  :", base_attn.shape)

    trainer = RLTrainer(
        backbone=backbone,
        policy_head=policy_head,
        classifier_head=classifier_head,
        learning_rate=LEARNING_RATE * 0.1,
    )

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(len(X_train), seed=RANDOM_STATE).batch(BATCH_SIZE)

    history = []

    for epoch in range(1, rl_epochs + 1):
        epoch_logs = []

        for xb, yb in train_dataset:
            logs = trainer.train_step(xb, yb)
            epoch_logs.append(logs)

        avg_logs = {
            key: float(np.mean([x[key] for x in epoch_logs]))
            for key in epoch_logs[0]
        }
        avg_logs["epoch"] = epoch
        history.append(avg_logs)

        print(
            f"Epoch {epoch:02d} | "
            f"policy_loss={avg_logs['policy_loss']:.4f} | "
            f"reward_mean={avg_logs['reward_mean']:.4f} | "
            f"base_loss_mean={avg_logs['base_loss_mean']:.4f} | "
            f"rl_loss_mean={avg_logs['rl_loss_mean']:.4f}"
        )

    save_json(os.path.join(output_dir, "rl_history.json"), history)

    # Evaluate RL prediction
    train_prob = trainer.predict(X_train, use_rl=True)
    val_prob = trainer.predict(X_val, use_rl=True)
    test_prob = trainer.predict(X_test, use_rl=True)

    train_metrics = compute_metrics(y_train, train_prob)
    val_metrics = compute_metrics(y_val, val_prob)
    test_metrics = compute_metrics(y_test, test_prob)

    save_json(os.path.join(output_dir, "train_metrics.json"), train_metrics)
    save_json(os.path.join(output_dir, "val_metrics.json"), val_metrics)
    save_json(os.path.join(output_dir, "test_metrics.json"), test_metrics)

    np.savez_compressed(
        os.path.join(output_dir, "train_predictions.npz"),
        y_true=y_train,
        y_prob=train_prob,
        y_pred=(train_prob >= 0.5).astype(int),
    )
    np.savez_compressed(
        os.path.join(output_dir, "val_predictions.npz"),
        y_true=y_val,
        y_prob=val_prob,
        y_pred=(val_prob >= 0.5).astype(int),
    )
    np.savez_compressed(
        os.path.join(output_dir, "test_predictions.npz"),
        y_true=y_test,
        y_prob=test_prob,
        y_pred=(test_prob >= 0.5).astype(int),
    )

    summary = {
        "fold": fold_id,
        "model": model_name,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "rl_epochs": rl_epochs,
        "rl_learning_rate": LEARNING_RATE * 0.1,
    }
    save_json(os.path.join(output_dir, "final_summary.json"), summary)

    # Save RL modules
    backbone.save(os.path.join(output_dir, "backbone.keras"))
    policy_head.save(os.path.join(output_dir, "policy_head.keras"))
    classifier_head.save(os.path.join(output_dir, "classifier_head.keras"))

    print("\nSaved RL results to:", output_dir)

    return {
        "fold": fold_id,
        "train_f1": train_metrics["f1"],
        "val_f1": val_metrics["f1"],
        "test_f1": test_metrics["f1"],
        "train_acc": train_metrics["accuracy"],
        "val_acc": val_metrics["accuracy"],
        "test_acc": test_metrics["accuracy"],
    }
