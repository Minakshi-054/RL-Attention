import os
import json
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras import callbacks
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
    N_FOLDS,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    RANDOM_STATE,
    get_model_output_dir,
)
from data_loader import load_fold_data, normalize_data, get_class_weights
#from models import build_model
#from updated_models import build_model
#from updated_models import build_model, TemporalAttention, TransformerBlock

from updated_models import (
    build_model,
    TemporalAttention,
    TransformerBlock,
    ReduceMeanLayer,
    ReduceSumLayer,
)






def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_json(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def evaluate_model(model, X, y, split_name, save_dir, threshold=0.5):
    y_prob = model.predict(X, batch_size=256, verbose=0).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    cm = confusion_matrix(y, y_pred).tolist()
    report = classification_report(y, y_pred, zero_division=0, output_dict=True)

    results = {
        "split": split_name,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm,
        "classification_report": report,
    }

    save_json(os.path.join(save_dir, f"{split_name}_metrics.json"), results)

    np.savez_compressed(
        os.path.join(save_dir, f"{split_name}_predictions.npz"),
        y_true=y,
        y_prob=y_prob,
        y_pred=y_pred,
    )

    return results


def train_one_fold(fold_id, model_name):
    print("\n" + "=" * 60)
    print(f"Training {model_name} - Fold {fold_id}")
    print("=" * 60)

    fold_dir = os.path.join(BASE_DIR, f"fold_{fold_id}")
    output_root = get_model_output_dir(model_name)
    output_dir = os.path.join(output_root, f"fold_{fold_id}")
    os.makedirs(output_dir, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test = load_fold_data(fold_dir)

    print("Train:", X_train.shape, y_train.shape)
    print("Val  :", X_val.shape, y_val.shape)
    print("Test :", X_test.shape, y_test.shape)

    X_train, X_val, X_test, mean, std = normalize_data(X_train, X_val, X_test)

    np.savez_compressed(
        os.path.join(output_dir, "normalization_stats.npz"),
        mean=mean,
        std=std,
    )

    class_weights = get_class_weights(y_train)
    print("Class weights:", class_weights)

    model = build_model(model_name, input_shape=X_train.shape[1:])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    model.summary()

    checkpoint_path = os.path.join(output_dir, f"{model_name}_best.keras")

    cb_list = [
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        #epochs=2,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=cb_list,
        verbose=1,
    )

    save_json(os.path.join(output_dir, "history.json"), history.history)

    #best_model = tf.keras.models.load_model(checkpoint_path)
    best_model = tf.keras.models.load_model(
    checkpoint_path,
    custom_objects={
        "TemporalAttention": TemporalAttention,
        "TransformerBlock": TransformerBlock,
        "ReduceMeanLayer": ReduceMeanLayer,
        "ReduceSumLayer": ReduceSumLayer,
    }
)
    





    train_results = evaluate_model(best_model, X_train, y_train, "train", output_dir)
    val_results = evaluate_model(best_model, X_val, y_val, "val", output_dir)
    test_results = evaluate_model(best_model, X_test, y_test, "test", output_dir)
    save_json(os.path.join(output_dir, "test_results.json"), test_results)
    summary = {
        "fold": fold_id,
        "model": model_name,
        "input_shape": list(X_train.shape[1:]),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "class_weights": class_weights,
        "train_results": train_results,
        "val_results": val_results,
        "test_results": test_results,
    }

    save_json(os.path.join(output_dir, "final_summary.json"), summary)

    return {
        "fold": fold_id,
        "train_acc": train_results["accuracy"],
        "val_acc": val_results["accuracy"],
        "test_acc": test_results["accuracy"],
        "train_f1": train_results["f1"],
        "val_f1": val_results["f1"],
        "test_f1": test_results["f1"],
    }


def run_all_folds(model_name):
    set_seed(RANDOM_STATE)

    output_root = get_model_output_dir(model_name)
    os.makedirs(output_root, exist_ok=True)

    all_results = []

    for fold_id in range(1, N_FOLDS + 1):
        result = train_one_fold(fold_id, model_name)
        all_results.append(result)

    save_json(os.path.join(output_root, "all_folds_summary.json"), all_results)

    avg_train_acc = np.mean([r["train_acc"] for r in all_results])
    avg_val_acc = np.mean([r["val_acc"] for r in all_results])
    avg_test_acc = np.mean([r["test_acc"] for r in all_results])

    avg_train_f1 = np.mean([r["train_f1"] for r in all_results])
    avg_val_f1 = np.mean([r["val_f1"] for r in all_results])
    avg_test_f1 = np.mean([r["test_f1"] for r in all_results])

    print("\n" + "=" * 60)
    print(f"FINAL CROSS-FOLD AVERAGE - {model_name}")
    print("=" * 60)
    print("Average Train Accuracy:", round(avg_train_acc, 4))
    print("Average Val Accuracy  :", round(avg_val_acc, 4))
    print("Average Test Accuracy :", round(avg_test_acc, 4))
    print("Average Train F1      :", round(avg_train_f1, 4))
    print("Average Val F1        :", round(avg_val_f1, 4))
    print("Average Test F1       :", round(avg_test_f1, 4))
