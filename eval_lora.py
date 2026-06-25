import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import TimesFm2_5ModelForPrediction

from data_handle.price_lora_dataset import PriceWindowDataset, load_price_splits
from train_lora import collate_price_windows


REPO_ROOT = Path(__file__).resolve().parent
DATA_PATH = REPO_ROOT / "datasets" / "datasets.csv"
BASE_MODEL_PATH = REPO_ROOT / "timesfm" / "transformers"
ADAPTER_DIR = REPO_ROOT / "adapters" / "timesfm_sichuan_lora_r4" / "best"
OUTPUT_DIR = REPO_ROOT / "outputs" / "timesfm_lora_eval"
PREDICTION_CSV = OUTPUT_DIR / "test_predictions.csv"
METRICS_JSON = OUTPUT_DIR / "metrics.json"

CONTEXT_LEN = 672
HORIZON_LEN = 96
EXPECTED_MINUTES = 15
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
BATCH_SIZE = 2
NUM_WORKERS = 0
HIGH_PRICE_QUANTILE = 0.90


def build_test_loader():
    _, _, test_split = load_price_splits(
        DATA_PATH,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
    )
    test_dataset = PriceWindowDataset(
        test_split,
        context_len=CONTEXT_LEN,
        horizon_len=HORIZON_LEN,
        expected_minutes=EXPECTED_MINUTES,
    )
    return DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_price_windows,
    )


def load_model(device):
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    base_model = TimesFm2_5ModelForPrediction.from_pretrained(
        str(BASE_MODEL_PATH),
        dtype=model_dtype,
    )
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    return model.to(device).eval()


@torch.no_grad()
def collect_predictions(model, test_loader, device):
    all_true = []
    all_pred = []
    for batch in test_loader:
        past_values = batch["past_values"].to(device)
        future_values = batch["future_values"].to(device)
        outputs = model(
            past_values=[row for row in past_values],
            forecast_context_len=CONTEXT_LEN,
        )
        pred = outputs.mean_predictions[:, :HORIZON_LEN]
        all_true.append(future_values.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().to(torch.float32).numpy())
    return np.concatenate(all_true, axis=0), np.concatenate(all_pred, axis=0)


def calculate_metrics(true_values, pred_values):
    error = pred_values - true_values
    abs_error = np.abs(error)
    mae = float(np.mean(abs_error))
    rmse = float(np.sqrt(np.mean(error ** 2)))

    threshold = float(np.quantile(true_values, HIGH_PRICE_QUANTILE))
    high_mask = true_values >= threshold
    high_price_mae = float(np.mean(abs_error[high_mask])) if high_mask.any() else None

    true_direction = np.sign(np.diff(true_values, axis=1))
    pred_direction = np.sign(np.diff(pred_values, axis=1))
    direction_accuracy = float(np.mean(true_direction == pred_direction))

    return {
        "mae": mae,
        "rmse": rmse,
        "high_price_quantile": HIGH_PRICE_QUANTILE,
        "high_price_threshold": threshold,
        "high_price_mae": high_price_mae,
        "direction_accuracy": direction_accuracy,
        "num_windows": int(true_values.shape[0]),
        "horizon_len": int(true_values.shape[1]),
    }


def save_outputs(true_values, pred_values, metrics):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for window_index in range(true_values.shape[0]):
        for point_index in range(true_values.shape[1]):
            actual = float(true_values[window_index, point_index])
            predicted = float(pred_values[window_index, point_index])
            rows.append(
                {
                    "预测窗口": window_index,
                    "预测点": point_index + 1,
                    "真实电价": round(actual, 3),
                    "预测电价": round(predicted, 3),
                    "误差": round(predicted - actual, 3),
                }
            )
    pd.DataFrame(rows).to_csv(PREDICTION_CSV, index=False, encoding="utf-8-sig")
    METRICS_JSON.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = build_test_loader()
    model = load_model(device)
    true_values, pred_values = collect_predictions(model, test_loader, device)
    metrics = calculate_metrics(true_values, pred_values)
    save_outputs(true_values, pred_values, metrics)
    print(metrics)
    print(f"Saved predictions to: {PREDICTION_CSV}")
    print(f"Saved metrics to: {METRICS_JSON}")


if __name__ == "__main__":
    main()
