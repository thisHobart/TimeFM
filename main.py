import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import timesfm

from data_handle.price_loader import load_daily_timesfm_price_windows


REPO_ROOT = Path(__file__).resolve().parent
DATA_PATH = REPO_ROOT / "datasets" / "datasets.csv"
MODEL_PATH = REPO_ROOT / "timesfm"
RESULT_DIR = REPO_ROOT / "result" / "data1"
print("RESULT_DIR:", RESULT_DIR)
PREDICTION_CSV = RESULT_DIR / "timesfm_price_predictions.csv"
RUN_CONFIG_JSON = RESULT_DIR / "timesfm_run_config.json"
CONTEXT_LEN = 1024
HORIZON = 96
ROLLING_STEPS = 1
TOTAL_FORECAST_STEPS = HORIZON * ROLLING_STEPS
INTERVAL_MINUTES = 15
FORECAST_START_HOUR = 0
FORECAST_START_MINUTE = 15
FORECAST_START_DATE = "2026-06-15"
FORECAST_END_DATE = None
BATCH_SIZE = 16
PREVIEW_ROWS = 10
COMPILE_CONFIG = {
    "max_context": CONTEXT_LEN,
    "max_horizon": HORIZON,
    "normalize_inputs": True,
    "use_continuous_quantile_head": True,
    "force_flip_invariance": True,
    "infer_is_positive": True,
    "fix_quantile_crossing": True,
}


def build_model():
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(str(MODEL_PATH))
    model.compile(timesfm.ForecastConfig(**COMPILE_CONFIG))
    return model


def iter_batches(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield start, items[start:start + batch_size]


def save_run_config(window_count):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "model": "TimesFM_2p5_200M_torch",
        "model_path": str(MODEL_PATH),
        "data_path": str(DATA_PATH),
        "context_len": CONTEXT_LEN,
        "horizon": HORIZON,
        "rolling_steps": ROLLING_STEPS,
        "total_forecast_steps": TOTAL_FORECAST_STEPS,
        "interval_minutes": INTERVAL_MINUTES,
        "forecast_start_hour": FORECAST_START_HOUR,
        "forecast_start_minute": FORECAST_START_MINUTE,
        "forecast_start_date": FORECAST_START_DATE,
        "forecast_end_date": FORECAST_END_DATE,
        "batch_size": BATCH_SIZE,
        "window_count": window_count,
        "compile_config": COMPILE_CONFIG,
        "prediction_csv": str(PREDICTION_CSV),
    }
    RUN_CONFIG_JSON.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_predictions(windows, forecasts):
    rows = []
    for window_index, (window, forecast) in enumerate(zip(windows, forecasts)):
        forecast_start_day = window.future_datetime.iloc[0].date().isoformat()
        for point_index, (timestamp, actual, predicted) in enumerate(
            zip(window.future_datetime, window.future_price, forecast),
            start=1,
        ):
            rows.append(
                {
                    "预测窗口": window_index,
                    "预测起始日": forecast_start_day,
                    "预测点": point_index,
                    "时间": timestamp.strftime("%Y-%m-%d %H:%M"),
                    "真实电价": round(float(actual), 3),
                    "预测电价": round(float(predicted), 3),
                    "误差": round(float(predicted - actual), 3),
                }
            )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(PREDICTION_CSV, index=False, encoding="utf-8-sig")


def rolling_forecast_batch(model, batch_windows):
    if ROLLING_STEPS <= 0:
        raise ValueError("ROLLING_STEPS must be positive")

    contexts = [window.context.astype(np.float32, copy=True) for window in batch_windows]
    step_forecasts = []

    for _ in range(ROLLING_STEPS):
        point_forecast, _ = model.forecast(horizon=HORIZON, inputs=contexts)
        forecast_block = np.asarray(point_forecast, dtype=np.float32)[:, :HORIZON]
        step_forecasts.append(forecast_block)
        contexts = [
            np.concatenate([context, forecast_values])[-CONTEXT_LEN:]
            for context, forecast_values in zip(contexts, forecast_block)
        ]

    return np.concatenate(step_forecasts, axis=1)


def main():
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    windows = load_daily_timesfm_price_windows(
        csv_path=DATA_PATH,
        context_len=CONTEXT_LEN,
        horizon=TOTAL_FORECAST_STEPS,
        expected_minutes=INTERVAL_MINUTES,
        forecast_start_hour=FORECAST_START_HOUR,
        forecast_start_minute=FORECAST_START_MINUTE,
        start_date=FORECAST_START_DATE,
        end_date=FORECAST_END_DATE,
    )
    print(f"Daily forecast windows: {len(windows)}")

    model = build_model()
    forecasts = []
    for batch_start, batch_windows in iter_batches(windows, BATCH_SIZE):
        forecasts.extend(rolling_forecast_batch(model, batch_windows))
        print(f"Predicted windows {batch_start} to {batch_start + len(batch_windows) - 1}")

    actual = np.concatenate([window.future_price for window in windows])
    predicted = np.concatenate(forecasts)
    mae = float(np.mean(np.abs(predicted - actual)))

    save_predictions(windows, forecasts)
    save_run_config(window_count=len(windows))

    print(f"Saved predictions: {PREDICTION_CSV}")
    print(f"Saved run config: {RUN_CONFIG_JSON}")
    print(f"MAE over {len(actual)} steps: {mae:.4f}")
    print("First predictions:")
    first_window = windows[0]
    first_forecast = forecasts[0]
    for idx in range(min(PREVIEW_ROWS, len(first_forecast))):
        print(
            f"{first_window.future_datetime.iloc[idx]} "
            f"pred={first_forecast[idx]:.3f} actual={first_window.future_price[idx]:.3f}"
        )


if __name__ == "__main__":
    main()
