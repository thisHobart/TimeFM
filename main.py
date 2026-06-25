import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import timesfm

from data_handle.price_loader import TimesFMPriceWindow, load_price_series


REPO_ROOT = Path(__file__).resolve().parent
DATA_PATH = REPO_ROOT / "datasets" / "datasets.csv"
MODEL_PATH = REPO_ROOT / "timesfm"
RESULT_DIR = REPO_ROOT / "result" / "train_data1"
print("RESULT_DIR:", RESULT_DIR)
PREDICTION_CSV = RESULT_DIR / "timesfm_price_predictions.csv"
RUN_CONFIG_JSON = RESULT_DIR / "timesfm_run_config.json"
CONTEXT_LEN = 672
SAVE_START_OFFSET_STEPS = 6
SAVE_STEPS = 96
HORIZON = SAVE_START_OFFSET_STEPS + SAVE_STEPS
ROLLING_STEPS = 1
TOTAL_FORECAST_STEPS = HORIZON * ROLLING_STEPS
INTERVAL_MINUTES = 15
ALLOW_TIME_GAPS = True
FORECAST_START_HOUR = 22
FORECAST_START_MINUTE = 45
FORECAST_START_DATE = "2026-06-14"
FORECAST_END_DATE = None
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


def save_run_config(window_count):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "model": "TimesFM_2p5_200M_torch",
        "model_path": str(MODEL_PATH),
        "data_path": str(DATA_PATH),
        "context_len": CONTEXT_LEN,
        "save_start_offset_steps": SAVE_START_OFFSET_STEPS,
        "save_steps": SAVE_STEPS,
        "horizon": HORIZON,
        "rolling_steps": ROLLING_STEPS,
        "total_forecast_steps": TOTAL_FORECAST_STEPS,
        "interval_minutes": INTERVAL_MINUTES,
        "allow_time_gaps": ALLOW_TIME_GAPS,
        "forecast_start_hour": FORECAST_START_HOUR,
        "forecast_start_minute": FORECAST_START_MINUTE,
        "forecast_start_date": FORECAST_START_DATE,
        "forecast_end_date": FORECAST_END_DATE,
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
        save_datetime, save_actual, save_forecast = get_saved_prediction_slice(window, forecast)
        forecast_start_day = save_datetime.iloc[0].date().isoformat()
        for point_index, (timestamp, actual, predicted) in enumerate(
            zip(save_datetime, save_actual, save_forecast),
            start=1,
        ):
            rows.append(
                {
                    "预测窗口": window_index,
                    "预测起始日": forecast_start_day,
                    "预测点": point_index,
                    "时间": timestamp.strftime("%Y-%m-%d %H:%M"),
                    "真实电价": format_optional_number(actual),
                    "预测电价": round(float(predicted), 3),
                    "误差": format_optional_number(predicted - actual),
                }
            )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(PREDICTION_CSV, index=False, encoding="utf-8-sig")


def get_saved_prediction_slice(window, forecast):
    start = SAVE_START_OFFSET_STEPS
    end = start + SAVE_STEPS
    return (
        window.future_datetime.iloc[start:end].reset_index(drop=True),
        window.future_price[start:end],
        forecast[start:end],
    )


def format_optional_number(value):
    value = float(value)
    if not np.isfinite(value):
        return ""
    return round(value, 3)


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


def build_forecast_start_times(series):
    start_day = pd.to_datetime(FORECAST_START_DATE).normalize()
    end_day = (
        pd.to_datetime(FORECAST_END_DATE).normalize()
        if FORECAST_END_DATE
        else series.datetime.iloc[-1].normalize()
    )
    start_time = pd.Timedelta(hours=FORECAST_START_HOUR, minutes=FORECAST_START_MINUTE)
    return [day + start_time for day in pd.date_range(start_day, end_day, freq="D")]


def build_forecast_window(series, forecast_start):
    context_times = pd.date_range(
        end=forecast_start - pd.Timedelta(minutes=INTERVAL_MINUTES),
        periods=CONTEXT_LEN,
        freq=f"{INTERVAL_MINUTES}min",
    )
    future_datetime = pd.Series(
        pd.date_range(
            start=forecast_start,
            periods=TOTAL_FORECAST_STEPS,
            freq=f"{INTERVAL_MINUTES}min",
        )
    )
    actual_by_time = pd.Series(series.price, index=series.datetime)

    context = [
        get_actual_context_value(timestamp, actual_by_time)
        for timestamp in context_times
    ]

    return TimesFMPriceWindow(
        context=np.asarray(context, dtype=np.float32),
        horizon=TOTAL_FORECAST_STEPS,
        context_datetime=pd.Series(context_times),
        future_datetime=future_datetime,
        future_price=actual_by_time.reindex(future_datetime).to_numpy(dtype=np.float32),
    )


def get_actual_context_value(timestamp, actual_by_time):
    if timestamp in actual_by_time.index and np.isfinite(actual_by_time.loc[timestamp]):
        return actual_by_time.loc[timestamp]
    return get_latest_available_actual(timestamp, actual_by_time)


def get_latest_available_actual(timestamp, actual_by_time):
    history = actual_by_time.loc[:timestamp].dropna()
    if history.empty:
        raise ValueError(f"no actual price available at or before {timestamp}")
    return history.iloc[-1]


def main():
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    if TOTAL_FORECAST_STEPS < SAVE_START_OFFSET_STEPS + SAVE_STEPS:
        raise ValueError("TOTAL_FORECAST_STEPS must cover the saved prediction range")

    series = load_price_series(DATA_PATH)
    forecast_starts = build_forecast_start_times(series)
    print(f"Daily forecast windows: {len(forecast_starts)}")

    model = build_model()
    windows = []
    forecasts = []
    for window_index, forecast_start in enumerate(forecast_starts):
        window = build_forecast_window(series, forecast_start)
        forecast = rolling_forecast_batch(model, [window])[0]
        windows.append(window)
        forecasts.append(forecast)
        print(f"Predicted window {window_index}: {forecast_start}")

    actual = np.concatenate([
        get_saved_prediction_slice(window, forecast)[1]
        for window, forecast in zip(windows, forecasts)
    ])
    predicted = np.concatenate([
        get_saved_prediction_slice(window, forecast)[2]
        for window, forecast in zip(windows, forecasts)
    ])
    valid_actual = np.isfinite(actual)
    if not valid_actual.any():
        raise ValueError("no actual prices are available for MAE calculation")
    mae = float(np.mean(np.abs(predicted[valid_actual] - actual[valid_actual])))

    save_predictions(windows, forecasts)
    save_run_config(window_count=len(windows))

    print(f"Saved predictions: {PREDICTION_CSV}")
    print(f"Saved run config: {RUN_CONFIG_JSON}")
    print(f"MAE over {valid_actual.sum()} actual points: {mae:.4f}")
    print("First predictions:")
    first_window = windows[0]
    first_datetime, first_actual, first_forecast = get_saved_prediction_slice(first_window, forecasts[0])
    for idx in range(min(PREVIEW_ROWS, len(first_forecast))):
        print(
            f"{first_datetime.iloc[idx]} "
            f"pred={first_forecast[idx]:.3f} actual={first_actual[idx]:.3f}"
        )


if __name__ == "__main__":
    main()
