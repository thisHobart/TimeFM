from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "datasets" / "datasets.csv"


@dataclass(frozen=True)
class PriceSeries:
    datetime: pd.Series
    price: np.ndarray


@dataclass(frozen=True)
class TimesFMPriceWindow:
    context: np.ndarray
    horizon: int
    context_datetime: pd.Series
    future_datetime: pd.Series
    future_price: np.ndarray

    def as_timesfm_inputs(self) -> list[np.ndarray]:
        return [self.context]


def load_price_series(
    csv_path: str | Path = DEFAULT_DATA_PATH,
    datetime_col: str = "datetime",
    price_col: str = "price",
) -> PriceSeries:
    path = Path(csv_path)
    df = pd.read_csv(path, encoding="utf-8-sig")

    missing = [col for col in (datetime_col, price_col) if col not in df.columns]
    if missing:
        raise ValueError(f"missing required columns in {path}: {', '.join(missing)}")

    df = df[[datetime_col, price_col]].copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.sort_values(datetime_col).reset_index(drop=True)

    # Forward fill uses past values only. Drop leading gaps to avoid leaking future prices.
    df[price_col] = df[price_col].ffill()
    df = df.dropna(subset=[price_col]).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"no valid price rows found in {path}")

    return PriceSeries(
        datetime=df[datetime_col],
        price=df[price_col].to_numpy(dtype=np.float32),
    )


def build_timesfm_price_window(
    series: PriceSeries,
    context_len: int = 1024,
    horizon: int = 128,
    cutoff_index: int | None = None,
    expected_minutes: int = 15,
    allow_time_gaps: bool = False,
) -> TimesFMPriceWindow:
    if context_len <= 0:
        raise ValueError("context_len must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    if cutoff_index is None:
        cutoff_index = len(series.price) - horizon

    context_start = cutoff_index - context_len
    future_end = cutoff_index + horizon

    if context_start < 0:
        raise ValueError(
            f"not enough history: need {context_len} rows before cutoff_index {cutoff_index}"
        )
    if not allow_time_gaps and future_end > len(series.price):
        raise ValueError(
            f"not enough future rows for evaluation: need horizon {horizon} from cutoff_index {cutoff_index}"
        )

    if not allow_time_gaps:
        window_datetime = series.datetime.iloc[context_start:future_end].reset_index(drop=True)
        _assert_continuous_datetime(window_datetime, expected_minutes)

    future_datetime = pd.Series(
        pd.date_range(
            start=series.datetime.iloc[cutoff_index],
            periods=horizon,
            freq=f"{expected_minutes}min",
        )
    )
    price_by_datetime = pd.Series(series.price, index=series.datetime)

    return TimesFMPriceWindow(
        context=series.price[context_start:cutoff_index],
        horizon=horizon,
        context_datetime=series.datetime.iloc[context_start:cutoff_index].reset_index(drop=True),
        future_datetime=future_datetime,
        future_price=price_by_datetime.reindex(future_datetime).to_numpy(dtype=np.float32),
    )


def load_timesfm_price_window(
    csv_path: str | Path = DEFAULT_DATA_PATH,
    context_len: int = 1024,
    horizon: int = 128,
    cutoff_index: int | None = None,
    allow_time_gaps: bool = False,
) -> TimesFMPriceWindow:
    series = load_price_series(csv_path)
    return build_timesfm_price_window(
        series=series,
        context_len=context_len,
        horizon=horizon,
        cutoff_index=cutoff_index,
        allow_time_gaps=allow_time_gaps,
    )


def build_daily_timesfm_price_windows(
    series: PriceSeries,
    context_len: int = 1024,
    horizon: int = 96,
    expected_minutes: int = 15,
    forecast_start_hour: int = 0,
    forecast_start_minute: int = 15,
    start_date: str | None = None,
    end_date: str | None = None,
    allow_time_gaps: bool = False,
) -> list[TimesFMPriceWindow]:
    start_ts = pd.to_datetime(start_date) if start_date else None
    end_ts = pd.to_datetime(end_date) if end_date else None
    windows = []

    for cutoff_index in range(context_len, len(series.price)):
        forecast_start = series.datetime.iloc[cutoff_index]
        if forecast_start.hour != forecast_start_hour:
            continue
        if forecast_start.minute != forecast_start_minute:
            continue
        if start_ts is not None and forecast_start.normalize() < start_ts.normalize():
            continue
        if end_ts is not None and forecast_start.normalize() > end_ts.normalize():
            continue

        if not allow_time_gaps:
            future_end = cutoff_index + horizon
            if future_end > len(series.price):
                continue
            window_datetime = series.datetime.iloc[
                cutoff_index - context_len:future_end
            ].reset_index(drop=True)
            if _find_first_datetime_gap(window_datetime, expected_minutes) is not None:
                continue

        window = build_timesfm_price_window(
            series=series,
            context_len=context_len,
            horizon=horizon,
            cutoff_index=cutoff_index,
            expected_minutes=expected_minutes,
            allow_time_gaps=allow_time_gaps,
        )
        windows.append(window)

    if not windows:
        raise ValueError("no valid daily forecast windows found")

    return windows


def load_daily_timesfm_price_windows(
    csv_path: str | Path = DEFAULT_DATA_PATH,
    context_len: int = 1024,
    horizon: int = 96,
    expected_minutes: int = 15,
    forecast_start_hour: int = 0,
    forecast_start_minute: int = 15,
    start_date: str | None = None,
    end_date: str | None = None,
    allow_time_gaps: bool = False,
) -> list[TimesFMPriceWindow]:
    series = load_price_series(csv_path)
    return build_daily_timesfm_price_windows(
        series=series,
        context_len=context_len,
        horizon=horizon,
        expected_minutes=expected_minutes,
        forecast_start_hour=forecast_start_hour,
        forecast_start_minute=forecast_start_minute,
        start_date=start_date,
        end_date=end_date,
        allow_time_gaps=allow_time_gaps,
    )


def _assert_continuous_datetime(datetimes: pd.Series, expected_minutes: int) -> None:
    first_bad_pos = _find_first_datetime_gap(datetimes, expected_minutes)
    if first_bad_pos is None:
        return

    raise ValueError(
        "datetime is not continuous at "
        f"{datetimes.iloc[first_bad_pos - 1]} -> {datetimes.iloc[first_bad_pos]}"
    )


def _find_first_datetime_gap(datetimes: pd.Series, expected_minutes: int) -> int | None:
    if len(datetimes) < 2:
        return None

    deltas = datetimes.diff().dt.total_seconds().div(60)
    bad_rows = deltas.iloc[1:] != expected_minutes
    if bad_rows.any():
        return int(bad_rows[bad_rows].index[0])
    return None
