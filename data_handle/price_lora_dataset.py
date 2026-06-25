from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PriceSplit:
    datetime: pd.Series
    price: np.ndarray


class PriceWindowDataset(Dataset):
    def __init__(
        self,
        split: PriceSplit,
        context_len: int,
        horizon_len: int,
        expected_minutes: int = 15,
        require_continuous: bool = True,
    ):
        self.datetime = split.datetime.reset_index(drop=True)
        self.price = split.price.astype(np.float32, copy=False)
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.expected_minutes = expected_minutes
        self.valid_starts = self._build_valid_starts(require_continuous)

        if not self.valid_starts:
            raise ValueError("no valid price windows found for this split")

    def _build_valid_starts(self, require_continuous: bool) -> list[int]:
        total_len = self.context_len + self.horizon_len
        if len(self.price) < total_len:
            return []

        if not require_continuous:
            return list(range(len(self.price) - total_len + 1))

        deltas = self.datetime.diff().dt.total_seconds().div(60).to_numpy()
        continuous_edge = np.ones(len(self.datetime), dtype=bool)
        continuous_edge[1:] = deltas[1:] == self.expected_minutes

        valid_starts = []
        for start in range(len(self.price) - total_len + 1):
            end = start + total_len
            if continuous_edge[start + 1:end].all():
                valid_starts.append(start)
        return valid_starts

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = self.valid_starts[index]
        context_end = start + self.context_len
        future_end = context_end + self.horizon_len
        return {
            "past_values": torch.from_numpy(self.price[start:context_end]),
            "future_values": torch.from_numpy(self.price[context_end:future_end]),
        }


def load_price_splits(
    csv_path: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    datetime_col: str = "datetime",
    price_col: str = "price",
) -> tuple[PriceSplit, PriceSplit, PriceSplit]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    missing = [col for col in (datetime_col, price_col) if col not in df.columns]
    if missing:
        raise ValueError(f"missing required columns in {csv_path}: {', '.join(missing)}")

    df = df[[datetime_col, price_col]].copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.sort_values(datetime_col).reset_index(drop=True)

    df[price_col] = df[price_col].ffill()
    df = df.dropna(subset=[price_col]).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"no valid price rows found in {csv_path}")

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    if train_end <= 0 or val_end <= train_end or val_end >= n:
        raise ValueError("invalid split ratios for dataset length")

    return (
        _make_split(df.iloc[:train_end], datetime_col, price_col),
        _make_split(df.iloc[train_end:val_end], datetime_col, price_col),
        _make_split(df.iloc[val_end:], datetime_col, price_col),
    )


def _make_split(df: pd.DataFrame, datetime_col: str, price_col: str) -> PriceSplit:
    return PriceSplit(
        datetime=df[datetime_col].reset_index(drop=True),
        price=df[price_col].to_numpy(dtype=np.float32),
    )
