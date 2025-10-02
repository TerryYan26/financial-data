
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np

@dataclass
class PreprocessConfig:
    date_col: str = "date"
    price_col: str = "close"
    index_utc: bool = False
    missing_strategy: str = "ffill_bfill"      # 'ffill_bfill' | 'interpolate_time' | 'none'
    outlier_strategy: str = "iqr"              # 'iqr' | 'mad' | 'zscore' | 'none'
    outlier_threshold: float = 3.0             # For zscore/MAD; for IQR it's the k-multiplier
    normalize_strategy: str = "zscore"         # 'zscore' | 'minmax' | 'robust' | 'none'
    train_end: Optional[pd.Timestamp] = None   # Fit scalers up to this timestamp (inclusive)

@dataclass
class FinancePreprocessor:
    config: PreprocessConfig
    fitted_: bool = field(default=False, init=False)
    norm_params_: Dict[str, Any] = field(default_factory=dict, init=False)

    def _coerce_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.config.date_col in df.columns:
            df[self.config.date_col] = pd.to_datetime(df[self.config.date_col], utc=self.config.index_utc)
            df = df.sort_values(self.config.date_col).set_index(self.config.date_col)
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a datetime index or a date_col")
        else:
            if self.config.index_utc:
                df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
            df = df.sort_index()
        return df

    def _handle_missing(self, s: pd.Series) -> pd.Series:
        if self.config.missing_strategy == "ffill_bfill":
            return s.ffill().bfill()
        elif self.config.missing_strategy == "interpolate_time":
            return s.interpolate(method="time").ffill().bfill()
        elif self.config.missing_strategy == "none":
            return s
        else:
            raise ValueError(f"Unknown missing strategy: {self.config.missing_strategy}")

    def _remove_outliers_series(self, s: pd.Series) -> pd.Series:
        method = self.config.outlier_strategy
        if method == "none":
            return s

        x = s.astype(float).copy()
        if x.dropna().empty:
            return x

        if method == "iqr":
            q1 = x.quantile(0.25)
            q3 = x.quantile(0.75)
            iqr = q3 - q1
            k = self.config.outlier_threshold
            lower, upper = q1 - k * iqr, q3 + k * iqr
            mask = (x < lower) | (x > upper)

        elif method == "mad":
            median = x.median()
            mad = (np.abs(x - median)).median()
            if mad == 0:
                return x
            score = 0.6745 * (x - median) / mad
            mask = np.abs(score) > self.config.outlier_threshold

        elif method == "zscore":
            mean = x.mean()
            std = x.std(ddof=0)
            if std == 0:
                return x
            z = (x - mean) / std
            mask = np.abs(z) > self.config.outlier_threshold
        else:
            raise ValueError(f"Unknown outlier strategy: {method}")

        # Winsorize extreme points to nearest non-outlier boundary
        if mask.any():
            non_outliers = x[~mask]
            lo, hi = non_outliers.min(), non_outliers.max()
            x[mask & x.notna()] = np.clip(x[mask], lo, hi)
        return x

    def _fit_normalizer(self, s: pd.Series) -> None:
        if self.config.normalize_strategy == "none":
            self.norm_params_.clear()
            return
        x = s.dropna()
        if x.empty:
            self.norm_params_.clear()
            return
        if self.config.normalize_strategy == "zscore":
            self.norm_params_ = {"mean": float(x.mean()), "std": float(x.std(ddof=0))}
        elif self.config.normalize_strategy == "minmax":
            self.norm_params_ = {"min": float(x.min()), "max": float(x.max())}
        elif self.config.normalize_strategy == "robust":
            self.norm_params_ = {"q1": float(x.quantile(0.25)), "q3": float(x.quantile(0.75))}
        else:
            raise ValueError(f"Unknown normalize strategy: {self.config.normalize_strategy}")

    def _apply_normalizer(self, s: pd.Series) -> pd.Series:
        if not self.norm_params_ or self.config.normalize_strategy == "none":
            return s
        if self.config.normalize_strategy == "zscore":
            mean, std = self.norm_params_["mean"], self.norm_params_["std"]
            std = std if std != 0 else 1.0
            return (s - mean) / std
        elif self.config.normalize_strategy == "minmax":
            mn, mx = self.norm_params_["min"], self.norm_params_["max"]
            rng = (mx - mn) if (mx - mn) != 0 else 1.0
            return (s - mn) / rng
        elif self.config.normalize_strategy == "robust":
            q1, q3 = self.norm_params_["q1"], self.norm_params_["q3"]
            iqr = (q3 - q1) if (q3 - q1) != 0 else 1.0
            return (s - q1) / iqr
        else:
            return s

    def fit(self, df: pd.DataFrame) -> "FinancePreprocessor":
        df = self._coerce_datetime_index(df)
        s = df[self.config.price_col].copy()
        s = self._handle_missing(s)
        s = self._remove_outliers_series(s)

        if self.config.train_end is not None:
            s_fit = s.loc[: self.config.train_end]
        else:
            s_fit = s
        self._fit_normalizer(s_fit)
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("Call fit() before transform().")
        df = self._coerce_datetime_index(df)
        out = df.copy()
        s = out[self.config.price_col].copy()

        s = self._handle_missing(s)
        s = self._remove_outliers_series(s)
        s_norm = self._apply_normalizer(s)

        out[self.config.price_col + "_clean"] = s
        out[self.config.price_col + "_norm"] = s_norm
        out["ret_1d"] = out[self.config.price_col + "_clean"].pct_change()
        out["logret_1d"] = np.log(out[self.config.price_col + "_clean"]).diff()
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
