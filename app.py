"""
WeightNet Inference API — FastAPI server for Railway deployment
Receives raw price/volume data (90 trading days), computes features,
runs 3-seed WeightNet ensemble, returns portfolio weights for DOW 30.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weightnet-api")

# ──────────────────────────────────────────────────────────────────────
# Constants (must match training)
# ──────────────────────────────────────────────────────────────────────
ESP = 1e-8
SEQ_LEN = 30
N_SEEDS = 3
N_STOCKS = 30
MODEL_DIR = Path("models")

FINAL_FEATURES = [
    "mav_volume_4d", "log_volume", "vol_to_mav_1d", "logret_std_21d",
    "bb_width_63d", "prices_to_ma_1d", "logret_std_63d", "vol_of_vol",
    "bb_width_21d", "obv_ret_63d_r63", "logret_std_5d", "obv_ret_4d_r1",
    "obv_ret_63d_r21", "obv_ret_21d_r21", "logret_std_4d", "bb_width_5d",
    "obv_ret_2d_r1", "obv_ret_2d_r63", "bb_width_4d", "bb_width_3d",
    "logret_std_3d", "obv_ret_21d_r63", "obv_ret_2d_r21", "bb_pct_b_2d",
    "rsi_21d",
]

# DOW 30 tickers (must match training order exactly)
STOCK_LST = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX",
    "DIS", "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM",
    "MRK", "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V",
    "VZ", "WMT",
]

# ──────────────────────────────────────────────────────────────────────
# Model definition (must match training exactly)
# ──────────────────────────────────────────────────────────────────────
class WeightNet(nn.Module):
    def __init__(self, n_feat, hidden=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1),
        )

    def forward(self, X):
        B, L, N, F = X.shape
        return torch.softmax(
            self.net(X.reshape(B * L * N, F)).reshape(B, L, N), dim=-1
        )


# ──────────────────────────────────────────────────────────────────────
# Feature engineering functions (from model_v05.py)
# ──────────────────────────────────────────────────────────────────────
def prepare_return(x, w):
    return x.pct_change(w)


def prepare_volatility(x, w):
    return x.rolling(w).std() if w >= 2 else x.abs()


def prepare_ratio_to_ma(x, w):
    return x / (x.rolling(w).mean() + ESP) - 1


def prepare_bollinger(x, w):
    mid = x.rolling(w).mean()
    std = x.rolling(w).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    return (upper - lower) / (mid + ESP), (x - lower) / (upper - lower + ESP)


def prepare_rsi(x, w):
    d = x.diff()
    gain = d.clip(lower=0).rolling(w).mean()
    loss = (-d.clip(upper=0)).rolling(w).mean()
    return 100 - 100 / (1 + gain / (loss + ESP))


def prepare_obv(ret, vol, w):
    obv = (np.sign(ret) * vol).cumsum()
    return obv / (obv.rolling(w).mean().abs() + ESP)


def build_features(prices: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """
    Build the 25 final features from raw prices and volume.
    Input: prices & volume DataFrames with DatetimeIndex and STOCK_LST columns.
    Output: stacked DataFrame with columns [Date, Ticker, <25 features>].
    """
    log_ret = np.log(prices / prices.shift(1))
    feat = {}

    # --- Rolling window features ---
    windows_needed = {1, 2, 3, 4, 5, 21, 63}

    for w in windows_needed:
        feat[f"logret_std_{w}d"] = prepare_volatility(log_ret, w)
        feat[f"mav_volume_{w}d"] = volume.rolling(w).mean()
        feat[f"vol_to_mav_{w}d"] = prepare_ratio_to_ma(volume, w)
        bb_w, bb_p = prepare_bollinger(prices, w)
        feat[f"bb_width_{w}d"] = bb_w
        feat[f"bb_pct_b_{w}d"] = bb_p

    feat["prices_to_ma_1d"] = prepare_ratio_to_ma(prices, 1)
    feat["rsi_21d"] = prepare_rsi(prices, 21)
    feat["log_volume"] = np.log(volume + 1)
    feat["vol_of_vol"] = feat["logret_std_5d"].rolling(21).std()

    # --- OBV features ---
    ret_windows = {2: "ret_2d", 4: "ret_4d", 21: "ret_21d", 63: "ret_63d"}
    for w, rk in ret_windows.items():
        feat[rk] = prepare_return(prices, w)

    obv_combos = [
        ("ret_2d", 1), ("ret_2d", 21), ("ret_2d", 63),
        ("ret_4d", 1),
        ("ret_21d", 21), ("ret_21d", 63),
        ("ret_63d", 21), ("ret_63d", 63),
    ]
    for rk, r in obv_combos:
        feat[f"obv_{rk}_r{r}"] = prepare_obv(feat[rk], volume, r)

    # --- Keep only the 25 final features ---
    feat_final = {k: feat[k] for k in FINAL_FEATURES if k in feat}

    # --- Stack into long format ---
    stacked = pd.concat(
        {k: v.stack() for k, v in feat_final.items()}, axis=1
    ).rename_axis(["Date", "Ticker"]).reset_index().sort_values(["Ticker", "Date"])

    return stacked


def cs_normalize(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Cross-sectional normalization (same as training)."""
    df = df.copy()
    for col in cols:
        g = df.groupby("Date")[col]
        df[col] = (df[col] - g.transform("mean")) / (g.transform("std") + ESP)
    return df


def df_to_tensor(df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """
    Convert long-format DataFrame to (T, N_stocks, N_features) array.
    Must maintain the same stock ordering as training.
    """
    piv = df.pivot(index="Date", columns="Ticker").fillna(0.0)
    T = len(piv)
    N = len(STOCK_LST)
    F = len(feature_cols)
    X = np.zeros((T, N, F), dtype=np.float32)
    for i, tk in enumerate(STOCK_LST):
        if isinstance(piv.columns, pd.MultiIndex):
            try:
                X[:, i, :] = piv[feature_cols].xs(tk, level="Ticker", axis=1).values
            except KeyError:
                logger.warning(f"Ticker {tk} missing from data, using zeros")
    return X


# ──────────────────────────────────────────────────────────────────────
# Load models at startup
# ──────────────────────────────────────────────────────────────────────
device = torch.device("cpu")  # Railway free tier = CPU
n_features = len(FINAL_FEATURES)

logger.info("Loading scaler and WeightNet ensemble ...")
scaler_X = joblib.load(MODEL_DIR / "scaler_X.pkl")

weightnet_models = []
for i in range(N_SEEDS):
    model = WeightNet(n_features).to(device)
    state = torch.load(MODEL_DIR / f"weightnet_seed{i}.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    weightnet_models.append(model)
    logger.info(f"  Loaded weightnet_seed{i}.pt")

logger.info("All models loaded ✓")

# ──────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="WeightNet Portfolio API",
    description="Weekly inference for DOW 30 portfolio allocation",
    version="1.0.0",
)


class MarketDataRequest(BaseModel):
    """
    Raw market data sent from n8n.
    - dates: list of date strings (YYYY-MM-DD), length ~90
    - prices: dict of {ticker: [price_list]}, 30 tickers × ~90 days
    - volumes: dict of {ticker: [volume_list]}, same shape
    """
    dates: list[str]
    prices: dict[str, list[float]]
    volumes: dict[str, list[float]]


class WeightResponse(BaseModel):
    inference_date: str
    weights: dict[str, float]
    model_version: str = "v5.0-weightnet"
    n_seeds: int = N_SEEDS
    status: str = "success"


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": len(weightnet_models),
        "features": len(FINAL_FEATURES),
        "stocks": len(STOCK_LST),
    }


@app.post("/predict", response_model=WeightResponse)
def predict_weights(req: MarketDataRequest):
    try:
        logger.info(f"Received data: {len(req.dates)} dates, {len(req.prices)} tickers")

        # ---- 1. Parse raw data into DataFrames ----
        dates = pd.to_datetime(req.dates)
        prices = pd.DataFrame(req.prices, index=dates).sort_index()
        volumes = pd.DataFrame(req.volumes, index=dates).sort_index()

        # Validate
        missing = [t for t in STOCK_LST if t not in prices.columns]
        if missing:
            logger.warning(f"Missing tickers: {missing}. Filling with NaN.")
        prices = prices.reindex(columns=STOCK_LST).ffill(limit=2)
        volumes = volumes.reindex(columns=STOCK_LST).fillna(0)

        if len(prices) < 60:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least 60 trading days, got {len(prices)}",
            )

        # ---- 2. Build features ----
        feat_df = build_features(prices, volumes)

        logger.info(f"  Features built: {feat_df.shape}, "
                     f"NaN counts: {feat_df[FINAL_FEATURES].isna().sum().to_dict()}")

        # Feature-type-aware NaN fill (matches training code in model_v05.py)
        rsi_cols = [c for c in FINAL_FEATURES if "rsi_" in c]
        obv_cols = [c for c in FINAL_FEATURES if "obv_" in c]
        other_cols = [c for c in FINAL_FEATURES if c not in rsi_cols + obv_cols]

        if rsi_cols:
            feat_df[rsi_cols] = feat_df[rsi_cols].fillna(50)
        if obv_cols:
            feat_df[obv_cols] = feat_df[obv_cols].fillna(0)
        if other_cols:
            feat_df[other_cols] = feat_df[other_cols].fillna(0)

        # Drop rows where ALL features are zero/NaN (truly empty rows)
        feat_df = feat_df.dropna(subset=FINAL_FEATURES, how="all")

        # Keep last SEQ_LEN dates
        available_dates = sorted(feat_df["Date"].unique())
        logger.info(f"  Available dates after NaN fill: {len(available_dates)}")
        if len(available_dates) < SEQ_LEN:
            raise HTTPException(
                status_code=400,
                detail=f"Only {len(available_dates)} valid feature dates after "
                       f"rolling window computation. Need {SEQ_LEN}. Send more data.",
            )
        keep_dates = available_dates[-SEQ_LEN:]
        feat_df = feat_df[feat_df["Date"].isin(keep_dates)]

        # ---- 3. Cross-sectional normalize ----
        feat_df = cs_normalize(feat_df, FINAL_FEATURES)

        # ---- 4. Build tensor (T, N, F) ----
        X_raw = df_to_tensor(feat_df, FINAL_FEATURES)

        # ---- 5. Scale with training scaler ----
        T, N, F = X_raw.shape
        X_scaled = scaler_X.transform(X_raw.reshape(-1, F)).reshape(T, N, F)

        # ---- 6. Run 3-seed ensemble ----
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        # Shape: (1, T, N, F)

        weight_preds = []
        for model in weightnet_models:
            with torch.no_grad():
                w = model(X_tensor).squeeze(0).cpu().numpy()  # (T, N)
            weight_preds.append(w)

        # Average across seeds, take last timestep's weights
        w_ensemble = np.mean(weight_preds, axis=0)  # (T, N)
        w_final = w_ensemble[-1]  # last day = current allocation
        w_final = w_final / (w_final.sum() + 1e-12)  # re-normalize

        weights_dict = {
            ticker: round(float(w_final[i]), 6) for i, ticker in enumerate(STOCK_LST)
        }

        logger.info(f"Inference complete. Top 5 weights: "
                     f"{sorted(weights_dict.items(), key=lambda x: -x[1])[:5]}")

        return WeightResponse(
            inference_date=str(keep_dates[-1].date()),
            weights=weights_dict,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
