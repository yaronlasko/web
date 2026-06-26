"""Central configuration: API endpoints, paths, and hyperparameters.

Every API fact here was verified against the live Polymarket API (June 2026),
not from memory. See README.md for the verification notes.

Mission: predict a market's ACCURACY (P it resolves to its favorite) from
engineered market-quality / activity features (bet size, volume, #traders, ...),
and rank which features matter via permutation importance.
"""
from pathlib import Path

# --- Paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"

MARKETS_CACHE = RAW_DIR / "markets.json"           # resolved market metadata
PRICES_DIR = RAW_DIR / "prices"                    # <market_id>.json: price history
TRADES_DIR = RAW_DIR / "trades"                    # <market_id>.json: trade aggregates

PROCESSED_NPZ = PROCESSED_DIR / "dataset.npz"

CHECKPOINT = MODELS_DIR / "best.pt"
SCALER_PATH = MODELS_DIR / "scaler.npz"
CALIBRATION_PLOT = MODELS_DIR / "calibration.png"
IMPORTANCE_PLOT = MODELS_DIR / "feature_importance.png"

for _d in (RAW_DIR, PRICES_DIR, TRADES_DIR, PROCESSED_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- API ----------------------------------------------------------------
GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
DATA_API_BASE = "https://data-api.polymarket.com"
USER_AGENT = "polymarket-dl-research/1.0 (educational)"   # urllib gets 403 without a UA
GAMMA_PAGE_SIZE = 100        # verified: page size capped at 100
REQUEST_SLEEP = 0.3          # polite delay between calls (seconds)
MAX_RETRIES = 5
# CLOB /prices-history rejects too-fine fidelity (empty when points exceed ~1000);
# fidelity is chosen per market from its duration to target ~TARGET_PRICE_POINTS.
TARGET_PRICE_POINTS = 500
FIDELITY_FALLBACKS = 4
# Data API /trades: page size caps at 1000; we sample the most recent TRADES_SAMPLE
# trades per market and compute aggregate features from them.
TRADES_SAMPLE = 1000
ACTIVE_PRICE_LO = 0.02       # exclude near-settlement trades from bet-size stats
ACTIVE_PRICE_HI = 0.98

# --- Dataset construction -----------------------------------------------
CUTOFF_FRACTION = 0.5        # price-shape features use data up to 50% of trading life
MIN_PRICE_POINTS = 8         # drop markets with fewer pre-cutoff price points

# Engineered tabular features (order defines the model's input columns).
# Monetary/count features are log1p-scaled (name prefixed log_) to tame heavy tails.
FEATURES = [
    # --- price / favorite strength (from price history, up to the 50% cutoff) ---
    "fav_prob",            # implied prob of the favorite at cutoff = max(p, 1-p)
    "price_volatility",    # std of pre-cutoff price returns
    "price_trend",         # OLS slope of price vs normalized time
    "n_price_points",      # number of pre-cutoff price points
    "frac_time_fav_led",   # fraction of pre-cutoff time the cutoff-favorite was leading
    "price_range",         # max-min of pre-cutoff price (swing)
    # --- volume (Gamma) ---
    "log_volume",          # log1p(total USDC volume)
    # --- trade activity (Data API /trades sample) ---
    "log_avg_bet_size",    # log1p mean $ of (non-settlement) trades
    "log_median_bet_size",
    "log_max_bet_size",    # whale presence
    "log_bet_size_std",
    "log_n_trades",        # log1p(#trades in sample, capped at TRADES_SAMPLE)
    "log_n_traders",       # log1p(distinct wallets in sample)
    "top_trader_share",    # top wallet's share of sample $ volume (concentration)
    "buy_share",           # fraction of BUY-side trades
    "trades_per_day",      # sample #trades / duration
    # --- structural ---
    "duration_days",
    "neg_risk",
]
N_FEATURES = len(FEATURES)

# Market categories (one-hot appended to the feature vector + used for breakdowns).
CATEGORIES = ["sports", "politics", "crypto", "tech", "business", "culture", "other"]
CAT_FEATURES = [f"cat_{c}" for c in CATEGORIES]
ALL_FEATURES = FEATURES + CAT_FEATURES
CATEGORIES_CACHE = RAW_DIR / "categories.json"     # {market_id: [tag labels]}

# Feature subsets for the full-vs-activity comparison. PRICE_FEATURES are the ones
# derived from the price chart (they reveal how strong the favorite is); the
# 'activity' model drops them and predicts accuracy from trading/structural data only.
# Category one-hots are included in BOTH (they are structural metadata, not price).
PRICE_FEATURES = [
    "fav_prob", "price_volatility", "price_trend",
    "n_price_points", "frac_time_fav_led", "price_range",
]
ACTIVITY_FEATURES = [f for f in FEATURES if f not in PRICE_FEATURES]
FEATURE_SETS = {
    "full": FEATURES + CAT_FEATURES,
    "activity": ACTIVITY_FEATURES + CAT_FEATURES,
}


def ckpt_path(mode):
    return MODELS_DIR / f"best_{mode}.pt"


def scaler_path(mode):
    return MODELS_DIR / f"scaler_{mode}.npz"


def importance_plot(mode):
    return MODELS_DIR / f"feature_importance_{mode}.png"


def calibration_plot(mode):
    return MODELS_DIR / f"calibration_{mode}.png"

# --- Training -----------------------------------------------------------
SEED = 42
MLP_HIDDEN = [64, 32]
DROPOUT = 0.2
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 12                # early-stopping patience on val loss
TRAIN_FRAC = 0.70            # temporal split (by market end date)
VAL_FRAC = 0.15              # test = remaining 0.15
