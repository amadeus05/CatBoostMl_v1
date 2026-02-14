from pathlib import Path

# --- ОСНОВНЫЕ ---
DB_PATH = "market_data.db"
SYMBOLS = [
    # "BTC/USDT",
    "ETH/USDT",
    # "BNB/USDT",
    # "SOL/USDT",
    # "XRP/USDT",
    # "ADA/USDT",
    # "DOGE/USDT",
    # "LINK/USDT",
    # "AVAX/USDT",
    # "DOT/USDT",
    # "LTC/USDT",
    # "MATIC/USDT",
    # "ATOM/USDT",
    # "NEAR/USDT",
]

# SYMBOLS = ["BTC/USDT", 
# "ETH/USDT",
# #  "SOL/USDT", 
# # "BNB/USDT", "XRP/USDT","LINK/USDT","NEAR/USDT","SUI/USDT","DOGE/USDT"
#  ]
# SYMBOLS = [
#     "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", # Твои текущие
#     "XRP/USDT", "ADA/USDT", "DOT/USDT", "LINK/USDT", "LTC/USDT", # Топ-ликвидность
#     "AVAX/USDT", "NEAR/USDT", "MATIC/USDT", "SUI/USDT", # Волатильность
#     "OP/USDT", "ARB/USDT", "DOGE/USDT" # Разные сектора
# ]
TIMEFRAME = "1h"
HTF_TIMEFRAME = "4h"  # Старший таймфрейм для мульти-TF фичей

# --- DATA LOADING ---
START_DATE = "2022-01-01"  # Дата начала загрузки данных
END_DATE = None            # None = до текущего времени, или "2026-01-01"
BINANCE_LIMIT = 1500       # Лимит свечей за один запрос
BINANCE_SLEEP = 0.3        # Пауза между запросами (сек)

# --- ML LABELING (Triple Barrier) ---
HORIZON = 12
ATR_MULTIPLIER = 2.0

# --- TRADING ---
CONFIDENCE_THRESHOLD = 0.53

# --- PATHS ---
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)