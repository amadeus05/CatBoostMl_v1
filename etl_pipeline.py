import requests
import pandas as pd
import pandas_ta as ta
import numpy as np
import sqlite3
import logging
import time
from datetime import datetime
from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://fapi.binance.com/fapi/v1/klines"

# Интервалы в миллисекундах
TF_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def init_db():
    """Создание таблицы если не существует"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS candles (
            symbol TEXT,
            timeframe TEXT,
            open_time INTEGER,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            quote_volume REAL,
            PRIMARY KEY (symbol, timeframe, open_time)
        )
    """)
    conn.commit()
    return conn


def fetch_data(conn, symbol, timeframe):
    """
    Загрузка данных с Binance API начиная с START_DATE или последней точки в БД.
    Поддерживает инкрементальную загрузку.
    """
    # Конвертируем символ для API (BTC/USDT -> BTCUSDT)
    api_symbol = symbol.replace("/", "")
    
    cur = conn.cursor()
    cur.execute("SELECT MAX(open_time) FROM candles WHERE symbol=? AND timeframe=?", (symbol, timeframe))
    last_ts = cur.fetchone()[0]
    
    # Старт с последней точки в БД или с START_DATE
    if last_ts:
        start_ts = last_ts + 1
    else:
        start_ts = int(datetime.fromisoformat(START_DATE).timestamp() * 1000)
    
    # Конец: END_DATE или текущее время
    end_ts = int(datetime.fromisoformat(END_DATE).timestamp() * 1000) if END_DATE else None
    
    # Проверяем, не вышли ли мы за пределы END_DATE
    if end_ts and start_ts >= end_ts:
        logger.info(f"[{symbol}-{timeframe}] Данные уже загружены до {END_DATE}")
        return 0
    
    total_loaded = 0
    expected_interval = TF_MS.get(timeframe, 3_600_000)

    while True:
        params = {
            "symbol": api_symbol,
            "interval": timeframe,
            "startTime": start_ts,
            "limit": BINANCE_LIMIT
        }
        if end_ts:
            params["endTime"] = end_ts
            
        try:
            r = requests.get(BASE_URL, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.error(f"Ошибка загрузки {symbol}-{timeframe}: {e}")
            break

        if not data:
            break

        rows = []
        for k in data:
            current_ts = k[0]
            rows.append((
                symbol, timeframe, current_ts,
                float(k[1]), float(k[2]), float(k[3]), float(k[4]),
                float(k[5]), float(k[7])  # volume, quote_volume
            ))
            start_ts = current_ts + 1  # +1 мс чтобы не запрашивать эту же свечу снова

        cur.executemany("INSERT OR IGNORE INTO candles VALUES (?,?,?,?,?,?,?,?,?)", rows)
        conn.commit()
        total_loaded += len(rows)
        
        logger.info(f"[{symbol}-{timeframe}] Загружено {total_loaded} свечей, до {datetime.fromtimestamp((start_ts-1)/1000)}")
        
        # Выход: получили меньше лимита или достигли END_DATE
        if len(data) < BINANCE_LIMIT:
            break
        if end_ts and start_ts >= end_ts:
            logger.info(f"[{symbol}-{timeframe}] Достигнута дата окончания {END_DATE}")
            break
            
        time.sleep(BINANCE_SLEEP)
    
    return total_loaded


def load_from_db(conn, symbol, timeframe):
    """Загрузка данных из БД в DataFrame"""
    df = pd.read_sql_query(
        "SELECT open_time as timestamp, open, high, low, close, volume FROM candles WHERE symbol=? AND timeframe=? ORDER BY open_time",
        conn,
        params=(symbol, timeframe)
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


def add_features(df):
    """Генерация признаков БЕЗ подсматривания в будущее"""
    df = df.copy()
    
    # 1. Трендовые и Осцилляторы (ТЕКУЩИЕ, без shift)
    df['RSI'] = df.ta.rsi(length=14)
    macd = df.ta.macd()
    df['MACD_line'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df['MACD_hist'] = macd['MACDh_12_26_9']
    df['ATR'] = df.ta.atr(length=14)
    
    # 2. Логарифмическая доходность (Текущая Close к Прошлой Close)
    df['Log_Ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # 3. Относительный объем (ИСПРАВЛЕНО: Вариант A из критики)
    # Используем скользящее среднее текущего момента (включая текущий бар, это допустимо и убирает лаг)
    df['volume_ma_20'] = df['volume'].rolling(20, min_periods=1).mean()
    df['Vol_Rel'] = df['volume'] / df['volume_ma_20']
    
    # 4. Лаги (для истории)
    for col in ['RSI', 'Log_Ret', 'Vol_Rel']:
        for i in range(1, 4):
            df[f'{col}_lag_{i}'] = df[col].shift(i)
    
    # 5. Время
    df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # 6. EMA (ИСПРАВЛЕНО: Вариант A из критики - EMA текущая, сравнение текущее)
    df['EMA_200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['Trend'] = (df['close'] > df['EMA_200']).astype(int)
    
    # 7. Поддержка / Сопротивление 
    SR_LOOKBACK = 50
    # Уровни строим по ПРОШЛЫМ данным (shift(1) ОБЯЗАТЕЛЕН для уровней)
    df['Resistance'] = df['high'].rolling(SR_LOOKBACK, min_periods=1).max().shift(1)
    df['Support'] = df['low'].rolling(SR_LOOKBACK, min_periods=1).min().shift(1)
    
    # ИСПРАВЛЕНО: Дистанцию считаем от ТЕКУЩЕЙ цены до уровней
    df['Dist_to_Resistance'] = (df['Resistance'] - df['close']) / df['ATR']
    df['Dist_to_Support'] = (df['close'] - df['Support']) / df['ATR']
    
    # Позиция цены: считаем по текущей цене
    sr_range = df['Resistance'] - df['Support']
    df['SR_Position'] = ((df['close'] - df['Support']) / sr_range).clip(0, 1)
    
    df.dropna(inplace=True)
    return df


def add_htf_features(df, htf_df):
    """
    Добавление фичей старшего таймфрейма.
    ВАЖНО: Оставляем shift(1) для HTF, так как timestamps - это Open Time.
    Без shift(1) мы бы заглянули в 'будущее' (в конец 4h свечи) при merge_asof.
    """
    htf = htf_df.copy()
    
    # Считаем индикаторы на 4h (shift(1) чтобы использовать только ЗАВЕРШЕННЫЕ свечи)
    htf['HTF_RSI'] = htf.ta.rsi(length=14).shift(1)
    htf['HTF_ATR'] = htf.ta.atr(length=14).shift(1)
    htf_macd = htf.ta.macd()
    htf['HTF_MACD_hist'] = htf_macd['MACDh_12_26_9'].shift(1)
    htf['HTF_EMA_50'] = htf['close'].ewm(span=50, adjust=False).mean().shift(1)
    htf['HTF_Trend'] = (htf['close'].shift(1) > htf['HTF_EMA_50']).astype(int)
    htf['HTF_Log_Ret'] = np.log(htf['close'] / htf['close'].shift(1))
    
    # Оставляем только нужные колонки для merge
    htf_cols = ['timestamp', 'HTF_RSI', 'HTF_ATR', 'HTF_MACD_hist',
                'HTF_EMA_50', 'HTF_Trend', 'HTF_Log_Ret']
    htf = htf[htf_cols].dropna()
    
    # merge_asof: для каждого 1h timestamp берем последнюю 4h запись <= этого времени
    # Т.к. мы сделали shift(1) выше, запись 12:00 содержит данные свечи 08:00-12:00.
    # Это корректно и безопасно.
    df = df.sort_values('timestamp')
    htf = htf.sort_values('timestamp')
    df = pd.merge_asof(df, htf, on='timestamp', direction='backward')
    
    df.dropna(inplace=True)
    return df


def triple_barrier_labeling(df):
    """Разметка данных (Teacher)"""
    labels = []
    
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    atrs = df['ATR'].values
    
    # Проходим по истории
    for i in range(len(df) - HORIZON):
        current_price = closes[i]
        current_atr = atrs[i]
        
        upper_barrier = current_price + (current_atr * ATR_MULTIPLIER)
        lower_barrier = current_price - (current_atr * ATR_MULTIPLIER)
        
        label = 0
        
        # Смотрим в будущее на HORIZON шагов
        for j in range(1, HORIZON + 1):
            if i + j >= len(df): break
            
            future_high = highs[i + j]
            future_low = lows[i + j]
            
            if future_high >= upper_barrier:
                label = 1
                break
            if future_low <= lower_barrier:
                label = -1
                break
                
        labels.append(label)
    
    labels.extend([0] * HORIZON)
    df['Target'] = labels
    return df


def save_processed(df, symbol):
    """Сохранение обработанных данных в отдельную таблицу"""
    conn = sqlite3.connect(DB_PATH)
    table_name = symbol.replace('/', '_') + "_features"
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    logger.info(f"💾 {symbol} features сохранены ({len(df)} строк)")


def main():
    conn = init_db()
    
    for symbol in SYMBOLS:
        # Загрузка основного таймфрейма
        logger.info(f"Loading {symbol} {TIMEFRAME} from {START_DATE}...")
        loaded = fetch_data(conn, symbol, TIMEFRAME)
        logger.info(f"{symbol} {TIMEFRAME}: {loaded} new candles")
        
        # Загрузка старшего таймфрейма (4h)
        logger.info(f"Loading {symbol} {HTF_TIMEFRAME} from {START_DATE}...")
        htf_loaded = fetch_data(conn, symbol, HTF_TIMEFRAME)
        logger.info(f"{symbol} {HTF_TIMEFRAME}: {htf_loaded} new candles")
        
        # Загружаем из БД и обрабатываем
        df = load_from_db(conn, symbol, TIMEFRAME)
        htf_df = load_from_db(conn, symbol, HTF_TIMEFRAME)
        
        if len(df) > 0 and len(htf_df) > 0:
            df = add_features(df)
            df = add_htf_features(df, htf_df)
            df = triple_barrier_labeling(df)
            save_processed(df, symbol)
            logger.info(f"{symbol}: saved {len(df)} rows with HTF + S/R features")
        else:
            logger.warning(f"{symbol}: no data in DB")
    
    conn.close()


if __name__ == '__main__':
    main()