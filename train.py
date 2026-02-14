import sqlite3
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from config import *

def load_data_from_db():
    conn = sqlite3.connect(DB_PATH)
    all_data = []
    
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_features';")
    tables = cursor.fetchall()
    
    for table_name in tables:
        df = pd.read_sql(f"SELECT * FROM {table_name[0]}", conn)
        all_data.append(df)
    
    conn.close()
    return pd.concat(all_data, ignore_index=True)

def train():
    df = load_data_from_db()
    
    # Конвертация timestamp и сортировка для корректного temporal split
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Подготовка X и y
    drop_cols = ['timestamp', 'Target', 'open', 'high', 'low', 'close', 'volume',
                 'Resistance', 'Support', 'HTF_EMA_50', 'volume_ma_20']
    features = [c for c in df.columns if c not in drop_cols]
    
    X = df[features]
    y = df['Target']
    
    # Маппинг для CatBoost
    y_mapped = y.map({-1: 0, 0: 1, 1: 2})
    
    # Разделение по времени (без перемешивания!)
    split_time = df['timestamp'].quantile(0.85)
    
    # Purge gap: HORIZON баров между train и test,
    # чтобы Triple Barrier labels не использовали test данные
    TF_TO_HOURS = {"1m": 1/60, "5m": 5/60, "15m": 0.25, "1h": 1, "4h": 4, "1d": 24}
    bar_hours = TF_TO_HOURS.get(TIMEFRAME, 1)
    purge_gap = pd.Timedelta(hours=bar_hours * HORIZON)
    
    train_mask = df['timestamp'] <= (split_time - purge_gap)
    test_mask = df['timestamp'] > split_time
    
    purged_count = len(df) - train_mask.sum() - test_mask.sum()
    print(f"Purge gap: {purge_gap}, удалено {purged_count} строк между train и test")
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y_mapped[train_mask], y_mapped[test_mask]
    
    print(f"Обучение на {len(X_train)} примерах, тест на {len(X_test)}")
    print(f"Распределение классов Train: {y_train.value_counts(normalize=True).to_dict()}")
    
    model = CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.05,
        loss_function='MultiClass',
        eval_metric='TotalF1', ##TotalF1,Accuracy
        auto_class_weights='Balanced',
        early_stopping_rounds=200,
        verbose=100,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        use_best_model=True
    )
    
    preds = model.predict(X_test)
    print("\nОТЧЕТ (0=Short, 1=Neutral, 2=Long):")
    print(classification_report(y_test, preds))
    
    model.save_model(str(MODELS_DIR / "catboost_model.cbm"))
    
    with open(MODELS_DIR / "features.pkl", "wb") as f:
        pickle.dump(features, f)
        
    print("Модель сохранена!")

if __name__ == '__main__':
    train()