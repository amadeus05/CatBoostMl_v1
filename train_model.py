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
    
    # Подготовка X и y
    drop_cols = ['timestamp', 'Target', 'open', 'high', 'low', 'close', 'volume']
    features = [c for c in df.columns if c not in drop_cols]
    
    X = df[features]
    y = df['Target']
    
    # Маппинг для CatBoost
    y_mapped = y.map({-1: 0, 0: 1, 1: 2})
    
    # Разделение по времени (без перемешивания!)
    split = int(len(df) * 0.85)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y_mapped.iloc[:split], y_mapped.iloc[split:]
    
    print(f"Обучение на {len(X_train)} примерах, тест на {len(X_test)}")
    print(f"Распределение классов Train: {y_train.value_counts(normalize=True).to_dict()}")
    
    model = CatBoostClassifier(
        iterations=1000,
        depth=7,
        learning_rate=0.03,
        loss_function='MultiClass',
        eval_metric='Accuracy',
        auto_class_weights='Balanced',
        early_stopping_rounds=200,
        verbose=100
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