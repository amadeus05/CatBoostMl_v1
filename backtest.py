import sqlite3
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from config import *

# === НАСТРОЙКИ КОМИССИЙ ===
COMMISSION = 0.001
SLIPPAGE = 0.0001

def backtest():
    print("Загружаем модель и данные...")
    model = CatBoostClassifier()
    model.load_model(str(MODELS_DIR / "catboost_model.cbm"))
    
    with open(MODELS_DIR / "features.pkl", "rb") as f:
        feature_names = pickle.load(f)
        
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM ETH_USDT_features", conn)
    conn.close()
    
    symbol = "ETH/USDT"
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        print("Ошибка: Нет колонки timestamp для группировки по месяцам!")
        return

    # Берем последние 15% для теста
    split = int(len(df) * 0.85)
    df = df.iloc[split:].reset_index(drop=True)
    
    balance = 100
    position = 0
    entry_price = 0.0
    trades = []
    
    # Для отслеживания просадки и эквити
    peak_balance = balance
    max_drawdown = 0.0
    equity_curve = []  # Баланс на каждом баре
    timestamps = []    # Время каждого бара
    
    monthly_stats = {}

    TP_PCT = 0.03
    SL_PCT = 0.015
    
    print(f"Старт симуляции на {len(df)} свечах...")
    print(f"Период: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")
    print(f"Торгуем: {symbol}")

    # Цикл торговли
    for i in range(len(df) - 1):
        # Сохраняем эквити на каждом баре
        equity_curve.append(balance)
        timestamps.append(df.loc[i, 'timestamp'])
        
        # Данные следующей свечи (это правильно - мы торгуем на открытии следующей свечи)
        next_open = df.loc[i+1, 'open']
        next_high = df.loc[i+1, 'high']
        next_low  = df.loc[i+1, 'low']
        
        current_date = df.loc[i+1, 'timestamp']
        month_key = current_date.strftime('%Y-%m')

        # --- ЛОГИКА ВЫХОДА ---
        if position != 0:
            exit_signal = False
            exit_price = 0.0
            pnl_clean = 0.0
            reason = ""
            
            if position == 1: # LONG
                stop_price = entry_price * (1 - SL_PCT)
                take_price = entry_price * (1 + TP_PCT)
                
                if next_low <= stop_price:
                    exit_price = stop_price * (1 - SLIPPAGE)
                    exit_signal = True
                    reason = "❌ SL"
                elif next_high >= take_price:
                    exit_price = take_price * (1 - SLIPPAGE)
                    exit_signal = True
                    reason = "✅ TP"
                    
            elif position == -1: # SHORT
                stop_price = entry_price * (1 + SL_PCT)
                take_price = entry_price * (1 - TP_PCT)
                
                if next_high >= stop_price:
                    exit_price = stop_price * (1 + SLIPPAGE)
                    exit_signal = True
                    reason = "❌ SL"
                elif next_low <= take_price:
                    exit_price = take_price * (1 + SLIPPAGE)
                    exit_signal = True
                    reason = "✅ TP"

            if exit_signal:
                raw_pnl = (exit_price - entry_price) / entry_price
                if position == -1: raw_pnl *= -1
                pnl_clean = raw_pnl - (COMMISSION * 2)
                
                balance *= (1 + pnl_clean)
                trades.append(pnl_clean)
                position = 0
                
                # Обновляем просадку
                if balance > peak_balance:
                    peak_balance = balance
                current_drawdown = (peak_balance - balance) / peak_balance * 100
                if current_drawdown > max_drawdown:
                    max_drawdown = current_drawdown
                
                if month_key not in monthly_stats:
                    monthly_stats[month_key] = {'pnl': 0.0, 'trades': 0, 'wins': 0}
                
                monthly_stats[month_key]['pnl'] += pnl_clean
                monthly_stats[month_key]['trades'] += 1
                if pnl_clean > 0:
                    monthly_stats[month_key]['wins'] += 1

                print(f"Bar {i+1} [{month_key}] {symbol}: {reason} | PnL: {pnl_clean*100:.2f}% | Bal: {balance:.2f}")
                continue

        # --- ЛОГИКА ВХОДА ---
        if position == 0:
            # Важное исправление: используем данные с предыдущего бара для принятия решения
            # На баре i мы можем использовать только данные до бара i
            if i > 0:  # Начинаем со второго бара
                current_features = df.iloc[[i-1]][feature_names]
                probs = model.predict_proba(current_features)[0]
                p_short, p_neutral, p_long = probs
                
                signal = 0
                if p_long > CONFIDENCE_THRESHOLD:
                    signal = 1
                elif p_short > CONFIDENCE_THRESHOLD:
                    signal = -1
                
                if signal != 0:
                    if signal == 1:
                        entry_price = next_open * (1 + SLIPPAGE)
                        direction = "LONG"
                        prob = p_long
                    else:
                        entry_price = next_open * (1 - SLIPPAGE)
                        direction = "SHORT"
                        prob = p_short
                    
                    position = signal
                    print(f"Bar {i+1} {symbol}: OPEN {direction} (Sig: {prob:.2f}) at {entry_price:.2f}")

    print("\n" + "="*40)
    print(f"РЕЗУЛЬТАТЫ ПО МЕСЯЦАМ ({symbol})")
    print("="*40)
    print(f"{'Месяц':<10} | {'Сделок':<8} | {'WinRate':<8} | {'Прибыль':<10}")
    print("-" * 45)

    sorted_months = sorted(monthly_stats.keys())
    
    total_percent_sum = 0
    
    for m in sorted_months:
        stats = monthly_stats[m]
        count = stats['trades']
        wins = stats['wins']
        pnl = stats['pnl'] * 100
        
        wr = (wins / count * 100) if count > 0 else 0
        total_percent_sum += pnl
        
        pnl_str = f"{pnl:+.2f}%"
        
        print(f"{m:<10} | {count:<8} | {wr:<7.1f}% | {pnl_str:<10}")

    print("-" * 45)
    if trades:
        win_rate = len([t for t in trades if t>0])/len(trades)*100
    else:
        win_rate = 0
    print(f"ИТОГО      | {len(trades):<8} | {win_rate:.1f}%     | {total_percent_sum:+.2f}% (Sum)")
    print(f"\nКонечный баланс: {balance:.2f}")
    print(f"Макс. просадка:  {max_drawdown:.2f}%")
    
    # === ДОПОЛНИТЕЛЬНЫЕ МЕТРИКИ ===
    if trades:
        gross_profit = sum([t for t in trades if t > 0])
        gross_loss = abs(sum([t for t in trades if t < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Sharpe Ratio (аннуализированный, предполагаем risk-free = 0)
        returns = np.array(trades)
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)  # Аннуализация
        else:
            sharpe = 0
        
        # Recovery Factor = Общая прибыль / Макс. просадка
        total_return = (balance - 100) / 100 * 100
        recovery_factor = total_return / max_drawdown if max_drawdown > 0 else float('inf')
        
        print(f"Profit Factor:   {profit_factor:.2f}")
        print(f"Sharpe Ratio:    {sharpe:.2f}")
        print(f"Recovery Factor: {recovery_factor:.2f}")
    
    # === ЭКВИТИ КУРВАЯ ===
    if len(equity_curve) > 1:
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, equity_curve, 'b-', linewidth=1, label='Equity')
        plt.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Start')
        plt.fill_between(timestamps, 100, equity_curve, 
                        where=[e >= 100 for e in equity_curve], 
                        color='green', alpha=0.3)
        plt.fill_between(timestamps, 100, equity_curve, 
                        where=[e < 100 for e in equity_curve], 
                        color='red', alpha=0.3)
        plt.title(f'Equity Curve: {symbol} | {len(trades)} trades | PF: {profit_factor:.2f} | Max DD: {max_drawdown:.1f}%')
        plt.xlabel('Date')
        plt.ylabel('Balance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('equity_curve.png', dpi=150)
        plt.show()
        print("\n📈 График сохранен: equity_curve.png")

if __name__ == '__main__':
    backtest()