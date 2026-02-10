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
TP_PCT = 0.03
SL_PCT = 0.015

def load_all_data(symbols, feature_names):
    conn = sqlite3.connect(DB_PATH)
    all_dfs = {}
    
    print(f"Загрузка данных для {len(symbols)} монет...")
    for sym in symbols:
        table_name = f"{sym.replace('/', '_')}_features"
        try:
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                # Оставляем только нужные колонки для экономии памяти
                cols_to_keep = ['timestamp', 'open', 'high', 'low', 'close'] + feature_names
                df = df[cols_to_keep]
                all_dfs[sym] = df
            else:
                print(f"⚠️ Пропуск {sym}: нет колонки timestamp")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки {sym}: {e}")
            
    conn.close()
    return all_dfs

def backtest():
    print("Загружаем модель и фичи...")
    model = CatBoostClassifier()
    model.load_model(str(MODELS_DIR / "catboost_model.cbm"))
    
    with open(MODELS_DIR / "features.pkl", "rb") as f:
        feature_names = pickle.load(f)
        
    all_dfs = load_all_data(SYMBOLS, feature_names)
    
    if not all_dfs:
        print("Ошибка: Нет данных для бектеста!")
        return

    # Находим общие таймстампы и обрезаем последние 15%
    # Для простоты возьмем пересечение всех доступных дат
    common_timestamps = sorted(list(set.intersection(*(set(df['timestamp']) for df in all_dfs.values()))))
    
    split_idx = int(len(common_timestamps) * 0.85)
    test_timestamps = common_timestamps[split_idx:]
    
    if not test_timestamps:
        print("Ошибка: Слишком мало данных для теста (15% от пересечения пусто)!")
        return

    # Фильтруем все DF по тестовым таймстампам
    for sym in all_dfs:
        df = all_dfs[sym]
        all_dfs[sym] = df[df['timestamp'].isin(test_timestamps)].sort_values('timestamp').reset_index(drop=True)

    # Инициализация состояния
    balance = 100.0
    positions = {sym: None for sym in all_dfs} # None или {'dir': 1/-1, 'entry': price}
    trades = []
    equity_curve = []
    equity_timestamps = []
    monthly_stats = {}
    peak_balance = balance
    max_drawdown = 0.0

    print(f"Старт симуляции на {len(test_timestamps)} свечах...")
    print(f"Период: {test_timestamps[0]} -> {test_timestamps[-1]}")
    print(f"Монеты: {', '.join(all_dfs.keys())}")

    # Основной цикл симуляции по времени
    # i - индекс текущей закрытой свечи, на основе которой принимаем решение
    # i+1 - следующая свеча, на открытии которой входим/выходим
    num_candles = len(test_timestamps)
    
    for i in range(num_candles - 1):
        current_ts = test_timestamps[i]
        next_ts = test_timestamps[i+1]
        
        # Сохраняем эквити
        equity_curve.append(balance)
        equity_timestamps.append(current_ts)
        
        month_key = next_ts.strftime('%Y-%m')
        if month_key not in monthly_stats:
            monthly_stats[month_key] = {'pnl': 0.0, 'trades': 0, 'wins': 0}

        # Обрабатываем каждую монету
        for sym, df in all_dfs.items():
            # Данные текущей (закрытой) свечи
            curr_row = df.iloc[i]
            # Данные следующей свечи (по которой исполняем)
            next_row = df.iloc[i+1]
            
            next_open = next_row['open']
            next_high = next_row['high']
            next_low = next_row['low']

            # --- ЛОГИКА ВЫХОДА ---
            if positions[sym] is not None:
                pos = positions[sym]
                entry_price = pos['entry']
                direction = pos['dir']
                
                exit_signal = False
                exit_price = 0.0
                reason = ""

                # Проверяем, были ли достигнуты лимиты на этой свече
                if direction == 1: # LONG
                    stop_price = entry_price * (1 - SL_PCT)
                    take_price = entry_price * (1 + TP_PCT)
                    
                    is_sl = next_low <= stop_price
                    is_tp = next_high >= take_price
                    
                    if is_sl and is_tp:
                        exit_price = stop_price * (1 - SLIPPAGE)
                        exit_signal = True
                        reason = "❌ SL (Both Hit)" # Худший сценарий - сначала SL
                    elif is_sl:
                        exit_price = stop_price * (1 - SLIPPAGE)
                        exit_signal = True
                        reason = "❌ SL"
                    elif is_tp:
                        exit_price = take_price * (1 - SLIPPAGE)
                        exit_signal = True
                        reason = "✅ TP"
                
                else: # SHORT
                    stop_price = entry_price * (1 + SL_PCT)
                    take_price = entry_price * (1 - TP_PCT)
                    
                    is_sl = next_high >= stop_price
                    is_tp = next_low <= take_price
                    
                    if is_sl and is_tp:
                        exit_price = stop_price * (1 + SLIPPAGE)
                        exit_signal = True
                        reason = "❌ SL (Both Hit)"
                    elif is_sl:
                        exit_price = stop_price * (1 + SLIPPAGE)
                        exit_signal = True
                        reason = "❌ SL"
                    elif is_tp:
                        exit_price = take_price * (1 + SLIPPAGE)
                        exit_signal = True
                        reason = "✅ TP"

                if exit_signal:
                    raw_pnl = (exit_price - entry_price) / entry_price
                    if direction == -1: raw_pnl *= -1
                    pnl_clean = raw_pnl - (COMMISSION * 2)
                    
                    # Фиксируем результат
                    trade_profit = balance * pnl_clean
                    balance += trade_profit
                    trades.append({'sym': sym, 'pnl': pnl_clean, 'ts': next_ts})
                    
                    # Обновляем статистику
                    monthly_stats[month_key]['pnl'] += pnl_clean
                    monthly_stats[month_key]['trades'] += 1
                    if pnl_clean > 0:
                        monthly_stats[month_key]['wins'] += 1
                        
                    # Обновляем просадку
                    if balance > peak_balance: peak_balance = balance
                    current_dd = (peak_balance - balance) / peak_balance * 100
                    if current_dd > max_drawdown: max_drawdown = current_dd

                    positions[sym] = None
                    print(f"[{next_ts}] {sym}: {reason} | PnL: {pnl_clean*100:.2f}% | Bal: {balance:.2f}")
                    continue

            # --- ЛОГИКА ВХОДА ---
            if positions[sym] is None:
                # На баре i мы уже знаем результат закрытия свечи i
                # Используем свечу i для принятия решения о входе на открытии i+1
                current_features = df.iloc[[i]][feature_names]
                probs = model.predict_proba(current_features)[0]
                p_short, p_neutral, p_long = probs
                
                signal = 0
                if p_long > CONFIDENCE_THRESHOLD: signal = 1
                elif p_short > CONFIDENCE_THRESHOLD: signal = -1
                
                if signal != 0:
                    if signal == 1:
                        entry_price = next_open * (1 + SLIPPAGE)
                        direction_str = "LONG"
                        prob = p_long
                    else:
                        entry_price = next_open * (1 - SLIPPAGE)
                        direction_str = "SHORT"
                        prob = p_short
                    
                    positions[sym] = {'dir': signal, 'entry': entry_price}
                    print(f"[{next_ts}] {sym}: OPEN {direction_str} (Sig: {prob:.2f}) at {entry_price:.2f}")

    # === РЕЗУЛЬТАТЫ ===
    print("\n" + "="*50)
    print(f"ИТОГОВЫЕ РЕЗУЛЬТАТЫ ПО ВСЕМ МОНЕТАМ")
    print("="*50)
    
    print(f"{'Месяц':<10} | {'Сделок':<8} | {'WinRate':<8} | {'Прибыль':<10}")
    print("-" * 50)

    total_pnl = 0
    total_trades = 0
    total_wins = 0

    for m in sorted(monthly_stats.keys()):
        stats = monthly_stats[m]
        count = stats['trades']
        wins = stats['wins']
        pnl = stats['pnl'] * 100
        wr = (wins / count * 100) if count > 0 else 0
        
        total_pnl += stats['pnl']
        total_trades += count
        total_wins += wins
        
        print(f"{m:<10} | {count:<8} | {wr:<7.1f}% | {pnl:+.2f}%")

    print("-" * 50)
    final_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    print(f"ИТОГО      | {total_trades:<8} | {final_wr:.1f}%     | {total_pnl*100:+.2f}%")
    print(f"\nКонечный баланс: {balance:.2f}")
    print(f"Макс. просадка:  {max_drawdown:.2f}%")

    # === ПРАВИЛЬНЫЙ РАСЧЕТ МЕТРИК (Time-Series) ===
    if len(equity_curve) > 0:
        # 1. Создаем временной ряд эквити
        equity_series = pd.Series(equity_curve, index=equity_timestamps)
        
        # 2. Ресемплим к ДНЕВНЫМ данным (берем последнее значение за каждый день)
        # Для крипты важно заполнить дни без сделок предыдущим значением (.ffill)
        daily_equity = equity_series.resample('D').last().ffill()
        
        # 3. Считаем дневную доходность (в процентах)
        daily_returns = daily_equity.pct_change().dropna()
        
        # 4. Считаем метрики
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            # Годовая доходность (CAGR - упрощенно)
            total_days = (daily_equity.index[-1] - daily_equity.index[0]).days
            if total_days > 0:
                cagr = (daily_equity.iloc[-1] / daily_equity.iloc[0]) ** (365 / total_days) - 1
            else:
                cagr = 0

            # Sharpe Ratio
            risk_free_rate = 0.0 
            mean_daily_return = daily_returns.mean()
            std_daily_return = daily_returns.std()
            
            # Формула Шарпа
            sharpe = ((mean_daily_return - (risk_free_rate/365)) / std_daily_return) * np.sqrt(365)
            
            # Sortino Ratio
            downside_returns = daily_returns[daily_returns < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.std()
                sortino = ((mean_daily_return - (risk_free_rate/365)) / downside_std) * np.sqrt(365)
            else:
                sortino = 0 
                
            # Calmar Ratio
            calmar = cagr / (max_drawdown / 100) if max_drawdown > 0 else 0
        else:
            sharpe = 0
            sortino = 0
            calmar = 0
            cagr = 0

        # Расчет Profit Factor (оставляем для полноты картины)
        if trades:
            returns = np.array([t['pnl'] for t in trades])
            gross_profit = sum([r for r in returns if r > 0])
            gross_loss = abs(sum([r for r in returns if r < 0]))
            pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            pf = 0

        print("\n" + "="*40)
        print("📊 ПРОФЕССИОНАЛЬНЫЕ МЕТРИКИ")
        print("="*40)
        print(f"Profit Factor:   {pf:.2f}")
        print(f"Sharpe Ratio:    {sharpe:.2f} (Норма: >1.0, Отлично: >2.0)")
        print(f"Sortino Ratio:   {sortino:.2f} (Лучше Шарпа, т.к. не наказывает за рост)")
        print(f"Calmar Ratio:    {calmar:.2f} (Доходность / Риск)")
        print(f"CAGR (Годовые):  {cagr*100:.2f}%")
        print("-" * 40)

    if len(equity_curve) > 1:
        plt.figure(figsize=(12, 6))
        plt.plot(equity_timestamps, equity_curve, 'b-', label='Portfolio Equity')
        plt.axhline(y=100, color='gray', linestyle='--')
        plt.title(f'Multi-Symbol Equity Curve | {total_trades} trades | DD: {max_drawdown:.1f}%')
        plt.grid(True, alpha=0.3)
        plt.savefig('equity_curve.png', dpi=150)
        plt.show()
        print("\n📈 График сохранен: equity_curve.png")

if __name__ == '__main__':
    backtest()
