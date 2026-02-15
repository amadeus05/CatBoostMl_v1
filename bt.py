import sqlite3
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from config import *

# === НАСТРОЙКИ ФЬЮЧЕРСОВ ===
TAKER_COM = 0.0004  # комиссия Taker Binance Futures
MAKER_COM = 0.0002  # комиссия Maker Binance Futures
SLIPPAGE = 0.0003
TP_PCT = 0.030
SL_PCT = 0.015
LEVERAGE = 1
CONFIDENCE_THRESHOLD = 0.65
RISK_PER_TRADE = 0.01  # 2% от депозита на сделку


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

    common_timestamps = sorted(list(set.intersection(*(set(df['timestamp']) for df in all_dfs.values()))))
    split_idx = int(len(common_timestamps) * 0.85)
    test_timestamps = common_timestamps[split_idx:]
    if not test_timestamps:
        print("Ошибка: Слишком мало данных для теста!")
        return

    for sym in all_dfs:
        df = all_dfs[sym]
        all_dfs[sym] = df[df['timestamp'].isin(test_timestamps)].sort_values('timestamp').reset_index(drop=True)

    balance = 500.0
    positions = {sym: None for sym in all_dfs}
    trades = []
    equity_curve = []
    equity_timestamps = []
    monthly_stats = {}
    peak_balance = balance
    max_drawdown = 0.0
    used_margin = 0.0  # Заблокированная маржа

    print(f"Старт симуляции на {len(test_timestamps)} свечах...")
    print(f"Период: {test_timestamps[0]} -> {test_timestamps[-1]}")
    print(f"Монеты: {', '.join(all_dfs.keys())}")

    num_candles = len(test_timestamps)
    for i in range(num_candles - 1):
        current_ts = test_timestamps[i]
        next_ts = test_timestamps[i+1]
        equity_curve.append(balance)
        equity_timestamps.append(current_ts)

        month_key = next_ts.strftime('%Y-%m')
        if month_key not in monthly_stats:
            monthly_stats[month_key] = {'pnl_abs': 0.0, 'trades': 0, 'wins': 0, 'start_balance': balance}

        for sym, df in all_dfs.items():
            curr_row = df.iloc[i]
            next_row = df.iloc[i+1]
            next_open = next_row['open']
            next_high = next_row['high']
            next_low = next_row['low']

            # --- ЛОГИКА ВЫХОДА ---
            if positions[sym] is not None:
                pos = positions[sym]
                entry_price = pos['entry']
                direction = pos['dir']
                position_notional = pos['size']
                exit_signal = False
                exit_price = 0.0
                reason = ""

                if direction == 1:  # LONG
                    stop_price = entry_price * (1 - SL_PCT)
                    take_price = entry_price * (1 + TP_PCT)
                    
                    if next_low <= stop_price:
                        exit_price = (next_open if next_open < stop_price else stop_price) * (1 - SLIPPAGE)
                        exit_signal = True
                        reason = "❌ SL"
                    elif next_high >= take_price:
                        exit_price = take_price * (1 - SLIPPAGE)
                        exit_signal = True
                        reason = "✅ TP"
                else:  # SHORT
                    stop_price = entry_price * (1 + SL_PCT)
                    take_price = entry_price * (1 - TP_PCT)
                    
                    if next_high >= stop_price:
                        exit_price = (next_open if next_open > stop_price else stop_price) * (1 + SLIPPAGE)
                        exit_signal = True
                        reason = "❌ SL"
                    elif next_low <= take_price:
                        exit_price = take_price * (1 + SLIPPAGE)
                        exit_signal = True
                        reason = "✅ TP"

                if exit_signal:
                    if direction == 1:
                        raw_pnl = (exit_price - entry_price) / entry_price
                    else:
                        raw_pnl = (entry_price - exit_price) / entry_price

                    commission = position_notional * (TAKER_COM + TAKER_COM)
                    pnl_clean = raw_pnl - (TAKER_COM + TAKER_COM)
                    trade_profit = position_notional * pnl_clean

                    # === ПРАВКА: освобождение маржи ===
                    used_margin -= pos['margin']
                    if used_margin < 0:
                        used_margin = 0.0

                    balance += trade_profit

                    trades.append({'sym': sym, 'pnl_pct': pnl_clean, 'pnl_abs': trade_profit, 'ts': next_ts})
                    monthly_stats[month_key]['pnl_abs'] += trade_profit
                    monthly_stats[month_key]['trades'] += 1
                    if pnl_clean > 0:
                        monthly_stats[month_key]['wins'] += 1

                    if balance > peak_balance: 
                        peak_balance = balance
                    current_dd = (peak_balance - balance) / peak_balance * 100
                    if current_dd > max_drawdown: 
                        max_drawdown = current_dd

                    positions[sym] = None
                    print(f"[{next_ts}] {sym}: {reason} | PnL: {pnl_clean*100:.2f}% | Com: {commission:.2f}$ | Bal: {balance:.2f}")
                    continue

            # --- ЛОГИКА ВХОДА ---
            if positions[sym] is None:
                current_features = df.iloc[[i]][feature_names]
                probs = model.predict_proba(current_features)[0]
                p_short, p_neutral, p_long = 0, 0, 0
                if len(probs) == 2:
                    p_short, p_long = probs
                else:
                    p_short, p_neutral, p_long = probs
                
                signal = 0
                if p_long > CONFIDENCE_THRESHOLD: 
                    signal = 1
                elif p_short > CONFIDENCE_THRESHOLD: 
                    signal = -1

                if signal != 0:
                    risk_capital = balance * RISK_PER_TRADE
                    position_notional = risk_capital / SL_PCT
                    
                    max_notional = balance * LEVERAGE
                    position_notional = min(position_notional, max_notional)
                    
                    required_margin = position_notional / LEVERAGE
                    available_balance = balance - used_margin
                    
                    if required_margin > available_balance:
                        required_margin = available_balance
                        position_notional = required_margin * LEVERAGE
                    
                    if position_notional < 10:
                        continue
                        
                    used_margin += required_margin
                    
                    if signal == 1:
                        entry_price = next_open * (1 + SLIPPAGE)
                        direction_str = "LONG"
                        prob = p_long
                    else:
                        entry_price = next_open * (1 - SLIPPAGE)
                        direction_str = "SHORT"
                        prob = p_short

                    positions[sym] = {
                        'dir': signal,
                        'entry': entry_price,
                        'size': position_notional,
                        'margin': required_margin
                    }
                    print(f"[{next_ts}] {sym}: 🚀 OPEN {direction_str} (Sig: {prob:.2f}) at {entry_price:.2f} | Size: {position_notional:.1f}$ Margin: {required_margin:.1f}$")

    # === РЕЗУЛЬТАТЫ ===
    print("\n" + "="*50)
    print(f"ИТОГОВЫЕ РЕЗУЛЬТАТЫ ПО ВСЕМ МОНЕТАМ")
    print("="*50)
    print(f"{'Месяц':<10} | {'Сделок':<8} | {'WinRate':<8} | {'Прибыль':<10}")
    print("-" * 50)

    total_pnl_abs = 0
    total_trades = 0
    total_wins = 0

    initial_balance = 500.0

    for m in sorted(monthly_stats.keys()):
        stats = monthly_stats[m]
        count = stats['trades']
        wins = stats['wins']
        pnl_abs = stats['pnl_abs']
        start_bal = stats['start_balance']
        
        pnl_pct = (pnl_abs / start_bal * 100) if start_bal > 0 else 0
        wr = (wins / count * 100) if count > 0 else 0
        total_pnl_abs += pnl_abs
        total_trades += count
        total_wins += wins
        print(f"{m:<10} | {count:<8} | {wr:<7.1f}% | {pnl_pct:+.2f}% ({pnl_abs:+.2f}$)")

    print("-" * 50)
    final_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    total_return_pct = ((balance - initial_balance) / initial_balance * 100) if initial_balance > 0 else 0
    print(f"ИТОГО      | {total_trades:<8} | {final_wr:.1f}%     | {total_return_pct:+.2f}% ({total_pnl_abs:+.2f}$)")
    print(f"\nКонечный баланс: {balance:.2f}")
    print(f"Макс. просадка:  {max_drawdown:.2f}%")

    # === МЕТРИКИ ===
    if len(equity_curve) > 0:
        equity_series = pd.Series(equity_curve, index=equity_timestamps)
        daily_equity = equity_series.resample('D').last().ffill()
        daily_returns = daily_equity.pct_change().dropna()
        
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            total_days = (daily_equity.index[-1] - daily_equity.index[0]).days
            cagr = (daily_equity.iloc[-1] / daily_equity.iloc[0]) ** (365 / total_days) - 1 if total_days > 0 else 0

            risk_free_rate = 0.0
            mean_daily_return = daily_returns.mean()
            std_daily_return = daily_returns.std()
            sharpe = ((mean_daily_return - (risk_free_rate/365)) / std_daily_return) * np.sqrt(365)

            downside_returns = daily_returns[daily_returns < 0]
            if len(downside_returns) > 1 and downside_returns.std() > 0:
                sortino = ((mean_daily_return - (risk_free_rate/365)) / downside_returns.std() * np.sqrt(365))
            else:
                sortino = 0

            calmar = cagr / (max_drawdown / 100) if max_drawdown > 0 else 0
        else:
            sharpe = sortino = calmar = cagr = 0

        if trades:
            returns = np.array([t['pnl_abs'] for t in trades])
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
        plt.axhline(y=500, color='gray', linestyle='--')
        plt.title(f'Multi-Symbol Equity Curve | {total_trades} trades | DD: {max_drawdown:.1f}%')
        plt.grid(True, alpha=0.3)
        plt.savefig('equity_curve.png', dpi=150)
        plt.show()
        print("\n📈 График сохранен: equity_curve.png")


if __name__ == '__main__':
    backtest()
