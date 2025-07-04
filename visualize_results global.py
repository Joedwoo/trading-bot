"""
Script for visualizing and analyzing Stock Trading Bot's performance.
This version runs a batch evaluation for all available symbols.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt

from trading_bot.agent import Agent
from trading_bot.environment import TradingEnvironment
from trading_bot.methods import evaluate_model
from trading_bot.utils import get_stock_data, format_currency, format_position

def get_symbols(datasets_path='datasets'):
    """
    Gets a list of stock symbols from the datasets directory.
    Assumes each subdirectory in `datasets_path` is a symbol.
    """
    try:
        symbols = [d for d in os.listdir(datasets_path) if os.path.isdir(os.path.join(datasets_path, d))]
        logging.info(f"Discovered symbols: {symbols}")
        return symbols
    except FileNotFoundError:
        logging.error(f"Directory not found: {datasets_path}")
        return []

def analyze_trades(history):
    """
    Analyzes the trade history to calculate performance metrics.

    Args:
        history (list): A list of tuples `(price, action)` from the evaluation.

    Returns:
        dict: A dictionary containing various performance metrics.
    """
    # --- Action Counts ---
    buy_count = sum(1 for _, action in history if action == "BUY")
    sell_count = sum(1 for _, action in history if action == "SELL")
    hold_count = sum(1 for _, action in history if action == "HOLD")

    # --- Trade Profit Calculations (FIFO Logic) ---
    dollar_trades = []
    percent_trades = []
    open_positions = []  # A queue to store buy prices

    for price, action in history:
        if action == "BUY":
            open_positions.append(price)
        elif action == "SELL":
            if open_positions:  # If there's an open position to sell
                buy_price = open_positions.pop(0)  # Get the first bought price (FIFO)
                profit = price - buy_price
                dollar_trades.append(profit)
                if buy_price != 0:
                    percent_trades.append((profit / buy_price) * 100)

    if not dollar_trades:
        return {
            "win_rate": 0,
            "total_trades": 0,
            "average_profit_usd": 0,
            "average_profit_pct": 0,
            "sharpe_ratio": 0,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "hold_count": hold_count,
        }

    dollar_trades = np.array(dollar_trades)
    
    # Win Rate
    wins = dollar_trades > 0
    win_rate = (np.sum(wins) / len(dollar_trades)) * 100
    
    # Average Profit
    average_profit_usd = np.mean(dollar_trades)
    average_profit_pct = np.mean(percent_trades) if percent_trades else 0
    
    # Sharpe Ratio (simplified)
    std_dev_profit = np.std(dollar_trades)
    sharpe_ratio = average_profit_usd / std_dev_profit if std_dev_profit != 0 else 0

    return {
        "win_rate": win_rate,
        "total_trades": len(dollar_trades),
        "average_profit_usd": average_profit_usd,
        "average_profit_pct": average_profit_pct,
        "sharpe_ratio": sharpe_ratio,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "hold_count": hold_count,
    }

def plot_single_performance(data, history, metrics, cumulative_profits, model_name, window_size, fg_index, symbol):
    """
    Generates and saves a modern plot of the agent's performance for a single symbol,
    including a cumulative profit chart.
    """
    prices = data['prices']
    dates = pd.to_datetime(data['dates'])
    
    # Align evaluation data (history, profits) with the full date range
    eval_start_index = window_size - 1
    eval_dates = dates[eval_start_index:]
    eval_prices = prices[eval_start_index:]

    buy_points = [(eval_dates[i], eval_prices[i]) for i, (_, action) in enumerate(history) if action == "BUY"]
    sell_points = [(eval_dates[i], eval_prices[i]) for i, (_, action) in enumerate(history) if action == "SELL"]

    plt.style.use('fivethirtyeight')
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=(16, 15),
        dpi=300,
        sharex=True, # Share the x-axis for aligned dates
        gridspec_kw={'height_ratios': [2.5, 1, 1]} # Give more space to the price chart
    )
    fig.suptitle(f'Analyse de Performance : {symbol} ({model_name})', fontsize=20, weight='bold')

    # --- Chart 1: Price History and Trades ---
    ax1.plot(dates, prices, label='Prix de Clôture', color='dimgray', linewidth=1.5, alpha=0.8)
    if buy_points:
        buy_dates, buy_prices = zip(*buy_points)
        ax1.scatter(buy_dates, buy_prices, label='Achat', marker='^', color='#2ca02c', s=120, edgecolors='black', zorder=5)
    if sell_points:
        sell_dates, sell_prices = zip(*sell_points)
        ax1.scatter(sell_dates, sell_prices, label='Vente', marker='v', color='#d62728', s=120, edgecolors='black', zorder=5)
    
    ax1.set_ylabel('Prix ($)', fontsize=14, weight='bold')
    ax1.legend(loc='lower left', fontsize=12)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- Chart 2: FGIndex ---
    ax2.plot(dates, fg_index, label='FGIndex', color='purple', linewidth=1.5)
    ax2.set_ylabel('Fear & Greed Index', fontsize=14, weight='bold')
    ax2.legend(loc='lower left', fontsize=12)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- Chart 3: Cumulative Profit ---
    # We plot profits against the dates from the *next* day onwards, as profit is realized after the day's action.
    # We must also ensure the profit array and date array have the same length for plotting.
    profit_dates = eval_dates[:len(cumulative_profits)]
    
    ax3.plot(profit_dates, cumulative_profits, label='Profit Cumulé', color='royalblue', linewidth=2)
    ax3.fill_between(profit_dates, cumulative_profits, 0,
                     where=(np.array(cumulative_profits) >= 0),
                     facecolor='green', alpha=0.3, interpolate=True)
    ax3.fill_between(profit_dates, cumulative_profits, 0,
                     where=(np.array(cumulative_profits) < 0),
                     facecolor='red', alpha=0.3, interpolate=True)
    
    ax3.set_ylabel('Profit Cumulé ($)', fontsize=14, weight='bold')
    ax3.legend(loc='lower left', fontsize=12)
    ax3.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- Text and General Formatting ---
    fig.autofmt_xdate()
    stats_text = (
        f"**Performance Stratégie**\n"
        f"  Rendement : {metrics['agent_profit_display']} ({metrics['agent_return_pct']:.2f}%)\n"
        f"  Trades Complets : {metrics['total_trades']} | Profit Moyen/Trade : {metrics['average_profit_pct']:.2f}%\n"
        f"  Winrate : {metrics['win_rate']:.2f}% | Sharpe : {metrics['sharpe_ratio']:.2f}\n"
        f"  Actions (B/S/H) : {metrics['buy_count']} / {metrics['sell_count']} / {metrics['hold_count']}\n"
        f"\n"
        f"**Benchmark : Buy & Hold**\n"
        f"  Rendement : {metrics['buy_hold_profit_display']} ({metrics['buy_hold_return_pct']:.2f}%)"
    )
    ax1.text(0.015, 0.985, stats_text, transform=ax1.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85))

    # We need to make sure the layout adjustment happens before saving.
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    
    # Save the plot with a symbol-specific name
    output_filename = f'Vizu/{symbol}_performance.png'
    fig.savefig(output_filename)
    plt.close(fig) # Close the figure to free up memory
    logging.info(f"📈 Graphique de performance pour {symbol} sauvegardé sous `{output_filename}`")

def plot_global_performance(all_results, global_stats):
    """
    Generates a global performance report comparing all strategies.
    """
    if not all_results:
        logging.warning("No results to plot for global performance.")
        return

    # --- Data Preparation ---
    # Create a unified date index from all evaluation periods
    all_dates = set()
    for res in all_results:
        all_dates.update(pd.to_datetime(res['trade_dates']))
    
    date_index = pd.DatetimeIndex(sorted(list(all_dates)))
    global_df = pd.DataFrame(index=date_index)

    # Process each symbol's results and add to the global DataFrame
    for res in all_results:
        symbol = res['symbol']
        
        # --- Cumulative Profit in % ---
        # The dates are now guaranteed to be in sync with the profits
        df_profit = pd.DataFrame({
            'dates': pd.to_datetime(res['trade_dates']),
            'profit_usd': res['cumulative_profit_usd']
        }).set_index('dates')
        
        initial_price = res['initial_price']
        if initial_price > 0:
            df_profit[f'{symbol}_profit_pct'] = (df_profit['profit_usd'] / initial_price) * 100
        else:
            df_profit[f'{symbol}_profit_pct'] = 0
        
        global_df = global_df.join(df_profit[[f'{symbol}_profit_pct']])

        # --- Open Trades Count ---
        open_trades = 0
        trade_counts = []
        for _, action in res['history']:
            if action == 'BUY':
                open_trades += 1
            elif action == 'SELL':
                # Ensure open trades don't go below zero
                open_trades = max(0, open_trades - 1)
            trade_counts.append(open_trades)

        df_trades = pd.DataFrame({
            'dates': pd.to_datetime(res['trade_dates']),
            f'{symbol}_open_trades': trade_counts
        }).set_index('dates')
        global_df = global_df.join(df_trades)

    # --- Data Cleaning ---
    # Forward-fill to propagate last known values, then back-fill for initial NaNs
    global_df.ffill(inplace=True)
    global_df.bfill(inplace=True)
    global_df.fillna(0, inplace=True) # Fill any remaining NaNs with 0

    # Calculate total open trades across all symbols
    trade_cols = [col for col in global_df.columns if 'open_trades' in col]
    global_df['total_open_trades'] = global_df[trade_cols].sum(axis=1)

    # Find the peak of open positions and log it
    max_open_positions = global_df['total_open_trades'].max()
    if not global_df.empty:
        max_open_positions_date = global_df['total_open_trades'].idxmax()
        logging.info(f"PIC des positions ouvertes : {max_open_positions:.0f} positions le {max_open_positions_date.date()}")
    else:
        logging.info(f"PIC des positions ouvertes : {max_open_positions:.0f} positions.")

    # --- Plotting ---
    plt.style.use('fivethirtyeight')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 15), dpi=300, sharex=True)
    fig.suptitle('Analyse de Performance Globale des Stratégies', fontsize=24, weight='bold')

    # Calculate global average profit
    profit_cols = [col for col in global_df.columns if 'profit_pct' in col]
    if profit_cols:
        global_df['average_profit_pct'] = global_df[profit_cols].mean(axis=1)

    # Plot 1: Cumulative Profit (%)
    if profit_cols:
        # Plot individual assets with a lighter style
        for col in profit_cols:
            label = col.replace('_profit_pct', '')
            global_df[col].plot(ax=ax1, linewidth=1.2, alpha=0.6, label=label)
        
        # Plot average line, "en gras" (thicker and dashed)
        if 'average_profit_pct' in global_df.columns:
            global_df['average_profit_pct'].plot(ax=ax1, color='black', linewidth=3, linestyle='--', label='Moyenne Globale')

    ax1.set_ylabel('Profit Cumulé (%)', fontsize=14, weight='bold')
    ax1.set_title('Évolution du Profit en Pourcentage par Actif', fontsize=16)
    ax1.legend(title='Symboles & Moyenne', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot 2: Total Open Trades
    global_df['total_open_trades'].plot(ax=ax2, color='navy', linewidth=2, label='Total Positions Ouvertes')
    ax2.fill_between(global_df.index, global_df['total_open_trades'], color='lightblue', alpha=0.4)
    ax2.set_ylabel('Nombre de Positions Ouvertes', fontsize=14, weight='bold')
    ax2.set_title(f'Nombre Total de Positions Ouvertes au Fil du Temps (Pic: {max_open_positions:.0f})', fontsize=16)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.legend(loc='upper left')

    # --- Key Figures Card ---
    stats_text = (
        f"**Chiffres Clés Globaux**\n\n"
        f"Rendement Total Moyen: {global_stats['avg_total_return']:.2f}%\n"
        f"Rendement Moyen/Trade: {global_stats['avg_return_per_trade']:.2f}%\n"
        f"--------------------------------------\n"
        f"Achats: {global_stats['total_buys']}\n"
        f"Ventes: {global_stats['total_sells']}\n"
        f"Holds: {global_stats['total_holds']}\n"
        f"Trades Complets: {global_stats['total_completed_trades']}\n"
        f"Positions Finales: {global_stats['final_open_positions']}"
    )
    
    # Place text box to the right of the plots. We use subplots_adjust to create space.
    fig.text(0.83, 0.45, stats_text, 
             fontsize=12,
             verticalalignment='center', 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95))

    # Adjust subplot parameters to make room for the card and legend on the right.
    # This replaces tight_layout for more precise control.
    fig.subplots_adjust(left=0.07, right=0.8, top=0.94, bottom=0.06)
    
    output_filename = 'Vizu/rapport_performance_global.png'
    fig.savefig(output_filename)
    plt.close(fig)
    logging.info(f"📊 Rapport global de performance sauvegardé sous `{output_filename}`")


def run_backtest_for_symbol(symbol):
    """
    Runs the backtest evaluation for a single symbol.
    """
    data_file = f"datasets/{symbol}/test_data.csv"
    model_name = f"models/model_{symbol.lower()}_colab_best.keras"

    if not os.path.exists(data_file):
        logging.warning(f"Data file not found for {symbol}: {data_file}. Skipping.")
        return None
    if not os.path.exists(model_name):
        logging.warning(f"Model file not found for {symbol}: {model_name}. Skipping.")
        return None

    # Load data
    data = get_stock_data(data_file)
    
    # Re-read the CSV to get the FGIndex column easily
    full_df = pd.read_csv(data_file)
    fg_index = full_df['FGIndex'].values

    window_size = 10 # Should be consistent with the trained model
    n_features = data['features'].shape[1]

    # Initialize Environment and Agent
    env = TradingEnvironment(data, window_size)
    agent = Agent(window_size, n_features, pretrained=True, model_name=model_name)

    # Evaluate the model
    agent_profit, history, cumulative_profits, trade_dates = evaluate_model(agent, env, debug=False)

    # --- Calculate All Metrics ---
    metrics = analyze_trades(history)
    metrics['agent_profit'] = agent_profit

    initial_price = data['prices'][0]
    final_price = data['prices'][-1]
    buy_hold_profit = final_price - initial_price
    
    metrics['agent_return_pct'] = (agent_profit / initial_price) * 100 if initial_price != 0 else 0
    metrics['buy_hold_return_pct'] = (buy_hold_profit / initial_price) * 100 if initial_price != 0 else 0

    metrics['agent_profit_display'] = format_position(agent_profit)
    metrics['buy_hold_profit_display'] = format_position(buy_hold_profit)

    # --- Display Results ---
    print("\n" + "="*50)
    print(f"📊 Analyse de Performance pour le Modèle: {symbol}")
    print("="*50)
    print(f"  - Rendement Stratégie : {metrics['agent_profit_display']} ({metrics['agent_return_pct']:.2f}%)")
    print(f"  - Rendement Buy & Hold: {metrics['buy_hold_profit_display']} ({metrics['buy_hold_return_pct']:.2f}%)")
    print(f"  - Trades Complets: {metrics['total_trades']} | Winrate: {metrics['win_rate']:.2f}% | Profit Moyen/Trade: {metrics['average_profit_pct']:.2f}%")
    print("="*50 + "\n")

    # Generate and save the plot for this single symbol
    plot_single_performance(data, history, metrics, cumulative_profits, model_name, window_size, fg_index, symbol)
    
    # Return data needed for the global plot
    return {
        "symbol": symbol,
        "trade_dates": trade_dates, # Use the synchronized dates
        "cumulative_profit_usd": cumulative_profits,
        "initial_price": initial_price,
        "history": history,
        "metrics": metrics
    }

def main():
    """
    Main function to run batch evaluation and generate global visualization.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    symbols = get_symbols()
    if not symbols:
        logging.error("Aucun symbole trouvé dans le dossier 'datasets'. Le script va s'arrêter.")
        return

    all_results = []
    for symbol in symbols:
        logging.info(f"--- Traitement du symbole : {symbol} ---")
        try:
            result = run_backtest_for_symbol(symbol)
            if result:
                all_results.append(result)
        except Exception as e:
            logging.error(f"Une erreur est survenue lors du traitement du symbole {symbol}: {e}", exc_info=True)

    if not all_results:
        logging.error("Aucun backtest n'a pu être exécuté avec succès. Le rapport global ne sera pas généré.")
        return

    # --- Global Stats Calculation ---
    total_buys = sum(res['metrics']['buy_count'] for res in all_results)
    total_sells = sum(res['metrics']['sell_count'] for res in all_results)
    total_holds = sum(res['metrics']['hold_count'] for res in all_results)
    total_completed_trades = sum(res['metrics']['total_trades'] for res in all_results)
    final_open_positions = total_buys - total_sells
    
    # Calculate average returns
    valid_trade_results = [res['metrics']['average_profit_pct'] for res in all_results if res['metrics']['total_trades'] > 0]
    avg_return_per_trade = np.mean(valid_trade_results) if valid_trade_results else 0
    avg_total_return = np.mean([res['metrics']['agent_return_pct'] for res in all_results])

    global_stats = {
        "avg_total_return": avg_total_return,
        "avg_return_per_trade": avg_return_per_trade,
        "total_buys": total_buys,
        "total_sells": total_sells,
        "total_holds": total_holds,
        "total_completed_trades": total_completed_trades,
        "final_open_positions": final_open_positions
    }

    print("\n" + "="*50)
    print("📊 Synthèse Globale des Transactions")
    print("="*50)
    print(f"  - Rendement Total Moyen ....................: {global_stats['avg_total_return']:.2f}%")
    print(f"  - Rendement Moyen/Trade ..................: {global_stats['avg_return_per_trade']:.2f}%")
    print(f"  - Achats (total) .........................: {global_stats['total_buys']}")
    print(f"  - Ventes (total) .........................: {global_stats['total_sells']}")
    print(f"  - Holds (total) ..........................: {global_stats['total_holds']}")
    print(f"  - Trades Complets ........................: {global_stats['total_completed_trades']}")
    print(f"  - Positions Ouvertes (fin de période) ....: {global_stats['final_open_positions']}")
    print("="*50 + "\n")

    # Generate the global performance plot
    plot_global_performance(all_results, global_stats)
    
    logging.info("Toutes les analyses sont terminées.")

if __name__ == "__main__":
    main()