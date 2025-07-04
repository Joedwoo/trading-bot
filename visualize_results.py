"""
Script for visualizing and analyzing Stock Trading Bot's performance.

Usage:
  visualize_results.py <data-file> --model-name=<model-name>

Options:
  <data-file>                   The dataset to use for evaluation (e.g., test data).
  --model-name=<model-name>     Name of the pretrained model to use.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt
from docopt import docopt

from trading_bot.agent import Agent
from trading_bot.environment import TradingEnvironment
from trading_bot.methods import evaluate_model
from trading_bot.utils import get_stock_data, format_currency, format_position

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

def plot_performance(data, history, metrics, cumulative_profits, model_name, window_size, fg_index):
    """
    Generates and saves a modern plot of the agent's performance,
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
    fig.suptitle(f'Analyse de Performance : {model_name}', fontsize=20, weight='bold')

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
    ax3.plot(eval_dates, cumulative_profits, label='Profit Cumulé', color='royalblue', linewidth=2)
    ax3.fill_between(eval_dates, cumulative_profits, 0,
                     where=(np.array(cumulative_profits) >= 0),
                     facecolor='green', alpha=0.3, interpolate=True)
    ax3.fill_between(eval_dates, cumulative_profits, 0,
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

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    
    output_filename = 'rapport_performance.png'
    fig.savefig(output_filename)
    logging.info(f"📈 Graphique de performance sauvegardé sous `{output_filename}`")

def main(data_file, model_name):
    """
    Main function to run the evaluation and visualization.
    """
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
    agent_profit, history, cumulative_profits, _ = evaluate_model(agent, env, debug=False)

    # --- Calculate All Metrics ---
    # 1. Agent's performance
    metrics = analyze_trades(history)
    metrics['agent_profit'] = agent_profit

    # 2. Buy and Hold benchmark & Percentage returns
    initial_price = data['prices'][0]
    final_price = data['prices'][-1]
    buy_hold_profit = final_price - initial_price
    
    metrics['agent_return_pct'] = (agent_profit / initial_price) * 100 if initial_price != 0 else 0
    metrics['buy_hold_return_pct'] = (buy_hold_profit / initial_price) * 100 if initial_price != 0 else 0

    # Format values for display
    metrics['agent_profit_display'] = format_position(agent_profit)
    metrics['buy_hold_profit_display'] = format_position(buy_hold_profit)

    # --- Display Results ---
    print("\n" + "="*50)
    print(f"📊 Analyse de Performance pour le Modèle: {model_name}")
    print("="*50)
    print(f"  - Rendement Stratégie : {metrics['agent_profit_display']} ({metrics['agent_return_pct']:.2f}%)")
    print(f"  - Rendement Buy & Hold: {metrics['buy_hold_profit_display']} ({metrics['buy_hold_return_pct']:.2f}%)")
    print("-"*50)
    print("  Actions sur la période :")
    print(f"    - Achats: {metrics['buy_count']} | Ventes: {metrics['sell_count']} | Maintiens (Hold): {metrics['hold_count']}")
    print("-"*50)
    print("  Statistiques des Trades (cycles Achat/Vente complets) :")
    print(f"    - Nombre de Trades Complets:  {metrics['total_trades']}")
    print(f"    - Taux de Réussite (Winrate): {metrics['win_rate']:.2f}%")
    print(f"    - Profit Moyen par Trade:     {metrics['average_profit_pct']:.2f}%")
    print(f"    - Ratio de Sharpe (simplifié):  {metrics['sharpe_ratio']:.2f}")
    print("="*50 + "\n")

    # Generate and save the plot
    plot_performance(data, history, metrics, cumulative_profits, model_name, window_size, fg_index)

if __name__ == "__main__":
    args = docopt(__doc__)
    data_file = args["<data-file>"]
    model_name = args["--model-name"]

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        main(data_file, model_name)
    except Exception as e:
        logging.error(f"Une erreur est survenue: {e}", exc_info=True) 