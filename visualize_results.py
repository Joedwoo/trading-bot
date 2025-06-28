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

    # --- Trade Profit Calculations ---
    trades = []
    buy_price = None

    for price, action in history:
        if action == "BUY":
            if buy_price is None:  # Register the first buy of a sequence
                buy_price = price
        elif action == "SELL":
            if buy_price is not None:
                profit = price - buy_price
                trades.append(profit)
                buy_price = None  # Reset for the next trade

    if not trades:
        return {
            "win_rate": 0,
            "total_trades": 0,
            "average_profit": 0,
            "sharpe_ratio": 0,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "hold_count": hold_count,
        }

    trades = np.array(trades)
    
    # Win Rate
    wins = trades > 0
    win_rate = (np.sum(wins) / len(trades)) * 100
    
    # Average Profit
    average_profit = np.mean(trades)
    
    # Sharpe Ratio (simplified)
    std_dev_profit = np.std(trades)
    sharpe_ratio = average_profit / std_dev_profit if std_dev_profit != 0 else 0

    return {
        "win_rate": win_rate,
        "total_trades": len(trades),
        "average_profit": average_profit,
        "sharpe_ratio": sharpe_ratio,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "hold_count": hold_count,
    }

def plot_performance(data, history, metrics, model_name):
    """
    Generates and saves a modern plot of the agent's performance.
    """
    prices = data['prices']
    dates = pd.to_datetime(data['dates'])
    
    buy_points = [(dates[i], prices[i]) for i, (_, action) in enumerate(history) if action == "BUY"]
    sell_points = [(dates[i], prices[i]) for i, (_, action) in enumerate(history) if action == "SELL"]

    plt.style.use('fivethirtyeight') # Use a modern, aesthetic style
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)

    # Plot price history
    ax.plot(dates, prices, label='Prix de Clôture', color='dimgray', linewidth=1.5, alpha=0.8)

    # Plot Buy/Sell markers
    if buy_points:
        buy_dates, buy_prices = zip(*buy_points)
        ax.scatter(buy_dates, buy_prices, label='Achat', marker='^', color='#2ca02c', s=120, edgecolors='black', zorder=5)
    if sell_points:
        sell_dates, sell_prices = zip(*sell_points)
        ax.scatter(sell_dates, sell_prices, label='Vente', marker='v', color='#d62728', s=120, edgecolors='black', zorder=5)
    
    # --- Text and Labels ---
    ax.set_title(f'Analyse de Performance : {model_name}', fontsize=20, weight='bold')
    ax.set_ylabel('Prix ($)', fontsize=14, weight='bold')
    fig.autofmt_xdate() # Auto-format date labels

    # Add metrics text box
    stats_text = (
        f"**Performance Stratégie**\n"
        f"  Rendement : {metrics['agent_profit_display']} ({metrics['agent_return_pct']:.2f}%)\n"
        f"  Winrate : {metrics['win_rate']:.2f}% | Sharpe : {metrics['sharpe_ratio']:.2f}\n"
        f"  Actions (B/S/H) : {metrics['buy_count']} / {metrics['sell_count']} / {metrics['hold_count']}\n"
        f"\n"
        f"**Benchmark : Buy & Hold**\n"
        f"  Rendement : {metrics['buy_hold_profit_display']} ({metrics['buy_hold_return_pct']:.2f}%)"
    )
    ax.text(0.015, 0.985, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85))

    ax.legend(loc='lower left', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    output_filename = 'rapport_performance.png'
    fig.savefig(output_filename)
    logging.info(f"📈 Graphique de performance sauvegardé sous `{output_filename}`")

def main(data_file, model_name):
    """
    Main function to run the evaluation and visualization.
    """
    # Load data
    data = get_stock_data(data_file)
    window_size = 10 # Should be consistent with the trained model
    n_features = data['features'].shape[1]

    # Initialize Environment and Agent
    env = TradingEnvironment(data, window_size)
    agent = Agent(window_size, n_features, pretrained=True, model_name=model_name)

    # Evaluate the model
    agent_profit, history = evaluate_model(agent, env, debug=False)

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
    print(f"    - Profit Moyen par Trade:     {format_currency(metrics['average_profit'])}")
    print(f"    - Ratio de Sharpe (simplifié):  {metrics['sharpe_ratio']:.2f}")
    print("="*50 + "\n")

    # Generate and save the plot
    plot_performance(data, history, metrics, model_name)

if __name__ == "__main__":
    args = docopt(__doc__)
    data_file = args["<data-file>"]
    model_name = args["--model-name"]

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        main(data_file, model_name)
    except Exception as e:
        logging.error(f"Une erreur est survenue: {e}", exc_info=True) 