"""
Script for single-model backtesting.

Usage:
  backtest.py <model-path> <data-file> [--output-dir=<output-dir>] [--window-size=<window-size>]

Options:
  --output-dir=<output-dir>   Directory to save backtest results. [default: backtest_results]
  --window-size=<window-size> Window size used during training. [default: 10]
"""
import os
import re
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt
from tensorflow.keras.utils import get_custom_objects
from docopt import docopt

from trading_bot.agent import Agent, huber_loss
from trading_bot.environment import TradingEnvironment
from trading_bot.methods_batch import evaluate_model
from trading_bot.utils import get_stock_data, format_currency, format_position

# Enregistrer manuellement la fonction de perte pour la désérialisation
get_custom_objects().update({'huber_loss': huber_loss})

def analyze_trades(history):
    """
    Analyzes the trade history to calculate performance metrics.

    Args:
        history (list): A list of tuples `(price, action, q_values)` from the evaluation.

    Returns:
        dict: A dictionary containing various performance metrics.
    """
    # --- Action Counts ---
    buy_count = sum(1 for _, action, _ in history if action == "BUY")
    sell_count = sum(1 for _, action, _ in history if action == "SELL")
    hold_count = sum(1 for _, action, _ in history if action == "HOLD")

    # --- Trade Profit Calculations (FIFO Logic) ---
    dollar_trades = []
    percent_trades = []
    open_positions = []  # A queue to store buy prices

    for price, action, _ in history:
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
            'buy_count': buy_count, 'sell_count': sell_count, 'hold_count': hold_count,
            'total_trades': 0, 'win_rate': 0, 'average_profit': 0,
            'average_profit_pct': 0, 'total_profit': 0
        }

    # --- Final Metrics ---
    win_rate = (np.sum(np.array(dollar_trades) > 0) / len(dollar_trades)) * 100 if dollar_trades else 0
    
    return {
        'buy_count': buy_count,
        'sell_count': sell_count,
        'hold_count': hold_count,
        'total_trades': len(dollar_trades),
        'win_rate': win_rate,
        'average_profit': np.mean(dollar_trades),
        'average_profit_pct': np.mean(percent_trades),
        'total_profit': np.sum(dollar_trades)
    }

def plot_performance(data, history, metrics, cumulative_profits, model_name, window_size, fg_index, output_path):
    """
    Plots the agent's performance and saves it to a file.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]})
    
    prices = data['prices']
    dates = pd.to_datetime(data['dates'])
    
    # --- Plot 1: Price, Trades, and Fear & Greed Index ---
    ax1.plot(dates, prices, label='Price', color='dodgerblue', alpha=0.7)
    
    buys = [(dates[i], prices[i]) for i, (_, act, _) in enumerate(history) if act == "BUY"]
    sells = [(dates[i], prices[i]) for i, (_, act, _) in enumerate(history) if act == "SELL"]

    if buys:
        buy_dates, buy_prices = zip(*buys)
        ax1.scatter(buy_dates, buy_prices, marker='^', color='green', s=100, label='Buy', zorder=5)
    if sells:
        sell_dates, sell_prices = zip(*sells)
        ax1.scatter(sell_dates, sell_prices, marker='v', color='red', s=100, label='Sell', zorder=5)
        
    ax1.set_title(f'Performance du Modèle: {model_name}', fontsize=16)
    ax1.set_ylabel('Prix ($)', fontsize=12)
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- Plot 2: Cumulative Profit ---
    ax2.plot(dates, cumulative_profits, label='Profit Cumulatif', color='purple')
    ax2.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax2.set_ylabel('Profit ($)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # --- Add Performance Metrics Table ---
    fig.text(0.86, 0.95, "Métriques de Performance", ha='left', va='top', fontsize=12, weight='bold')
    
    metrics_text = (
        f"Profit Total: {format_position(metrics['agent_profit'])}\n"
        f"Rendement Agent: {metrics['agent_return_pct']:.2f}%\n"
        f"Rendement Buy&Hold: {metrics['buy_hold_return_pct']:.2f}%\n\n"
        f"Total Trades: {metrics['total_trades']}\n"
        f"Taux de Réussite: {metrics['win_rate']:.2f}%\n"
        f"Profit Moyen/Trade: {format_position(metrics['average_profit'])}\n\n"
        f"Achats: {metrics['buy_count']} | Ventes: {metrics['sell_count']}"
    )
    
    fig.text(0.86, 0.85, metrics_text, ha='left', va='top', fontsize=10, linespacing=1.5,
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.3))
             
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logging.info(f"📈 Graphique de performance sauvegardé sous `{output_path}`")
    plt.close(fig)

def run_backtest(model_path, data_file, output_dir, window_size=10):
    """
    Main function to run the evaluation and visualization for a single model.
    """
    model_name = os.path.basename(model_path)
    logging.info(f"--- Running backtest for {model_name} on {data_file} ---")

    # Load data
    data = get_stock_data(data_file)
    
    # Re-read the CSV to get the FGIndex column easily
    full_df = pd.read_csv(data_file)
    fg_index = full_df['FGIndex'].values if 'FGIndex' in full_df.columns else np.full(len(data['prices']), 50)

    n_features = data['features'].shape[1]

    # Initialize Environment and Agent
    env = TradingEnvironment(data, window_size)
    agent = Agent(window_size, n_features, pretrained=True, model_name=model_path)

    # Evaluate the model
    agent_profit, history, cumulative_profits = evaluate_model(agent, env, debug=False)

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
    print(f"📊 Analyse de Performance pour le Modèle: {model_name}")
    print("="*50)
    print(f"  - Rendement Stratégie : {metrics['agent_profit_display']} ({metrics['agent_return_pct']:.2f}%)")
    print(f"  - Rendement Buy & Hold: {metrics['buy_hold_profit_display']} ({metrics['buy_hold_return_pct']:.2f}%)")
    print("-"*50)
    print(f"  - Total Trades: {metrics['total_trades']}")
    print(f"  - Taux de Réussite: {metrics['win_rate']:.2f}%")
    print(f"  - Actions: {metrics['buy_count']} Achat(s), {metrics['sell_count']} Vente(s), {metrics['hold_count']} Hold(s)")
    print("="*50)
    
    # --- Save Visualization and Results ---
    plot_output_path = os.path.join(output_dir, f"backtest_{os.path.splitext(model_name)[0]}.png")
    plot_performance(data, history, metrics, cumulative_profits, model_name, window_size, fg_index, plot_output_path)
    
    csv_output_path = os.path.join(output_dir, f"backtest_{os.path.splitext(model_name)[0]}.csv")
    results_df = pd.DataFrame([metrics])
    results_df.to_csv(csv_output_path, index=False)
    logging.info(f"💾 Fichier CSV des résultats sauvegardé sous `{csv_output_path}`")

def main():
    """
    Main entry point for the backtesting script.
    """
    args = docopt(__doc__)
    
    model_path = args['<model-path>']
    data_file = args['<data-file>']
    output_dir = args['--output-dir']
    window_size = int(args.get('--window-size') or 10)
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        run_backtest(model_path, data_file, output_dir, window_size)
    except Exception as e:
        logging.error(f"Une erreur est survenue lors du backtest pour {model_path}: {e}", exc_info=True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
