"""
Script for Batch Backtesting a Stock Trading Bot on multiple test datasets.
This script generates both a performance summary and detailed plots for each asset.

Usage:
  batch_backtest.py <data-dir> <model-name> [--window-size=<window-size>] [--debug]

Options:
  <data-dir>                    Directory containing the split data folders for each asset.
  <model-name>                  Name of the pretrained model to use for evaluation.
  --window-size=<window-size>   Size of the n-day window stock data representation. [default: 10]
  --debug                       Enable verbose logs during evaluation.
"""

import os
import logging
import coloredlogs
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for saving plots
import matplotlib.pyplot as plt
from docopt import docopt

from trading_bot.agent import Agent
from trading_bot.methods import evaluate_model
from trading_bot.environment import TradingEnvironment
from trading_bot.utils import (
    load_prepared_data,
    format_currency,
    format_position,
    switch_k_backend_device
)


def analyze_trades(history):
    """
    Analyzes the trade history to calculate performance metrics.
    """
    buy_count = sum(1 for _, action, _ in history if action == "BUY")
    sell_count = sum(1 for _, action, _ in history if action == "SELL")
    hold_count = sum(1 for _, action, _ in history if action == "HOLD")

    dollar_trades = []
    percent_trades = []
    open_positions = []

    for price, action, _ in history:
        if action == "BUY":
            open_positions.append(price)
        elif action == "SELL":
            if open_positions:
                buy_price = open_positions.pop(0)
                profit = price - buy_price
                dollar_trades.append(profit)
                if buy_price != 0:
                    percent_trades.append((profit / buy_price) * 100)

    if not dollar_trades:
        return { "win_rate": 0, "total_trades": 0, "average_profit_usd": 0,
                 "average_profit_pct": 0, "sharpe_ratio": 0, "buy_count": buy_count,
                 "sell_count": sell_count, "hold_count": hold_count }

    dollar_trades = np.array(dollar_trades)
    wins = dollar_trades > 0
    win_rate = (np.sum(wins) / len(dollar_trades)) * 100 if dollar_trades.size > 0 else 0
    average_profit_usd = np.mean(dollar_trades)
    average_profit_pct = np.mean(percent_trades) if percent_trades else 0
    std_dev_profit = np.std(dollar_trades)
    sharpe_ratio = average_profit_usd / std_dev_profit if std_dev_profit != 0 else 0

    return {
        "win_rate": win_rate, "total_trades": len(dollar_trades),
        "average_profit_usd": average_profit_usd, "average_profit_pct": average_profit_pct,
        "sharpe_ratio": sharpe_ratio, "buy_count": buy_count,
        "sell_count": sell_count, "hold_count": hold_count,
    }

def plot_performance(data, history, metrics, cumulative_profits, model_name, window_size, asset_name, env):
    """
    Generates and saves a plot of the agent's performance.
    """
    prices = data['prices']
    dates = pd.to_datetime(data['dates'])
    
    # The evaluation starts from the first point where a full window is available
    eval_start_index = window_size - 1
    
    # The dates corresponding to the evaluation period
    eval_dates = dates[eval_start_index:]
    
    # Safety check: ensure all lists have the same size to prevent plotting errors.
    min_len = min(len(eval_dates), len(history), len(cumulative_profits))
    eval_dates = eval_dates[:min_len]
    history = history[:min_len]
    cumulative_profits = cumulative_profits[:min_len]

    buy_points = [(eval_dates[i], price) for i, (price, action, _) in enumerate(history) if action == "BUY"]
    sell_points = [(eval_dates[i], price) for i, (price, action, _) in enumerate(history) if action == "SELL"]

    plt.style.use('fivethirtyeight')
    fig, ax1 = plt.subplots(figsize=(16, 8), dpi=300)
    
    fig.suptitle(f'Performance pour {asset_name} avec {model_name}', fontsize=16, weight='bold')

    ax1.plot(dates, prices, label='Prix de Clôture', color='dimgray', linewidth=1.5, alpha=0.8)
    if buy_points:
        buy_dates, buy_prices = zip(*buy_points)
        ax1.scatter(buy_dates, buy_prices, label='Achat', marker='^', color='#2ca02c', s=100, edgecolors='black', zorder=5)
    if sell_points:
        sell_dates, sell_prices = zip(*sell_points)
        ax1.scatter(sell_dates, sell_prices, label='Vente', marker='v', color='#d62728', s=100, edgecolors='black', zorder=5)
    
    ax1.set_ylabel('Prix ($)', fontsize=12, weight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax2 = ax1.twinx()
    ax2.plot(eval_dates, cumulative_profits, label='Profit Cumulé', color='royalblue', linewidth=2)
    ax2.set_ylabel('Profit Cumulé ($)', fontsize=12, weight='bold', color='royalblue')
    ax2.tick_params(axis='y', labelcolor='royalblue')
    ax2.legend(loc='lower left', fontsize=10)

    # Escape dollar signs for Matplotlib rendering
    agent_profit_display = metrics['agent_profit_display'].replace('$', r'\$')
    initial_balance_display = format_currency(env.initial_balance).replace('$', r'\$')
    
    buy_hold_profit_on_capital = (metrics['buy_hold_return_pct'] / 100) * env.initial_balance if hasattr(env, 'initial_balance') else metrics['buy_hold_profit_display']
    buy_hold_display = f"{format_position(buy_hold_profit_on_capital)} ({metrics['buy_hold_return_pct']:.2f}%)".replace('$', r'\$')
    
    avg_profit_display = format_currency(metrics['average_profit_usd']).replace('$', r'\$')

    stats_text = (
        f"Rendement Agent: {agent_profit_display} ({metrics['agent_return_pct']:.2f}% sur {initial_balance_display})\n"
        f"Rendement Buy & Hold: {buy_hold_display}\n"
        f"Winrate: {metrics['win_rate']:.2f}% | Trades: {metrics['total_trades']}\n"
        f"Profit Moyen/Trade: {avg_profit_display}"
    )
    ax1.text(0.01, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    os.makedirs('plots', exist_ok=True)
    output_filename = f"plots/rapport_performance_{asset_name}.png"
    fig.savefig(output_filename)
    plt.close(fig)
    logging.info(f"📈 Graphique de performance sauvegardé : {output_filename}")


def main(data_dir, model_name, window_size, debug):
    asset_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if not asset_dirs:
        logging.error(f"Aucun dossier d'actif trouvé dans {data_dir}")
        return

    logging.info(f"Début du backtesting pour le modèle : {model_name} sur {len(asset_dirs)} actifs.")
    
    try:
        _, _, temp_test_data = load_prepared_data(os.path.join(data_dir, asset_dirs[0]))
        n_features = temp_test_data['features'].shape[1]
    except Exception as e:
        logging.error(f"Impossible de charger les données initiales : {e}")
        return

    agent = Agent(window_size, n_features, pretrained=True, model_name=model_name)
    
    all_results = []
    total_profit = 0

    print("=" * 80)
    
    for asset_folder in asset_dirs:
        asset_path = os.path.join(data_dir, asset_folder)
        logging.info(f"🔄 Évaluation de l'actif : {asset_folder}...")
        
        try:
            _, _, test_data = load_prepared_data(asset_path)
            
            if len(test_data['prices']) <= window_size:
                logging.warning(f"  -> Pas assez de données de test pour {asset_folder}. Ignoré.")
                continue

            env = TradingEnvironment(test_data, window_size, initial_balance=2000, trade_size=500)
            
            agent_profit, history, cumulative_profits = evaluate_model(agent, env, debug)
            
            metrics = analyze_trades(history)
            metrics['agent_profit'] = agent_profit
            
            initial_price = test_data['prices'][0]
            final_price = test_data['prices'][-1]
            buy_hold_profit = final_price - initial_price
            
            metrics['agent_return_pct'] = (agent_profit / env.initial_balance) * 100 if env.initial_balance != 0 else 0
            metrics['buy_hold_return_pct'] = (buy_hold_profit / initial_price) * 100 if initial_price != 0 else 0
            
            metrics['agent_profit_display'] = format_position(agent_profit)
            metrics['buy_hold_profit_display'] = format_position(buy_hold_profit)

            logging.info(f"  - Profit Agent: {metrics['agent_profit_display']} ({metrics['agent_return_pct']:.2f}%)")
            logging.info(f"  - Winrate: {metrics['win_rate']:.2f}% ({metrics['total_trades']} trades)")
            
            plot_performance(test_data, history, metrics, cumulative_profits, model_name, window_size, asset_folder, env)
            
            all_results.append({"Actif": asset_folder, "Profit": agent_profit, "Winrate(%)": metrics['win_rate'], "Nb_Trades": metrics['total_trades']})
            total_profit += agent_profit
            print("-" * 40)

        except FileNotFoundError:
            logging.warning(f"  -> Données de test non trouvées pour {asset_folder}. Ignoré.")
            continue
        except Exception as e:
            logging.error(f"  -> Erreur lors de l'évaluation de {asset_folder}: {e}", exc_info=debug)
            continue
    
    print("\n" + "=" * 80)
    logging.info("📊 RÉSUMÉ FINAL DU BACKTESTING 📊")
    print("=" * 80)

    if all_results:
        results_df = pd.DataFrame(all_results)
        profitable_trades = (results_df['Profit'] > 0).sum()
        
        print(results_df.to_string(index=False))
        
        output_file = f"backtest_summary_{model_name}.csv"
        results_df.to_csv(output_file, index=False)
        logging.info(f"\n💾 Résumé sauvegardé dans : {output_file}")

        print("-" * 80)
        logging.info(f"📈 Performance Globale :")
        logging.info(f"  - Profit Total : {format_currency(total_profit)}")
        logging.info(f"  - Profit Moyen par Actif : {format_currency(results_df['Profit'].mean())}")
        logging.info(f"  - Winrate Moyen : {results_df['Winrate(%)'].mean():.2f}%")
        logging.info(f"  - Actifs Rentables : {profitable_trades} / {len(all_results)} ({ (profitable_trades / len(all_results) * 100):.2f}%)")
    print("=" * 80)


if __name__ == "__main__":
    args = docopt(__doc__)
    data_dir = args["<data-dir>"]
    model_name = args["<model-name>"]
    window_size = int(args["--window-size"])
    debug = args["--debug"]

    coloredlogs.install(level="INFO")
    switch_k_backend_device()

    try:
        main(data_dir, model_name, window_size, debug)
    except KeyboardInterrupt:
        print("\nProcessus de backtesting interrompu.")
    except Exception as e:
        logging.error(f"Une erreur inattendue est survenue: {e}", exc_info=True)
