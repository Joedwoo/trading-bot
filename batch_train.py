"""
Script for batch training of Stock Trading Bots for multiple tickers.

Usage:
  batch_train.py [--strategy=<strategy>] [--episode-count=<episode-count>] [--batch-size=<batch-size>] [--patience=<patience>]

Options:
  --strategy=<strategy>         Q-learning strategy: dqn | t-dqn | double-dqn. [default: double-dqn]
  --episode-count=<episode-count>   Number of trading episodes for each agent. [default: 50]
  --batch-size=<batch-size>         Number of samples to train on in one mini-batch. [default: 32]
  --patience=<patience>             Number of episodes to wait for improvement before early stopping. [default: 5]
"""
import os
import logging
from docopt import docopt
import coloredlogs

from train import main as train_main
from trading_bot.utils import switch_k_backend_device


def batch_train(ep_count, batch_size, patience, strategy):
    """
    Trains agents for all tickers found in the data/split directory.
    """
    base_data_dir = "data/split"
    try:
        tickers = sorted([d for d in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, d))])
        # Find the index of CRWD and slice the list from that point
        try:
            crwd_index = tickers.index("CRWD")
            tickers = tickers[crwd_index:]
        except ValueError:
            logging.error("CRWD not found in the ticker list. Starting from the beginning.")
    except FileNotFoundError:
        logging.error(f"Directory not found: {base_data_dir}. Please run prepare_data.py first.")
        return

    logging.info(f"Found {len(tickers)} tickers to train.")

    for ticker in tickers:
        print(f"\n{'='*80}")
        logging.info(f"🚀 Starting training for ticker: {ticker}")
        print(f"{'='*80}\n")
        
        data_dir = os.path.join(base_data_dir, ticker)
        model_name = f"model_{ticker.lower()}"
        
        # Default parameters from train.py, can be exposed here if needed
        window_size = 10
        pretrained = False
        debug = False
        
        try:
            train_main(
                data_dir,
                window_size,
                batch_size,
                ep_count,
                patience,
                strategy=strategy,
                model_name=model_name,
                pretrained=pretrained,
                debug=debug
            )
        except Exception as e:
            logging.error(f"❌ Error training {ticker}: {e}")
            logging.info("Skipping to next ticker.")
            continue


if __name__ == "__main__":
    args = docopt(__doc__)
    
    strategy = args['--strategy']
    ep_count = int(args['--episode-count'])
    batch_size = int(args['--batch-size'])
    patience = int(args['--patience'])

    coloredlogs.install(level="INFO", format='%(asctime)s %(levelname)s %(message)s')
    switch_k_backend_device()
    
    batch_train(ep_count, batch_size, patience, strategy)
