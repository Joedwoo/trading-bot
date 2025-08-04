"""
Script for training Stock Trading Bot on multiple assets.

Usage:
  train.py <data-dir> [--strategy=<strategy>]
    [--window-size=<window-size>] [--batch-size=<batch-size>]
    [--episode-count=<episode-count>] [--model-name=<model-name>]
    [--patience=<patience>] [--time-penalty=<time-penalty>] [--pretrained] [--debug]

Options:
  --strategy=<strategy>             Q-learning strategy to use. [default: t-dqn]
  --window-size=<window-size>       Size of the n-day window. [default: 10]
  --batch-size=<batch-size>         Number of samples to train on in one mini-batch. [default: 32]
  --episode-count=<episode-count>   Number of epochs to use for training. [default: 50]
  --model-name=<model-name>         Name of the pretrained model to use. [default: model_debug]
  --patience=<patience>             Patience for early stopping. [default: 3]
  --time-penalty=<time-penalty>     Penalty factor for holding a position over time. [default: 0.0]
  --pretrained                      Continue training a previously trained model.
  --debug                           Enable verbose logs.
"""

import logging
import coloredlogs
import os
import random

from docopt import docopt

from trading_bot.agent import Agent
from trading_bot.methods import train_model, evaluate_model
from trading_bot.environment import TradingEnvironment
from trading_bot.utils import (
    load_prepared_data,
    format_currency,
    format_position,
    show_train_result,
    switch_k_backend_device,
)


def main(data_dir, window_size, batch_size, ep_count, patience=3,
         strategy="t-dqn", model_name="model_debug", time_penalty=0.0,
         pretrained=False, debug=False):
    """ Trains the stock trading bot using Deep Q-Learning on multiple assets.
    """
    asset_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if not asset_dirs:
        logging.error(f"Aucun dossier d'actif trouvé dans {data_dir}")
        return
        
    logging.info(f"{len(asset_dirs)} actifs détectés pour l'entraînement.")

    temp_train_data, _, _ = load_prepared_data(asset_dirs[0])
    n_features = temp_train_data['features'].shape[1]
    
    agent = Agent(window_size, n_features, strategy=strategy, 
                  pretrained=pretrained, model_name=model_name)

    best_val_profit = -float('inf')
    patience_counter = 0
    best_epoch = 0
    
    print("=" * 80)
    logging.info("🚀 Début de l'entraînement multi-actifs...")
    logging.info(f"  - Stratégie: {strategy}")
    logging.info(f"  - Époques: {ep_count}")
    logging.info(f"  - Pénalité de temps: {time_penalty}")
    logging.info(f"  - Modèle: {model_name}")
    print("=" * 80)

    total_episodes = 0
    for epoch in range(1, ep_count + 1):
        print(f"\n{'─' * 60}")
        logging.info(f"🏛️  Époque {epoch}/{ep_count}")
        
        random.shuffle(asset_dirs)

        epoch_has_improved = False
        for asset_path in asset_dirs:
            asset_name = os.path.basename(asset_path)
            total_episodes += 1
            
            try:
                train_data, val_data, _ = load_prepared_data(asset_path)

                # Pass the time_penalty to the environment
                train_env = TradingEnvironment(train_data, window_size, time_penalty=time_penalty)
                val_env = TradingEnvironment(val_data, window_size, time_penalty=time_penalty)
                
                initial_offset = val_data['prices'][1] - val_data['prices'][0] if len(val_data['prices']) > 1 else 0

                train_result = train_model(agent, total_episodes, train_env, ep_count=ep_count * len(asset_dirs),
                                           batch_size=batch_size)
                val_result, _, _ = evaluate_model(agent, val_env, debug)
                
                logging.info(f"Résultat validation pour {asset_name}:")
                show_train_result(train_result, val_result, initial_offset)

                if val_result > best_val_profit:
                    best_val_profit = val_result
                    best_epoch = epoch
                    agent.save_best()
                    epoch_has_improved = True
                    logging.info(f"🏆 Nouveau record sur '{asset_name}'! Validation: {format_currency(best_val_profit)}")
            
            except Exception as e:
                logging.error(f"Erreur sur {asset_name}: {e}")
                continue
        
        if epoch_has_improved:
            patience_counter = 0
        else:
            patience_counter += 1
            logging.info(f"⏳ Pas d'amélioration depuis {patience_counter}/{patience} époque(s)")
        
        if patience_counter >= patience:
            logging.info(f"🛑 Early stopping déclenché !")
            break
        
    print(f"\n{'═' * 80}")
    logging.info("✅ Entraînement terminé !")
    logging.info(f"🏆 Meilleur résultat validation: {format_currency(best_val_profit)} (Époque {best_epoch})")
    print(f"{'═' * 80}")


if __name__ == "__main__":
    args = docopt(__doc__)

    data_dir = args["<data-dir>"]
    strategy = args["--strategy"]
    window_size = int(args["--window-size"])
    batch_size = int(args["--batch-size"])
    ep_count = int(args["--episode-count"])
    model_name = args["--model-name"]
    patience = int(args["--patience"])
    time_penalty = float(args["--time-penalty"])
    pretrained = args["--pretrained"]
    debug = args["--debug"]

    coloredlogs.install(level="INFO")
    switch_k_backend_device()

    try:
        main(data_dir, window_size, batch_size, ep_count, patience,
             strategy=strategy, model_name=model_name, time_penalty=time_penalty,
             pretrained=pretrained, debug=debug)
    except KeyboardInterrupt:
        print("Aborted!")
    except Exception as e:
        logging.error(f"Erreur: {e}", exc_info=True)
