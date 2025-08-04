"""
Script for training Stock Trading Bot on multiple assets.

Usage:
  train.py <data-dir> [--strategy=<strategy>]
    [--window-size=<window-size>] [--batch-size=<batch-size>]
    [--episode-count=<episode-count>] [--model-name=<model-name>]
    [--patience=<patience>] [--pretrained] [--debug]

Options:
  --strategy=<strategy>             Q-learning strategy to use for training the network. Options:
                                     `dqn` i.e. Vanilla DQN,
                                     `t-dqn` i.e. DQN with fixed target distribution,
                                     `double-dqn` i.e. DQN with separate network for value estimation. [default: t-dqn]
  --window-size=<window-size>       Size of the n-day window stock data representation
                                   used as the feature vector. [default: 10]
  --batch-size=<batch-size>         Number of samples to train on in one mini-batch
                                   during training. [default: 32]
  --episode-count=<episode-count>   Number of epochs to use for training (one epoch is one pass over all assets). [default: 50]
  --model-name=<model-name>         Name of the pretrained model to use. [default: model_debug]
  --patience=<patience>             Number of epochs to wait for improvement before early stopping. [default: 3]
  --pretrained                      Specifies whether to continue training a previously
                                   trained model (reads `model-name`).
  --debug                           Specifies whether to use verbose logs during eval operation.
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
         strategy="t-dqn", model_name="model_debug", pretrained=False,
         debug=False):
    """ Trains the stock trading bot using Deep Q-Learning on multiple assets.
    """
    # Lister tous les datasets disponibles
    asset_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if not asset_dirs:
        logging.error(f"Aucun dossier d'actif trouvé dans {data_dir}")
        return
        
    logging.info(f"{len(asset_dirs)} actifs détectés pour l'entraînement.")

    # On charge un dataset pour initialiser l'agent avec les bonnes dimensions
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
    logging.info(f"  - Actifs: {len(asset_dirs)}")
    logging.info(f"  - Modèle: {model_name}")
    logging.info(f"  - Patience: {patience} époques")
    print("=" * 80)

    total_episodes = 0
    for epoch in range(1, ep_count + 1):
        print(f"\n{'─' * 60}")
        logging.info(f"🏛️  Époque {epoch}/{ep_count}")
        logging.info(f"{'─' * 60}")
        
        random.shuffle(asset_dirs) # Mélanger les actifs à chaque époque

        epoch_has_improved = False
        for asset_path in asset_dirs:
            asset_name = os.path.basename(asset_path)
            total_episodes += 1
            
            print(f"\n{'─' * 25} Entraînement sur: {asset_name} {'─' * 25}")
            
            try:
                # Charger les données pour l'actif courant
                train_data, val_data, _ = load_prepared_data(asset_path)

                # Initialiser les environnements
                train_env = TradingEnvironment(train_data, window_size)
                val_env = TradingEnvironment(val_data, window_size)
                
                initial_offset = val_data['prices'][1] - val_data['prices'][0]

                # Entraîner pour un épisode
                train_result = train_model(agent, total_episodes, train_env, ep_count=ep_count * len(asset_dirs),
                                           batch_size=batch_size)
                val_result, _, _ = evaluate_model(agent, val_env, debug)
                
                print(f"Résultat validation pour {asset_name}:")
                show_train_result(train_result, val_result, initial_offset)

                # Mise à jour du meilleur modèle
                if val_result > best_val_profit:
                    best_val_profit = val_result
                    best_epoch = epoch
                    agent.save_best()
                    epoch_has_improved = True
                    logging.info(f"🏆 Nouveau record sur '{asset_name}'! Validation: ${best_val_profit:.2f}")
                    logging.info(f"💾 Meilleur modèle sauvegardé")

            except Exception as e:
                logging.error(f"Erreur lors de l'entraînement sur {asset_name}: {e}")
                continue
        
        # Gestion de la patience à la fin de chaque époque
        print(f"\n{'─' * 40}")
        logging.info(f"Fin de l'époque {epoch}")
        if epoch_has_improved:
            patience_counter = 0
            logging.info(f"✅ Amélioration détectée dans cette époque.")
        else:
            patience_counter += 1
            logging.info(f"⏳ Pas d'amélioration depuis {patience_counter}/{patience} époque(s)")
        
        logging.info(f"🎯 Record actuel: ${best_val_profit:.2f} (Époque {best_epoch})")
            
        if patience_counter >= patience:
            print(f"\n{'═' * 60}")
            logging.info(f"🛑 Early stopping déclenché !")
            logging.info(f"🏆 Meilleur résultat: ${best_val_profit:.2f} (Époque {best_epoch})")
            print(f"{'═' * 60}")
            break
        
        if epoch % 10 == 0:
            agent.save(epoch)
            logging.info(f"💾 Sauvegarde périodique: models/{model_name}_{epoch}")

    print(f"\n{'═' * 80}")
    logging.info("✅ Entraînement terminé !")
    print(f"{'═' * 80}")
    logging.info(f"📊 RÉSUMÉ FINAL:")
    logging.info(f"  🏆 Meilleur résultat validation: ${best_val_profit:.2f} (Époque {best_epoch})")
    logging.info(f"  💾 Meilleur modèle: models/{model_name}_best")
    logging.info(f"  📈 Total époques: {epoch}/{ep_count}")
    logging.info(f"  🤖 Total épisodes d'entraînement: {total_episodes}")
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
    pretrained = args["--pretrained"]
    debug = args["--debug"]

    coloredlogs.install(level="INFO") # Changed to INFO for cleaner logs
    switch_k_backend_device()

    try:
        main(data_dir, window_size, batch_size, ep_count, patience,
             strategy=strategy, model_name=model_name, 
             pretrained=pretrained, debug=debug)
    except KeyboardInterrupt:
        print("Aborted!")
    except Exception as e:
        logging.error(f"Erreur: {e}", exc_info=True)
