"""
Script for training Stock Trading Bot.

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
  --episode-count=<episode-count>   Number of trading episodes to use for training. [default: 50]
  --model-name=<model-name>         Name of the pretrained model to use. [default: model_debug]
  --patience=<patience>             Number of episodes to wait for improvement before early stopping. [default: 3]
  --pretrained                      Specifies whether to continue training a previously
                                    trained model (reads `model-name`).
  --debug                           Specifies whether to use verbose logs during eval operation.
"""

import logging
import coloredlogs

from docopt import docopt

from trading_bot.agent import Agent
from trading_bot.methods import train_model, evaluate_model
from trading_bot.utils import (
    load_prepared_data,
    format_currency,
    format_position,
    show_train_result,
    switch_k_backend_device
)


def main(data_dir, window_size, batch_size, ep_count, patience=3,
         strategy="t-dqn", model_name="model_debug", pretrained=False,
         debug=False):
    """ Trains the stock trading bot using Deep Q-Learning.
    Please see https://arxiv.org/abs/1312.5602 for more details.

    Args: [python train.py --help]
    """
    # Charger les données pré-splittées
    logging.info(f"Chargement des données pré-splittées depuis {data_dir}")
    train_data, val_data, test_data = load_prepared_data(data_dir)
    
    # Calculer le nombre de features pour l'agent
    n_features = train_data['features'].shape[1]
    logging.info(f"Nombre de features: {n_features}")
    logging.info(f"Taille de l'état: {(window_size - 1) + n_features}")
    
    # Initialiser l'agent avec la nouvelle signature
    agent = Agent(window_size, n_features, strategy=strategy, 
                  pretrained=pretrained, model_name=model_name)
    
    initial_offset = val_data['prices'][1] - val_data['prices'][0]
    
    # Variables pour early stopping et meilleur modèle
    best_val_profit = -float('inf')
    patience_counter = 0
    best_episode = 0
    
    logging.info("🚀 Début de l'entraînement...")
    logging.info(f"  - Stratégie: {strategy}")
    logging.info(f"  - Episodes: {ep_count}")
    logging.info(f"  - Window size: {window_size}")
    logging.info(f"  - Batch size: {batch_size}")
    logging.info(f"  - Modèle: {model_name}")
    logging.info(f"  - Patience: {patience} épisodes")

    for episode in range(1, ep_count + 1):
        train_result = train_model(agent, episode, train_data, ep_count=ep_count,
                                   batch_size=batch_size, window_size=window_size)
        val_result, _ = evaluate_model(agent, val_data, window_size, debug)
        show_train_result(train_result, val_result, initial_offset)
        
        # Vérifier si c'est le meilleur modèle
        if val_result > best_val_profit:
            best_val_profit = val_result
            best_episode = episode
            patience_counter = 0
            
            # Sauvegarder le meilleur modèle
            agent.save_best()
            logging.info(f"🏆 Nouveau meilleur résultat de validation: ${best_val_profit:.2f} (Episode {episode})")
        else:
            patience_counter += 1
            logging.info(f"⏳ Pas d'amélioration depuis {patience_counter} épisode(s)")
            
            # Early stopping si patience dépassée
            if patience_counter >= patience:
                logging.info(f"🛑 Early stopping déclenché après {patience} épisodes sans amélioration")
                logging.info(f"🏆 Meilleur résultat: ${best_val_profit:.2f} (Episode {best_episode})")
                break
        
        # Sauvegarde périodique (tous les 10 épisodes)
        if episode % 10 == 0:
            agent.save(episode)
    
    logging.info("✅ Entraînement terminé !")
    logging.info(f"🏆 Meilleur résultat de validation: ${best_val_profit:.2f} (Episode {best_episode})")
    logging.info(f"💾 Meilleur modèle: models/{model_name}_best")
    logging.info(f"📁 Modèles sauvegardés: models/{model_name}_*")


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

    coloredlogs.install(level="DEBUG")
    switch_k_backend_device()

    try:
        main(data_dir, window_size, batch_size, ep_count, patience,
             strategy=strategy, model_name=model_name, 
             pretrained=pretrained, debug=debug)
    except KeyboardInterrupt:
        print("Aborted!")
    except Exception as e:
        logging.error(f"Erreur: {e}")
