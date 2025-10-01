"""
Script for training Stock Trading Bot.

Usage:
  train.py <data-path> [--strategy=<strategy>]
    [--window-size=<window-size>] [--batch-size=<batch-size>]
    [--episode-count=<episode-count>] [--model-name=<model-name>]
    [--patience=<patience>] [--pretrained] [--debug] [--cpu]
  train.py --train-csv=<train-csv> --val-csv=<val-csv> [--strategy=<strategy>]
    [--window-size=<window-size>] [--batch-size=<batch-size>]
    [--episode-count=<episode-count>] [--model-name=<model-name>]
    [--patience=<patience>] [--pretrained] [--debug] [--cpu]

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
  --cpu                             Force CPU usage even if a GPU is available.
  --train-csv=<train-csv>           Path to a CSV file for training split (raw CSV mode).
  --val-csv=<val-csv>               Path to a CSV file for validation split (raw CSV mode).

Notes:
  - <data-path> peut être soit un dossier contenant `train_data.csv`, `val_data.csv`, `test_data.csv`,
    soit un fichier CSV brut unique. Dans ce dernier cas, les données seront splittées en mémoire
    (70%/15%/15%) après détection des colonnes `date|timestamp` et `price|close`. Les colonnes
    cibles comme `target` ou `next_return` sont exclues des features.
  - Alternativement, fournir `--train-csv` et `--val-csv` pour entraîner/valider sur deux CSV distincts
    (sans split automatique). Dans ce mode, le test set n'est pas requis et sera aligné sur la validation.
"""

import logging
import coloredlogs

from docopt import docopt

from trading_bot.agent import Agent
from trading_bot.methods import train_model, evaluate_model
from trading_bot.environment import TradingEnvironment
from trading_bot.utils import (
    load_prepared_data,
    get_stock_data,
    split_data,
    format_currency,
    format_position,
    show_train_result,
    switch_k_backend_device,
)


def main(data_path, window_size, batch_size, ep_count, patience=3,
         strategy="t-dqn", model_name="model_debug", pretrained=False,
         debug=False, train_csv=None, val_csv=None):
    """ Trains the stock trading bot using Deep Q-Learning.
    Please see https://arxiv.org/abs/1312.5602 for more details.

    Args: [python train.py --help]
    """
    # Déterminer la source de données: deux CSV explicites, dossier, ou un CSV unique
    import os

    if train_csv and val_csv:
        logging.info(f"Chargement du train CSV: {train_csv}")
        train_data = get_stock_data(train_csv)
        logging.info(f"Chargement du val CSV: {val_csv}")
        val_data = get_stock_data(val_csv)
        test_data = val_data
    else:
        if os.path.isdir(data_path):
            logging.info(f"Chargement des données pré-splittées depuis {data_path}")
            train_data, val_data, test_data = load_prepared_data(data_path)
        else:
            logging.info(f"Chargement du CSV brut puis split en mémoire depuis {data_path}")
            full_data = get_stock_data(data_path)
            train_data, val_data, test_data = split_data(full_data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    # Initialiser les environnements
    train_env = TradingEnvironment(train_data, window_size)
    val_env = TradingEnvironment(val_data, window_size)
    
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
    
    print("=" * 80)
    logging.info("🚀 Début de l'entraînement...")
    logging.info(f"  - Stratégie: {strategy}")
    logging.info(f"  - Episodes: {ep_count}")
    logging.info(f"  - Window size: {window_size}")
    logging.info(f"  - Batch size: {batch_size}")
    logging.info(f"  - Modèle: {model_name}")
    logging.info(f"  - Patience: {patience} épisodes")
    print("=" * 80)

    for episode in range(1, ep_count + 1):
        print(f"\n{'─' * 60}")
        logging.info(f"🎯 Episode {episode}/{ep_count}")
        
        train_result = train_model(agent, episode, train_env, ep_count=ep_count,
                                   batch_size=batch_size)
        val_result, _, _ = evaluate_model(agent, val_env, debug)
        show_train_result(train_result, val_result, initial_offset)
        
        # Status du meilleur modèle et early stopping
        print(f"{'─' * 40}")
        if val_result > best_val_profit:
            best_val_profit = val_result
            best_episode = episode
            patience_counter = 0
            
            # Sauvegarder le meilleur modèle
            agent.save_best()
            logging.info(f"🏆 Nouveau record ! Validation: ${best_val_profit:.2f}")
            logging.info(f"💾 Meilleur modèle sauvegardé")
        else:
            patience_counter += 1
            logging.info(f"⏳ Pas d'amélioration depuis {patience_counter}/{patience} épisode(s)")
            logging.info(f"🎯 Record actuel: ${best_val_profit:.2f} (Episode {best_episode})")
            
            # Early stopping si patience dépassée
            if patience_counter >= patience:
                print(f"\n{'═' * 60}")
                logging.info(f"🛑 Early stopping déclenché !")
                logging.info(f"🏆 Meilleur résultat: ${best_val_profit:.2f} (Episode {best_episode})")
                print(f"{'═' * 60}")
                break
        
        # Sauvegarde périodique (tous les 10 épisodes)
        # if episode % 10 == 0:
        #     agent.save(episode)
        #     logging.info(f"💾 Sauvegarde périodique: models/{model_name}_{episode}")
    
    print(f"\n{'═' * 80}")
    logging.info("✅ Entraînement terminé !")
    print(f"{'═' * 80}")
    logging.info(f"📊 RÉSUMÉ FINAL:")
    logging.info(f"  🏆 Meilleur résultat validation: ${best_val_profit:.2f} (Episode {best_episode})")
    logging.info(f"  💾 Meilleur modèle: models/{model_name}_best")
    logging.info(f"  📁 Modèles sauvegardés: models/{model_name}_*")
    logging.info(f"  📈 Total épisodes: {episode}/{ep_count}")
    print(f"{'═' * 80}")


if __name__ == "__main__":
    args = docopt(__doc__)

    data_path = args.get("<data-path>")
    strategy = args["--strategy"]
    window_size = int(args["--window-size"])
    batch_size = int(args["--batch-size"])
    ep_count = int(args["--episode-count"])
    model_name = args["--model-name"]
    patience = int(args["--patience"])
    pretrained = args["--pretrained"]
    debug = args["--debug"]
    cpu_force = args["--cpu"]

    coloredlogs.install(level="INFO", format='%(asctime)s %(levelname)s %(message)s')
    if cpu_force:
        switch_k_backend_device()

    try:
        main(data_path, window_size, batch_size, ep_count, patience,
             strategy=strategy, model_name=model_name, 
             pretrained=pretrained, debug=debug,
             train_csv=args.get("--train-csv"), val_csv=args.get("--val-csv"))
    except KeyboardInterrupt:
        print("Aborted!")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        # Optionally, re-raise or handle specific exceptions for better debugging
        # raise e
