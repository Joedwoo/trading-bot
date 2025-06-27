"""
Script for preparing and splitting dataset for Stock Trading Bot.

Usage:
  prepare_data.py <data-file> [--train-ratio=<train-ratio>] [--val-ratio=<val-ratio>] [--test-ratio=<test-ratio>] [--output-dir=<output-dir>]

Options:
  --train-ratio=<train-ratio>   Ratio for training data [default: 0.7]
  --val-ratio=<val-ratio>       Ratio for validation data [default: 0.15]
  --test-ratio=<test-ratio>     Ratio for test data [default: 0.15]
  --output-dir=<output-dir>     Output directory for split data [default: data/split]
"""

import os
import logging
import pandas as pd
import numpy as np
import coloredlogs
from docopt import docopt

from trading_bot.utils import get_stock_data, split_data


def save_split_data(train_data, val_data, test_data, output_dir):
    """Sauvegarde les données splittées dans des fichiers séparés"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarder les données d'entraînement
    train_df = pd.DataFrame(train_data['features'], columns=['FGIndex', 'rsi', 'adx', 'standard_deviation', 'sma50', 'five_day_percentage'])
    train_df['price'] = train_data['prices']
    if 'dates' in train_data:
        train_df.insert(0, 'date', train_data['dates'])  # Insérer date en première colonne
    train_df.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
    
    # Sauvegarder les données de validation
    val_df = pd.DataFrame(val_data['features'], columns=['FGIndex', 'rsi', 'adx', 'standard_deviation', 'sma50', 'five_day_percentage'])
    val_df['price'] = val_data['prices']
    if 'dates' in val_data:
        val_df.insert(0, 'date', val_data['dates'])  # Insérer date en première colonne
    val_df.to_csv(os.path.join(output_dir, 'val_data.csv'), index=False)
    
    # Sauvegarder les données de test
    test_df = pd.DataFrame(test_data['features'], columns=['FGIndex', 'rsi', 'adx', 'standard_deviation', 'sma50', 'five_day_percentage'])
    test_df['price'] = test_data['prices']
    if 'dates' in test_data:
        test_df.insert(0, 'date', test_data['dates'])  # Insérer date en première colonne
    test_df.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)
    
    # Sauvegarder un fichier de métadonnées
    with open(os.path.join(output_dir, 'split_info.txt'), 'w') as f:
        f.write(f"Dataset split information\n")
        f.write(f"========================\n")
        f.write(f"Total samples: {len(train_data['prices']) + len(val_data['prices']) + len(test_data['prices'])}\n")
        f.write(f"Train samples: {len(train_data['prices'])} ({len(train_data['prices'])/(len(train_data['prices']) + len(val_data['prices']) + len(test_data['prices'])):.1%})\n")
        f.write(f"Val samples: {len(val_data['prices'])} ({len(val_data['prices'])/(len(train_data['prices']) + len(val_data['prices']) + len(test_data['prices'])):.1%})\n") 
        f.write(f"Test samples: {len(test_data['prices'])} ({len(test_data['prices'])/(len(train_data['prices']) + len(val_data['prices']) + len(test_data['prices'])):.1%})\n")
        f.write(f"Number of features: {train_data['features'].shape[1]}\n")
        f.write(f"Feature names: {['FGIndex', 'rsi', 'adx', 'standard_deviation', 'sma50', 'five_day_percentage']}\n")
        if 'dates' in train_data:
            f.write(f"Date range: {train_data['dates'][0]} to {test_data['dates'][-1]}\n")
            f.write(f"Train dates: {train_data['dates'][0]} to {train_data['dates'][-1]}\n")
            f.write(f"Val dates: {val_data['dates'][0]} to {val_data['dates'][-1]}\n")
            f.write(f"Test dates: {test_data['dates'][0]} to {test_data['dates'][-1]}\n")


def show_data_statistics(data, dataset_name):
    """Affiche les statistiques des données"""
    logging.info(f"\n=== Statistiques {dataset_name} ===")
    logging.info(f"Nombre d'échantillons: {len(data['prices'])}")
    
    # Afficher les dates si disponibles
    if 'dates' in data:
        logging.info(f"Période: {data['dates'][0]} à {data['dates'][-1]}")
    
    logging.info(f"Prix - Min: ${data['prices'].min():.2f}, Max: ${data['prices'].max():.2f}, Moyenne: ${data['prices'].mean():.2f}")
    
    feature_names = ['FGIndex', 'rsi', 'adx', 'standard_deviation', 'sma50', 'five_day_percentage']
    for i, name in enumerate(feature_names):
        if i < data['features'].shape[1]:
            feature_data = data['features'][:, i]
            logging.info(f"{name} - Min: {feature_data.min():.2f}, Max: {feature_data.max():.2f}, Moyenne: {feature_data.mean():.2f}")


def main(data_file, train_ratio, val_ratio, test_ratio, output_dir):
    """Prépare et split les données"""
    
    # Vérifier que les ratios sont valides
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Les ratios doivent sommer à 1.0. Actuel: {train_ratio + val_ratio + test_ratio}")
    
    logging.info(f"Chargement des données depuis: {data_file}")
    
    # Charger les données
    full_data = get_stock_data(data_file)
    logging.info(f"Données chargées: {len(full_data['prices'])} échantillons, {full_data['features'].shape[1]} features")
    
    # Afficher les statistiques des données complètes
    show_data_statistics(full_data, "Dataset complet")
    
    # Splitter les données
    logging.info(f"Split des données avec ratios: Train={train_ratio}, Val={val_ratio}, Test={test_ratio}")
    train_data, val_data, test_data = split_data(full_data, train_ratio, val_ratio, test_ratio)
    
    # Afficher les statistiques de chaque split
    show_data_statistics(train_data, "Train")
    show_data_statistics(val_data, "Validation") 
    show_data_statistics(test_data, "Test")
    
    # Sauvegarder les données splittées
    logging.info(f"Sauvegarde des données dans: {output_dir}")
    save_split_data(train_data, val_data, test_data, output_dir)
    
    logging.info(f"✅ Préparation terminée ! Fichiers sauvegardés dans {output_dir}/")
    logging.info(f"   - train_data.csv: {len(train_data['prices'])} échantillons")
    logging.info(f"   - val_data.csv: {len(val_data['prices'])} échantillons")
    logging.info(f"   - test_data.csv: {len(test_data['prices'])} échantillons")
    logging.info(f"   - split_info.txt: Métadonnées du split")


if __name__ == "__main__":
    args = docopt(__doc__)
    
    data_file = args["<data-file>"]
    train_ratio = float(args["--train-ratio"])
    val_ratio = float(args["--val-ratio"])
    test_ratio = float(args["--test-ratio"])
    output_dir = args["--output-dir"]
    
    coloredlogs.install(level="DEBUG")
    
    try:
        main(data_file, train_ratio, val_ratio, test_ratio, output_dir)
    except KeyboardInterrupt:
        print("Aborted!")
    except Exception as e:
        logging.error(f"Erreur: {e}") 