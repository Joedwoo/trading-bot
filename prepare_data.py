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


def save_split_data(train_data, val_data, test_data, output_dir, feature_names):
    """Sauvegarde les données splittées dans des fichiers séparés"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarder les données d'entraînement
    train_df = pd.DataFrame(train_data['features'], columns=feature_names)
    train_df['price'] = train_data['prices']
    if 'dates' in train_data:
        train_df.insert(0, 'date', train_data['dates'])  # Insérer date en première colonne
    train_df.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
    
    # Sauvegarder les données de validation
    val_df = pd.DataFrame(val_data['features'], columns=feature_names)
    val_df['price'] = val_data['prices']
    if 'dates' in val_data:
        val_df.insert(0, 'date', val_data['dates'])  # Insérer date en première colonne
    val_df.to_csv(os.path.join(output_dir, 'val_data.csv'), index=False)
    
    # Sauvegarder les données de test
    test_df = pd.DataFrame(test_data['features'], columns=feature_names)
    test_df['price'] = test_data['prices']
    if 'dates' in test_data:
        test_df.insert(0, 'date', test_data['dates'])  # Insérer date en première colonne
    test_df.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)
    
    # Sauvegarder un fichier de métadonnées
    with open(os.path.join(output_dir, 'split_info.txt'), 'w') as f:
        f.write(f"Dataset split information\n")
        f.write(f"========================\n")
        total = len(train_data['prices']) + len(val_data['prices']) + len(test_data['prices'])
        f.write(f"Total samples: {total}\n")
        f.write(f"Train samples: {len(train_data['prices'])} ({len(train_data['prices'])/total:.1%})\n")
        f.write(f"Val samples: {len(val_data['prices'])} ({len(val_data['prices'])/total:.1%})\n") 
        f.write(f"Test samples: {len(test_data['prices'])} ({len(test_data['prices'])/total:.1%})\n")
        f.write(f"Number of features: {len(feature_names)}\n")
        f.write(f"Feature names: {feature_names}\n")
        if 'dates' in train_data:
            f.write(f"Date range: {train_data['dates'][0]} to {test_data['dates'][-1]}\n")
            f.write(f"Train dates: {train_data['dates'][0]} to {train_data['dates'][-1]}\n")
            f.write(f"Val dates: {val_data['dates'][0]} to {val_data['dates'][-1]}\n")
            f.write(f"Test dates: {test_data['dates'][0]} to {test_data['dates'][-1]}\n")


def show_data_statistics(data, dataset_name, feature_names):
    """Affiche les statistiques des données"""
    logging.info(f"\n=== Statistiques {dataset_name} ===")
    logging.info(f"Nombre d'échantillons: {len(data['prices'])}")
    
    # Afficher les dates si disponibles
    if 'dates' in data:
        logging.info(f"Période: {data['dates'][0]} à {data['dates'][-1]}")
    
    logging.info(f"Prix - Min: ${data['prices'].min():.2f}, Max: ${data['prices'].max():.2f}, Moyenne: ${data['prices'].mean():.2f}")
    
    # Afficher un résumé sur les features
    num_feats = data['features'].shape[1]
    logging.info(f"Nombre de features: {num_feats}")
    # Afficher quelques stats pour les 10 premières features (pour ne pas surcharger les logs)
    for i in range(min(num_feats, 10)):
        name = feature_names[i] if i < len(feature_names) else f'feat_{i}'
        col = data['features'][:, i]
        logging.info(f"{name} - Min: {col.min():.4f}, Max: {col.max():.4f}, Moyenne: {col.mean():.4f}")


def main(data_file, train_ratio, val_ratio, test_ratio, output_dir):
    """Prépare et split les données"""
    
    # Vérifier que les ratios sont valides
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Les ratios doivent sommer à 1.0. Actuel: {train_ratio + val_ratio + test_ratio}")
    
    logging.info(f"Chargement des données depuis: {data_file}")
    
    # Charger les données avec la logique utilitaire
    full_data = get_stock_data(data_file)
    logging.info(f"Données chargées: {len(full_data['prices'])} échantillons, {full_data['features'].shape[1]} features")

    # Recalculer les noms de features à partir du CSV brut pour conserver les labels d'origine
    raw_df = pd.read_csv(data_file)
    # Détection des colonnes de date et de prix (alignée avec utils.get_stock_data)
    date_col = 'date' if 'date' in raw_df.columns else ('timestamp' if 'timestamp' in raw_df.columns else None)
    price_col = 'price' if 'price' in raw_df.columns else ('close' if 'close' in raw_df.columns else None)
    if date_col is None or price_col is None:
        raise ValueError("Colonnes de date/prix introuvables. Requis: date|timestamp et price|close")
    excluded = {date_col, price_col, 'target', 'next_return'}
    feature_names = [c for c in raw_df.columns if c not in excluded]

    # Afficher les statistiques des données complètes
    show_data_statistics(full_data, "Dataset complet", feature_names)
    
    # Splitter les données
    logging.info(f"Split des données avec ratios: Train={train_ratio}, Val={val_ratio}, Test={test_ratio}")
    train_data, val_data, test_data = split_data(full_data, train_ratio, val_ratio, test_ratio)
    
    # Afficher les statistiques de chaque split
    show_data_statistics(train_data, "Train", feature_names)
    show_data_statistics(val_data, "Validation", feature_names)
    show_data_statistics(test_data, "Test", feature_names)
    
    # Sauvegarder les données splittées avec les noms de colonnes complets
    logging.info(f"Sauvegarde des données dans: {output_dir}")
    save_split_data(train_data, val_data, test_data, output_dir, feature_names)
    
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