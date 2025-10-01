import os
import logging
import math

import numpy as np
import pandas as pd

import keras.backend as K

from tqdm import tqdm
import coloredlogs

# Set Keras backend to TensorFlow
os.environ["KERAS_BACKEND"] = "tensorflow"


def format_currency(price):
    """Formats a price float into a currency string."""
    return "${:,.2f}".format(price)


def format_position(price):
    """Formats a price float into a currency string with sign."""
    return "${:+.2f}".format(price)


def show_train_result(result, val_position, initial_offset):
    """Displays training results."""
    if val_position == initial_offset:
        logging.info("Validation Position: USELESS")
    else:
        logging.info(f"Validation Position: {format_position(val_position)} | Profit: {format_currency(val_position - initial_offset)}")


def get_stock_price_change(stock_prices):
    """Calculates the change in stock prices."""
    return stock_prices.diff().dropna().to_numpy()


def get_stock_data(csv_file):
    """Charge les données d'un fichier CSV brut et construit le tenseur de features.

    - Détecte automatiquement les noms de colonnes pour la date ("date"/"timestamp")
      et le prix ("price"/"close").
    - Exclut explicitement les colonnes cibles ("target", "next_return") des features.
    - Trie par date croissante et supprime les lignes NaN.
    """
    logging.info(f"Chargement et préparation des données depuis {csv_file}...")
    data = pd.read_csv(csv_file)

    # Détection des colonnes de date et de prix
    date_col = 'date' if 'date' in data.columns else ('timestamp' if 'timestamp' in data.columns else None)
    price_col = 'price' if 'price' in data.columns else ('close' if 'close' in data.columns else None)
    if date_col is None or price_col is None:
        raise ValueError("Colonnes de date/prix introuvables. Requis: date|timestamp et price|close")

    # Normaliser la colonne de date en string triable
    # Laisser pandas gérer l'ordre lexicographique ISO8601 si déjà formatté
    data.sort_values(date_col, inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Supprimer les lignes avec des NaN (souvent dues aux indicateurs)
    data.dropna(inplace=True)

    # Extraire dates et prix
    dates = data[date_col].to_numpy()
    prices = data[price_col].to_numpy()

    # Construire la liste des features: toutes colonnes sauf date/prix et colonnes cibles connues
    excluded_cols = {date_col, price_col, 'target', 'next_return'}
    feature_cols = [col for col in data.columns if col not in excluded_cols]
    if len(feature_cols) == 0:
        raise ValueError("Aucune colonne de features après exclusion des colonnes date/prix/targets.")
    features = data[feature_cols].to_numpy()

    return {'dates': dates, 'prices': prices, 'features': features}


def split_data(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Splits data into training, validation, and test sets."""
    data_length = len(data['prices'])
    
    train_size = int(data_length * train_ratio)
    val_size = int(data_length * val_ratio)
    
    train_end = train_size
    val_end = train_size + val_size
    
    train_data = {key: val[:train_end] for key, val in data.items()}
    val_data = {key: val[train_end:val_end] for key, val in data.items()}
    test_data = {key: val[val_end:] for key, val in data.items()}
    
    return train_data, val_data, test_data


def load_prepared_data(data_dir):
    """Charge les données pré-splittées depuis le disque."""
    logging.info(f"Chargement des données depuis le dossier: {data_dir}")
    train_df = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'val_data.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))

    def extract_from_df(df):
        return {
            'dates': df['date'].to_numpy(),
            'prices': df['price'].to_numpy(),
            'features': df.drop(columns=['date', 'price']).to_numpy()
        }

    return extract_from_df(train_df), extract_from_df(val_df), extract_from_df(test_df)


def switch_k_backend_device():
    """Switches Keras backend to TensorFlow for CPU."""
    if K.backend() == "tensorflow":
        logging.debug("switching to TensorFlow for CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
