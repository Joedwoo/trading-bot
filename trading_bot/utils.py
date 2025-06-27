import os
import logging
import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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
    """Loads data from a CSV file and calculates features."""
    logging.info(f"Chargement et préparation des données depuis {csv_file}...")
    data = pd.read_csv(csv_file)
    data.sort_values('date', inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Assurer que les colonnes nécessaires sont présentes
    required_cols = {'date', 'price', 'FGIndex', 'rsi', 'adx', 'standard_deviation', 'sma50', 'five_day_percentage'}
    if not required_cols.issubset(data.columns):
        missing = required_cols - set(data.columns)
        raise ValueError(f"Colonnes manquantes dans le fichier CSV: {missing}")

    # Supprimer les lignes avec des NaN (générés par les indicateurs)
    data.dropna(inplace=True)

    # Récupérer les dates, prix et features
    dates = data['date'].to_numpy()
    prices = data['price'].to_numpy()
    
    feature_cols = ['FGIndex', 'rsi', 'adx', 'standard_deviation', 'sma50', 'five_day_percentage']
    features = data[feature_cols].to_numpy()
    
    # Normalisation des features
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features)

    return {'dates': dates, 'prices': prices, 'features': features_scaled}


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
