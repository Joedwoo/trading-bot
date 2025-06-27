import os
import math
import logging

import pandas as pd
import numpy as np

import keras.backend as K


# Formats Position
format_position = lambda price: ('-$' if price < 0 else '+$') + '{0:.2f}'.format(abs(price))


# Formats Currency
format_currency = lambda price: '${0:.2f}'.format(abs(price))


def show_train_result(result, val_position, initial_offset):
    """ Displays training results
    """
    if val_position == initial_offset or val_position == 0.0:
        logging.info('Episode {}/{} - Train Position: {}  Val Position: USELESS  Train Loss: {:.4f}'
                     .format(result[0], result[1], format_position(result[2]), result[3]))
    else:
        logging.info('Episode {}/{} - Train Position: {}  Val Position: {}  Train Loss: {:.4f})'
                     .format(result[0], result[1], format_position(result[2]), format_position(val_position), result[3],))


def show_eval_result(model_name, profit, initial_offset):
    """ Displays eval results
    """
    if profit == initial_offset or profit == 0.0:
        logging.info('{}: USELESS\n'.format(model_name))
    else:
        logging.info('{}: {}\n'.format(model_name, format_position(profit)))


def get_stock_data(stock_file):
    """Reads stock data from csv file with multiple features
    Returns: dict with 'features' (all features except price/one_day_percentage), 'prices' (price column), and 'dates'
    """
    df = pd.read_csv(stock_file)
    
    # Features à utiliser (toutes sauf date, price et one_day_percentage)
    feature_columns = ['FGIndex', 'rsi', 'adx', 'standard_deviation', 'sma50', 'five_day_percentage']
    
    # Vérifier que toutes les colonnes existent
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        logging.warning(f"Colonnes manquantes: {missing_cols}")
        feature_columns = [col for col in feature_columns if col in df.columns]
    
    features = df[feature_columns].values
    prices = df['price'].values
    
    # Récupérer les dates si elles existent
    dates = None
    if 'date' in df.columns:
        dates = df['date'].values
    
    result = {
        'features': features,
        'prices': prices
    }
    
    if dates is not None:
        result['dates'] = dates
    
    return result


def load_prepared_data(data_dir):
    """Charge les données pré-splittées depuis un répertoire
    Returns: train_data, val_data, test_data (chacun étant un dict avec 'features' et 'prices')
    """
    train_file = os.path.join(data_dir, 'train_data.csv')
    val_file = os.path.join(data_dir, 'val_data.csv') 
    test_file = os.path.join(data_dir, 'test_data.csv')
    
    # Vérifier que tous les fichiers existent
    for file_path in [train_file, val_file, test_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fichier manquant: {file_path}. Exécutez d'abord prepare_data.py")
    
    # Charger chaque dataset
    train_data = get_stock_data(train_file)
    val_data = get_stock_data(val_file)
    test_data = get_stock_data(test_file)
    
    logging.info(f"Données chargées depuis {data_dir}:")
    logging.info(f"  - Train: {len(train_data['prices'])} échantillons")
    logging.info(f"  - Val: {len(val_data['prices'])} échantillons") 
    logging.info(f"  - Test: {len(test_data['prices'])} échantillons")
    logging.info(f"  - Features: {train_data['features'].shape[1]}")
    
    return train_data, val_data, test_data


def split_data(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split data into train, validation and test sets
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Les ratios doivent sommer à 1.0")
    
    total_len = len(data['prices'])
    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)
    
    train_data = {
        'features': data['features'][:train_len],
        'prices': data['prices'][:train_len]
    }
    
    val_data = {
        'features': data['features'][train_len:train_len + val_len],
        'prices': data['prices'][train_len:train_len + val_len]
    }
    
    test_data = {
        'features': data['features'][train_len + val_len:],
        'prices': data['prices'][train_len + val_len:]
    }
    
    # Ajouter les dates si elles existent
    if 'dates' in data:
        train_data['dates'] = data['dates'][:train_len]
        val_data['dates'] = data['dates'][train_len:train_len + val_len]
        test_data['dates'] = data['dates'][train_len + val_len:]
    
    logging.info(f"Split data: Train={len(train_data['prices'])}, Val={len(val_data['prices'])}, Test={len(test_data['prices'])}")
    
    return train_data, val_data, test_data


def switch_k_backend_device():
    """ Switches `keras` backend from GPU to CPU if required.

    Faster computation on CPU (if using tensorflow-gpu).
    """
    if K.backend() == "tensorflow":
        logging.debug("switching to TensorFlow for CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
