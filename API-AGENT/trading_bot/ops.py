import os
import math
import logging

import numpy as np


def sigmoid(x):
    """Performs sigmoid operation
    """
    try:
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        return 1 / (1 + math.exp(-x))
    except Exception as err:
        print("Error in sigmoid: " + err)


def get_state(data, t, n_days):
    """Returns an n-day state representation ending at time t
    Adapté pour traiter les données multi-features
    """
    prices = data['prices']
    features = data['features']
    
    d = t - n_days + 1
    
    # Extraire la fenêtre de prix
    if d >= 0:
        price_block = prices[d: t + 1]
    else:
        # Corriger la multiplication: répéter la première valeur -d fois
        price_block = [prices[0]] * (-d) + list(prices[0: t + 1])
    
    # Calculer les différences de prix normalisées avec sigmoid
    price_diffs = []
    for i in range(n_days - 1):
        price_diffs.append(sigmoid(price_block[i + 1] - price_block[i]))
    
    # Extraire les features pour le timestep actuel
    if t < len(features):
        current_features = features[t]
        # Normaliser les features entre 0 et 1 (simple min-max scaling sur la fenêtre)
        feature_window = features[max(0, d):t + 1]
        normalized_features = []
        
        for i in range(len(current_features)):
            feature_col = feature_window[:, i]
            min_val, max_val = np.min(feature_col), np.max(feature_col)
            if max_val > min_val:
                normalized_val = (current_features[i] - min_val) / (max_val - min_val)
            else:
                normalized_val = 0.5  # valeur neutre si pas de variation
            normalized_features.append(normalized_val)
    else:
        # Si on dépasse les données, utiliser les dernières features connues
        normalized_features = [0.5] * features.shape[1]
    
    # Combiner les différences de prix et les features normalisées
    state = price_diffs + normalized_features
    
    return np.array([state])


def get_feature_size(data):
    """Retourne la taille totale de l'état (price_diffs + features)
    """
    n_price_features = 1  # pour les différences de prix (sera multiplié par window_size - 1)
    n_additional_features = data['features'].shape[1]  # nombre de features additionnelles
    return n_price_features, n_additional_features
