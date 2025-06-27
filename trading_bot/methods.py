import os
import logging

import numpy as np

from tqdm import tqdm

from .utils import (
    format_currency,
    format_position
)


def train_model(agent, episode, data, states, ep_count=100, batch_size=32, window_size=10):
    total_profit = 0
    prices = data['prices']  # Utiliser les prix pour le trading
    data_length = len(prices) - 1

    agent.inventory = []
    avg_loss = []
    
    # Compteurs pour les actions
    buy_count = 0
    sell_count = 0
    hold_count = 0

    # L'état initial est déjà calculé
    state = states[0]

    # Progress bar simplifiée qui se met à jour moins souvent
    progress_bar = tqdm(range(data_length), 
                       desc=f'📈 Episode {episode:2d}/{ep_count}', 
                       leave=False, 
                       ncols=100,
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    for t in progress_bar:        
        reward = 0
        # Le prochain état est simplement récupéré du tableau pré-calculé
        next_state = states[t + 1]

        # select an action
        action = agent.act(state.reshape(1, -1))

        # BUY
        if action == 1:
            agent.inventory.append(prices[t])
            buy_count += 1

        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            delta = prices[t] - bought_price
            reward = delta #max(delta, 0)
            total_profit += delta
            sell_count += 1

        # HOLD
        else:
            hold_count += 1

        done = (t == data_length - 1)
        # L'état doit avoir la bonne shape pour le replay buffer
        agent.remember(state.reshape(1, -1), action, reward, next_state.reshape(1, -1), done)

        if len(agent.memory) > batch_size:
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)

        state = next_state
        
        # Mise à jour de la description de la progress bar moins fréquemment
        if t % 50 == 0 or done:
            progress_bar.set_postfix({
                'Profit': f'{total_profit:.2f}',
                'ε': f'{agent.epsilon:.3f}',
                'Inv': len(agent.inventory)
            })

    progress_bar.close()
    
    # Calculer les métriques finales
    avg_loss_val = np.mean(np.array(avg_loss)) if avg_loss else 0.0
    
    # Log des résultats de l'épisode
    logging.info(f"📊 Episode {episode:2d}: Profit={total_profit:8.2f} | "
                f"Loss={avg_loss_val:.4f} | Actions: BUY={buy_count:3d} SELL={sell_count:3d} HOLD={hold_count:3d} | "
                f"ε={agent.epsilon:.3f} | Inventory={len(agent.inventory)}")

    if episode % 10 == 0:
        agent.save(episode)

    return (episode, ep_count, total_profit, avg_loss_val)


def evaluate_model(agent, data, states, window_size, debug):
    total_profit = 0
    prices = data['prices']  # Utiliser les prix pour le trading
    data_length = len(prices) - 1

    history = []
    agent.inventory = []
    
    # L'état initial est déjà calculé
    state = states[0]

    for t in range(data_length):        
        reward = 0
        # Le prochain état est simplement récupéré du tableau pré-calculé
        next_state = states[t + 1]
        
        # select an action
        action = agent.act(state.reshape(1, -1), is_eval=True)

        # BUY
        if action == 1:
            agent.inventory.append(prices[t])

            history.append((prices[t], "BUY"))
            if debug:
                logging.debug("Buy at: {}".format(format_currency(prices[t])))
        
        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            delta = prices[t] - bought_price
            reward = delta #max(delta, 0)
            total_profit += delta

            history.append((prices[t], "SELL"))
            if debug:
                logging.debug("Sell at: {} | Position: {}".format(
                    format_currency(prices[t]), format_position(prices[t] - bought_price)))
        # HOLD
        else:
            history.append((prices[t], "HOLD"))

        done = (t == data_length - 1)
        # L'état doit avoir la bonne shape pour le replay buffer
        agent.memory.append((state.reshape(1, -1), action, reward, next_state.reshape(1, -1), done))

        state = next_state
        if done:
            return total_profit, history
