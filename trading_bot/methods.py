import os
import logging

import numpy as np

from tqdm import tqdm

from .utils import (
    format_currency,
    format_position
)


def train_model(agent, episode, env, ep_count=100, batch_size=32):
    state = env.reset()
    # The agent's own inventory should be synced with the env's
    agent.inventory = env.inventory
    avg_loss = []
    
    # Action counters
    buy_count, sell_count, hold_count = 0, 0, 0

    # Use a while loop to iterate through the environment steps
    progress_bar = tqdm(total=env.data_length - env.window_size, desc=f'📈 Episode {episode:2d}/{ep_count}', leave=False, ncols=100)

    while True:
        action = agent.act(state.reshape(1, -1))

        # Count actions
        if action == 1:
            buy_count += 1
        elif action == 2 and len(agent.inventory) > 0:
            sell_count += 1
        else:
            hold_count += 1

        next_state, reward, done, info = env.step(action)
        
        # Sync agent inventory since env is the source of truth
        agent.inventory = env.inventory

        agent.remember(state.reshape(1, -1), action, reward, next_state.reshape(1, -1), done)

        if len(agent.memory) > batch_size:
            loss = agent.train_experience_replay(batch_size, episode)
            avg_loss.append(loss)

        state = next_state
        progress_bar.update(1)
        
        progress_bar.set_postfix({
            'Profit': f"{info['total_profit']:.2f}",
            'ε': f'{agent.epsilon:.3f}',
            'Inv': info['inventory_size']
        })

        if done:
            break

    progress_bar.close()
    
    # Final info is retrieved from the last step
    total_profit = info['total_profit']
    avg_loss_val = np.mean(np.array(avg_loss)) if avg_loss else 0.0
    
    logging.info(f"📊 Episode {episode:2d}: Profit={total_profit:8.2f} | "
                f"Loss={avg_loss_val:.4f} | Actions: BUY={buy_count:3d} SELL={sell_count:3d} HOLD={hold_count:3d} | "
                f"ε={agent.epsilon:.3f} | Inventory={info['inventory_size']}")

    # if episode % 10 == 0:
    #     agent.save(episode)

    return (episode, ep_count, total_profit, avg_loss_val)


def evaluate_model(agent, env, debug):
    state = env.reset()
    agent.inventory = env.inventory  # Sync inventory
    history = []
    cumulative_profits = []

    while True:
        action = agent.act(state.reshape(1, -1), is_eval=True)

        current_price = env.prices[env.current_step]

        # Log action before stepping
        if action == 1:  # BUY
            history.append((current_price, "BUY"))
            if debug:
                logging.debug(f"Buy at: {format_currency(current_price)}")
        elif action == 2 and len(agent.inventory) > 0:  # SELL
            history.append((current_price, "SELL"))
            if debug:
                logging.debug(f"Sell at: {format_currency(current_price)}")
        else:  # HOLD
            history.append((current_price, "HOLD"))

        next_state, reward, done, info = env.step(action)
        
        # Log profit after a sell action
        if reward != 0 and debug:
            logging.debug("Position Closed. Profit: {}".format(format_position(reward)))

        state = next_state
        agent.inventory = env.inventory
        cumulative_profits.append(info['total_profit'])

        if done:
            return info['total_profit'], history, cumulative_profits
