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
    
    # Action counters (executed only)
    buy_count, sell_count, hold_count = 0, 0, 0
    win_trades = 0
    
    # Use a while loop to iterate through the environment steps
    progress_bar = tqdm(total=env.data_length - env.window_size, desc=f'📈 Episode {episode:2d}/{ep_count}', leave=False, ncols=100)

    while True:
        action = agent.act(state.reshape(1, -1))

        # Track inventory change to count only executed actions
        prev_inv = len(agent.inventory)

        next_state, reward, done, info = env.step(action)
        
        # Sync agent inventory since env is the source of truth
        agent.inventory = env.inventory

        # Count executed actions
        if action == 1 and len(agent.inventory) > prev_inv:
            buy_count += 1
        elif action == 2 and len(agent.inventory) < prev_inv:
            sell_count += 1
            if reward > 0:
                win_trades += 1
        else:
            hold_count += 1

        agent.remember(state.reshape(1, -1), action, reward, next_state.reshape(1, -1), done)

        if len(agent.memory) > batch_size:
            loss = agent.train_experience_replay(batch_size, episode)
            avg_loss.append(loss)

        state = next_state
        progress_bar.update(1)
        
        # Show progress (profit and epsilon)
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
    total_trades = sell_count  # completed trades are closes
    winrate = (win_trades / total_trades * 100.0) if total_trades > 0 else 0.0
    
    logging.info(f"📊 Episode {episode:2d}: Profit={total_profit:8.2f} | "
                f"Loss={avg_loss_val:.4f} | Actions EXEC: BUY={buy_count:3d} SELL={sell_count:3d} HOLD={hold_count:3d} | "
                f"Winrate={winrate:.2f}% ({win_trades}/{total_trades}) | "
                f"ε={agent.epsilon:.3f} | Inventory={info['inventory_size']}")

    if episode % 10 == 0:
        agent.save(episode)

    return (episode, ep_count, total_profit, avg_loss_val)


def evaluate_model(agent, env, debug):
    state = env.reset()
    agent.inventory = env.inventory  # Sync inventory
    history = []
    cumulative_profits = []

    # Trade counters for validation
    buy_exec, sell_exec = 0, 0
    win_exec = 0

    while True:
        action = agent.act(state.reshape(1, -1), is_eval=True)

        current_price = env.prices[env.current_step]

        # Log action before stepping
        if action == 1:  # BUY
            history.append((current_price, "BUY"))
        elif action == 2 and len(agent.inventory) > 0:  # SELL
            history.append((current_price, "SELL"))
        else:  # HOLD
            history.append((current_price, "HOLD"))

        prev_inv = len(agent.inventory)
        next_state, reward, done, info = env.step(action)
        
        # Count executed actions
        if action == 1 and len(env.inventory) > prev_inv:
            buy_exec += 1
        if action == 2 and len(env.inventory) < prev_inv:
            sell_exec += 1
            if reward > 0:
                win_exec += 1

        # Log profit after a sell action
        if reward != 0 and debug:
            logging.debug("Position Closed. Profit: {}".format(format_position(reward)))

        state = next_state
        agent.inventory = env.inventory
        cumulative_profits.append(info['total_profit'])

        if done:
            # Validation winrate logging
            val_winrate = (win_exec / sell_exec * 100.0) if sell_exec > 0 else 0.0
            logging.info(f"✅ Validation: Trades={sell_exec} | Winrate={val_winrate:.2f}% ({win_exec}/{sell_exec}) | Profit Réalisé={info['total_profit']:.2f}")
            return info['total_profit'], history, cumulative_profits
