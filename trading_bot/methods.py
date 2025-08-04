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
    avg_loss = []
    
    buy_count, sell_count, hold_count = 0, 0, 0

    progress_bar = tqdm(total=env.data_length - env.window_size, desc=f'📈 Episode {episode:2d}/{ep_count}', leave=False, ncols=100)

    while True:
        action = agent.act(state.reshape(1, -1))

        if action == 1:
            buy_count += 1
        elif action == 2 and (env.shares_held > 0 if env.is_portfolio_sim else len(env.inventory) > 0):
            sell_count += 1
        else:
            hold_count += 1

        next_state, reward, done, info = env.step(action)
        
        agent.remember(state.reshape(1, -1), action, reward, next_state.reshape(1, -1), done)

        if len(agent.memory) > batch_size:
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)

        state = next_state
        progress_bar.update(1)
        
        profit_display = info.get('net_worth', info.get('total_profit', 0))
        inv_display = info.get('shares_held', info.get('inventory_size', 0))

        progress_bar.set_postfix({
            'Profit': f"{profit_display:.2f}",
            'ε': f'{agent.epsilon:.3f}',
            'Inv': f"{inv_display:.2f}" if isinstance(inv_display, float) else inv_display
        })

        if done:
            break

    progress_bar.close()
    
    total_profit = info.get('total_profit', 0)
    avg_loss_val = np.mean(np.array(avg_loss)) if avg_loss else 0.0
    
    logging.info(f"📊 Episode {episode:2d}: Profit={total_profit:8.2f} | "
                f"Loss={avg_loss_val:.4f} | Actions: BUY={buy_count:3d} SELL={sell_count:3d} HOLD={hold_count:3d} | "
                f"ε={agent.epsilon:.3f}")

    return (episode, ep_count, total_profit, avg_loss_val)


def evaluate_model(agent, env, debug):
    state = env.reset()
    history = []
    cumulative_profits = []

    while True:
        action = agent.act(state.reshape(1, -1), is_eval=True)
        
        current_price = env.prices[env.current_step]
        
        # We need to get the reward from the step to log it
        next_state, reward, done, info = env.step(action)
        
        action_str = "HOLD"
        if action == 1:
            action_str = "BUY"
        elif action == 2 and (info.get('shares_held', info.get('inventory_size', 0)) > 0 or reward != 0): # Check if sell was effective
             action_str = "SELL"

        # The history now includes the profit/loss from that specific trade (reward)
        history.append((current_price, action_str, reward))
        
        if debug:
            if action_str == "BUY":
                logging.debug(f"Buy at: {format_currency(current_price)}")
            elif action_str == "SELL":
                logging.debug(f"Sell at: {format_currency(current_price)}, Profit: {format_position(reward)}")
        
        state = next_state
        
        # Use total_profit from info dict which is consistent for both modes
        current_total_profit = info.get('total_profit', 0)
        cumulative_profits.append(current_total_profit)

        if done:
            return current_total_profit, history, cumulative_profits
