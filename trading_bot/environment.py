import numpy as np

class TradingEnvironment:
    """
    A stock trading environment that simulates market interactions.
    Supports two modes:
    1. Simple mode (for training): Buys/sells single shares, reward is immediate profit.
    2. Portfolio mode (for backtesting): Manages a portfolio with a balance,
       reward is the change in total net worth.
    Also supports a time penalty for holding positions too long.
    """
    def __init__(self, data, window_size, initial_balance=None, trade_size=None, time_penalty=0.0):
        self.prices = data['prices']
        self.features = data['features']
        self.data_length = len(self.prices)
        self.window_size = window_size
        self.n_features = self.features.shape[1]
        self.state_size = (self.window_size - 1) + self.n_features

        # Portfolio simulation parameters
        self.initial_balance = initial_balance
        self.trade_size = trade_size
        self.is_portfolio_sim = initial_balance is not None

        # NEW: Time penalty parameter
        self.time_penalty = time_penalty

        # Episode state variables
        self.inventory = []
        self.shares_held = 0
        self.balance = 0
        self.net_worth = 0
        self.total_profit = 0
        self.current_step = 0
        # NEW: Track when a position was entered
        self.entry_step = 0

    def _get_state(self):
        if self.current_step < self.window_size - 1:
            return None
        start_idx = self.current_step - (self.window_size - 1)
        end_idx = self.current_step + 1
        price_window = self.prices[start_idx:end_idx]
        price_diffs = np.diff(price_window)
        current_features = self.features[self.current_step]
        state = np.concatenate((price_diffs, current_features)).flatten()
        return state

    def reset(self):
        """Resets the environment for a new episode."""
        self.current_step = self.window_size - 1
        self.entry_step = 0
        if self.is_portfolio_sim:
            self.balance = self.initial_balance
            self.net_worth = self.initial_balance
            self.shares_held = 0
        else:
            self.inventory = []
            self.total_profit = 0
        return self._get_state()

    def step(self, action):
        """Executes one time step in the environment."""
        if self.is_portfolio_sim:
            return self._step_portfolio(action)
        else:
            return self._step_simple(action)

    def _apply_time_penalty(self, reward):
        """Applies a penalty if a position is held."""
        if self.time_penalty > 0 and self.entry_step > 0:
            duration = self.current_step - self.entry_step
            # Penalty is linear to the duration
            penalty = duration * self.time_penalty
            return reward - penalty
        return reward

    def _step_simple(self, action):
        current_price = self.prices[self.current_step]
        reward = 0
        
        position_open = bool(self.inventory)
        
        # BUY only if inventory is empty
        if action == 1 and not position_open:
            self.inventory.append(current_price)
            self.entry_step = self.current_step # Start tracking duration
        # SELL only if inventory is not empty
        elif action == 2 and position_open:
            bought_price = self.inventory.pop(0)
            reward = current_price - bought_price
            self.total_profit += reward
            self.entry_step = 0 # Reset duration tracking
        # HOLD
        elif action == 0 and position_open:
            reward = self._apply_time_penalty(reward)

        is_done = (self.current_step + 1) >= self.data_length
        self.current_step += 1
        
        next_state = self._get_state() if not is_done else np.zeros(self.state_size)
            
        info = {'total_profit': self.total_profit, 'inventory_size': len(self.inventory)}
        return next_state, reward, is_done, info

    def _step_portfolio(self, action):
        current_price = self.prices[self.current_step]
        previous_net_worth = self.net_worth
        reward = 0

        position_open = self.shares_held > 0

        # BUY only if no shares are held
        if action == 1 and not position_open and self.balance >= self.trade_size:
            shares_to_buy = self.trade_size / current_price
            self.shares_held += shares_to_buy
            self.balance -= self.trade_size
            self.entry_step = self.current_step # Start tracking duration
        # SELL only if shares are held
        elif action == 2 and position_open:
            self.balance += self.shares_held * current_price
            self.shares_held = 0
            self.entry_step = 0 # Reset duration tracking
        
        # Update net worth and calculate P&L-based reward
        self.net_worth = self.balance + (self.shares_held * current_price)
        reward = self.net_worth - previous_net_worth

        # If holding, apply the time penalty on top of the P&L change
        if action == 0 and position_open:
            reward = self._apply_time_penalty(reward)
        
        is_done = (self.current_step + 1) >= self.data_length
        if is_done and self.shares_held > 0:
            self.balance += self.shares_held * current_price
            self.shares_held = 0
            self.net_worth = self.balance
            
        self.current_step += 1
        next_state = self._get_state() if not is_done else np.zeros(self.state_size)
            
        total_profit = self.net_worth - self.initial_balance
        info = {'total_profit': total_profit, 'net_worth': self.net_worth, 'shares_held': self.shares_held}
        return next_state, reward, is_done, info
