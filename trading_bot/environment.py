import numpy as np

class TradingEnvironment:
    """
    A stock trading environment that simulates market interactions step-by-step.

    This environment generates states on-the-fly to prevent look-ahead bias.
    It manages the agent's portfolio, calculates rewards, and tracks performance.
    """
    def __init__(self, data, window_size):
        """
        Initialize the trading environment.

        Args:
            data (dict): A dictionary containing 'prices', 'features'.
                         'prices' is a 1D numpy array of stock prices.
                         'features' is a 2D numpy array of technical indicators.
            window_size (int): The number of past days of data to include in the state.
        """
        self.prices = data['prices']
        self.features = data['features']
        self.data_length = len(self.prices)
        self.window_size = window_size
        self.n_features = self.features.shape[1]
        
        # The state size is the number of price differences + number of features
        self.state_size = (self.window_size - 1) + self.n_features

        # Episode state
        self.inventory = []
        self.total_profit = 0
        self.current_step = 0
        self.max_open_positions = 4

    def _get_state(self):
        """
        Constructs and returns the state for the current time step.
        The state is composed of two parts:
        1. Price history: `window_size - 1` consecutive price differences.
        2. Current features: Technical indicators for the current day.

        Returns:
            np.ndarray: The current state vector, or None if there's not enough data.
        """
        # We need at least `window_size` data points to create one state.
        if self.current_step < self.window_size - 1:
            return None

        # 1. Price history (price differences)
        start_idx = self.current_step - (self.window_size - 1)
        end_idx = self.current_step + 1
        price_window = self.prices[start_idx:end_idx]
        price_diffs = np.diff(price_window)
        
        # 2. Current features
        current_features = self.features[self.current_step]
        
        # Combine them to form the state
        state = np.concatenate((price_diffs, current_features)).flatten()
        return state

    def reset(self):
        """
        Resets the environment to its initial state for a new episode.

        Returns:
            np.ndarray: The initial state of the environment.
        """
        self.inventory = []
        self.total_profit = 0
        # Start at the first point where a full state can be formed
        self.current_step = self.window_size - 1
        return self._get_state()

    def step(self, action):
        """
        Executes one time step in the environment based on the agent's action.

        Args:
            action (int): The action to take (0: HOLD, 1: BUY, 2: SELL).

        Returns:
            tuple: A tuple containing:
                - next_state (np.ndarray): The state for the next time step.
                - reward (float): The reward for the current action.
                - done (bool): True if the episode has ended, False otherwise.
                - info (dict): A dictionary with auxiliary information (e.g., profit).
        """
        # An episode is done if the next step is out of bounds
        is_done = (self.current_step + 1) >= self.data_length
        
        current_price = self.prices[self.current_step]
        reward = 0
        
        # Define a small penalty for holding
        holding_penalty = 0.0001 # 0.05% per step

        # --- Execute action ---
        # BUY
        if action == 1:
            # Empêcher d'ouvrir plus que max_open_positions
            if len(self.inventory) < self.max_open_positions:
                self.inventory.append(current_price)
            else:
                # Légère pénalité pour tentative d'achat au-delà de la limite
                reward = -abs(current_price) * 0.00005

        # SELL
        elif action == 2 and len(self.inventory) > 0:
            bought_price = self.inventory.pop(0)
            pnl = current_price - bought_price
            # Asymmetric reward focused on winrate
            if pnl > 0:
                reward = 1.0
            elif pnl < 0:
                reward = -1.5
            else:
                reward = 0.0
            # Still accumulate true PnL for reporting
            self.total_profit += pnl
            
        # HOLD
        elif action == 0 and len(self.inventory) > 0:
            # Penalize for holding a position
            # This encourages the agent to close positions instead of holding indefinitely
            bought_price = self.inventory[0] # Get the price of the oldest share
            reward = - (bought_price * holding_penalty)

        # --- Move to the next time step ---
        self.current_step += 1

        # --- Get the next state ---
        if is_done:
            next_state = np.zeros(self.state_size)
        else:
            next_state = self._get_state()

        # --- Auxiliary info ---
        info = {
            'total_profit': self.total_profit,
            'inventory_size': len(self.inventory),
            'current_price': current_price
        }

        return next_state, reward, is_done, info 