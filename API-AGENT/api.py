import os
import sys
import logging
from datetime import datetime, timedelta
import math
import time

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from trading_bot.agent import Agent

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION & CONSTANTS ---
# Load environment variables from a .env file for local development
load_dotenv() 

# These are now loaded from environment variables for security
PARTNER_ID = os.getenv("PARTNER_ID")
API_KEY = os.getenv("API_KEY")

# Check if the environment variables are set
if not PARTNER_ID or not API_KEY:
    raise ValueError("Les variables d'environnement PARTNER_ID et API_KEY doivent être définies.")

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Agents")

# Model-specific constants
WINDOW_SIZE = 10  # This must match the window size used during training
FEATURE_COLS = ['FGIndex', 'rsi', 'adx', 'standard_deviation', 'sma50', 'five_day_percentage']
ACTION_MAP = {0: "HOLD", 1: "BUY", 2: "SELL"}

# --- DATA COLLECTION CLASS (as provided by user) ---
class FearGreedIndexCollector:
    def __init__(self, partner_id, api_key):
        self.partner_id = partner_id
        self.api_key = api_key
        self.base_url = "https://f8kz2nqvwm3tdx9rlb.vercel.app/partner-data"
        
    def get_fear_greed_data_chunk(self, symbol, start_date, end_date, version="v7", max_retries=3):
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        url = f"{self.base_url}/{self.partner_id}/{symbol}"
        params = {"api_key": self.api_key, "version": version, "start_date": start_str, "end_date": end_str}
        headers = {"Content-Type": "application/json"}
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, headers=headers, timeout=60)
                if response.status_code == 429:
                    wait_time = 60 + (attempt * 30)
                    logging.warning(f"Rate limit hit. Waiting {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                response.raise_for_status()
                data = response.json()
                if not data.get("success"):
                    raise Exception(data.get('error', 'Unknown API error'))
                if not data.get("data", {}).get("scores"):
                    return None
                
                scores_data = data["data"]["scores"]
                df_data = []
                for score in scores_data:
                    row_data = {'date': score['date'], 'FGIndex': score['value'], 'sentiment': score['sentiment']}
                    if 'raw_data' in score and score['raw_data']:
                        raw_data = score['raw_data']
                        row_data.update({
                            'price': raw_data.get('price'), 'rsi': raw_data.get('rsi'), 'adx': raw_data.get('adx'),
                            'standard_deviation': raw_data.get('standard_deviation'), 'sma50': raw_data.get('sma50'),
                            'one_day_percentage': raw_data.get('one_day_percentage'),
                            'five_day_percentage': raw_data.get('five_day_percentage')
                        })
                    df_data.append(row_data)
                
                df = pd.DataFrame(df_data)
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date').reset_index(drop=True)
                return df
            except requests.exceptions.RequestException as e:
                logging.error(f"Request error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1: time.sleep(15)
                else: return None
            except Exception as e:
                logging.error(f"An error occurred on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1: time.sleep(10)
                else: return None
        return None

    def get_fear_greed_data(self, symbol="MA", days=30, version="v7", chunk_size=30):
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=days)
        
        logging.info(f"Fetching data for {symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        df = self.get_fear_greed_data_chunk(symbol, start_date, end_date, version)
        if df is not None and not df.empty:
            logging.info(f"Successfully fetched {len(df)} records.")
            return df
        else:
            logging.warning("Failed to fetch data or no data available for the period.")
            return None

# --- STATE PREPARATION ---
def prepare_state_for_prediction(df, window_size, feature_cols):
    """Prepares the most recent state from a dataframe for prediction."""
    if df is None or len(df) < window_size:
        logging.error(f"Not enough data to create a state. Need {window_size} rows, but got {len(df) if df is not None else 0}.")
        return None, 0

    # Ensure all required feature columns are present and drop NaNs
    required_cols = ['price'] + feature_cols
    df.dropna(subset=required_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)

    if len(df) < window_size:
        logging.error(f"Not enough data after dropping NaNs. Need {window_size}, have {len(df)}.")
        return None, 0

    # Get the last `window_size` data points
    latest_data = df.tail(window_size)
    
    # 1. Price history (price differences)
    price_window = latest_data['price'].to_numpy()
    price_diffs = np.diff(price_window)
    
    # 2. Current features (from the very last day)
    current_features = latest_data[feature_cols].iloc[-1].to_numpy()
    
    # 3. Combine them to form the state
    state = np.concatenate((price_diffs, current_features)).flatten()
    state = np.reshape(state, (1, -1)) # Reshape for the model
    
    n_features = len(feature_cols)
    return state, n_features

# --- FASTAPI APP ---
app = FastAPI(
    title="Trading Bot Inference API",
    description="API to get real-time trading predictions from trained agents.",
    version="1.0.0"
)

@app.get("/predict/{symbol}", tags=["Inference"])
async def predict(symbol: str):
    """
    Performs inference for a given stock symbol using the corresponding trained agent.
    
    - **symbol**: The stock ticker symbol (e.g., 'MA', 'GOOGL').
    """
    logging.info(f"Received prediction request for symbol: {symbol.upper()}")
    
    # 1. Validate model existence
    model_name = f"model_{symbol.lower()}_colab_best.keras"
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.isfile(model_path):
        logging.error(f"Model file not found at: {model_path}")
        raise HTTPException(
            status_code=404,
            detail=f"Model for symbol '{symbol.upper()}' not found. Available models are in the {MODELS_DIR} directory."
        )

    # 2. Fetch latest data
    collector = FearGreedIndexCollector(PARTNER_ID, API_KEY)
    # Fetch more days to ensure we have enough after dropping NaNs
    data_df = collector.get_fear_greed_data(symbol=symbol.upper(), days=60, chunk_size=60)
    
    if data_df is None:
        raise HTTPException(status_code=503, detail="Failed to fetch market data. The external API might be down.")

    # 3. Prepare state for inference
    state, n_features = prepare_state_for_prediction(data_df, WINDOW_SIZE, FEATURE_COLS)
    
    if state is None:
        raise HTTPException(
            status_code=400,
            detail=f"Could not prepare state for prediction. Not enough recent, valid data for symbol '{symbol.upper()}'."
        )

    try:
        # 4. Initialize agent and predict
        agent = Agent(WINDOW_SIZE, n_features, pretrained=True, model_name=model_path)
        
        # Use the new method to get action and Q-values
        action_idx, q_values = agent.predict_with_q_values(state)
        
        prediction = ACTION_MAP.get(action_idx, "UNKNOWN")
        
        logging.info(f"Prediction for {symbol.upper()}: {prediction} | Q-Values: {q_values}")

        # 5. Format and return response
        return {
            "symbol": symbol.upper(),
            "model_used": model_name,
            "prediction_timestamp_utc": datetime.utcnow().isoformat(),
            "latest_data_date": data_df['date'].iloc[-1].strftime('%Y-%m-%d'),
            "prediction": prediction,
            "q_values": {
                "hold": float(q_values[0]),
                "buy": float(q_values[1]),
                "sell": float(q_values[2])
            }
        }
    except FileNotFoundError as e:
        logging.error(f"File not found during agent loading: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logging.error(f"An unexpected error occurred during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint providing basic information about the API."""
    return {"message": "Welcome to the Trading Bot API. Use the /docs endpoint for documentation."}

if __name__ == "__main__":
    import uvicorn
    # To run: uvicorn api:app --reload --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000) 