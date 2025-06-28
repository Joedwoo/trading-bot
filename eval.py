"""
Script for evaluating Stock Trading Bot.

Usage:
  eval.py <eval-stock> [--window-size=<window-size>] [--model-name=<model-name>] [--debug]

Options:
  --window-size=<window-size>   Size of the n-day window stock data representation used as the feature vector. [default: 10]
  --model-name=<model-name>     Name of the pretrained model to use (will eval all models in `models/` if unspecified).
  --debug                       Specifies whether to use verbose logs during eval operation.
"""

import os
import coloredlogs

from docopt import docopt

from trading_bot.agent import Agent
from trading_bot.methods import evaluate_model
from trading_bot.environment import TradingEnvironment
from trading_bot.utils import (
    get_stock_data,
    format_currency,
    format_position,
    switch_k_backend_device
)


def main(eval_stock, window_size, model_name, debug):
    """ Evaluates the stock trading bot.
    Please see https://arxiv.org/abs/1312.5602 for more details.

    Args: [python eval.py --help]
    """    
    data = get_stock_data(eval_stock)
    n_features = data['features'].shape[1]
    
    # Create the evaluation environment
    env = TradingEnvironment(data, window_size)

    # Single Model Evaluation
    if model_name is not None:
        agent = Agent(window_size, n_features, pretrained=True, model_name=model_name)
        profit, _ = evaluate_model(agent, env, debug)
        print(f"Evaluation Result for {model_name}:")
        print(f"  - Profit: {format_position(profit)}")
        
    # Multiple Model Evaluation
    else:
        for model_file in os.listdir("models"):
            if os.path.isfile(os.path.join("models", model_file)):
                model_path = os.path.splitext(model_file)[0]
                agent = Agent(window_size, n_features, pretrained=True, model_name=model_path)
                profit, _ = evaluate_model(agent, env, debug)
                print(f"Evaluation Result for {model_file}:")
                print(f"  - Profit: {format_position(profit)}")
                del agent


if __name__ == "__main__":
    args = docopt(__doc__)

    eval_stock = args["<eval-stock>"]
    window_size = int(args["--window-size"])
    model_name = args["--model-name"]
    debug = args["--debug"]

    coloredlogs.install(level="DEBUG")
    switch_k_backend_device()

    try:
        main(eval_stock, window_size, model_name, debug)
    except KeyboardInterrupt:
        print("Aborted")
