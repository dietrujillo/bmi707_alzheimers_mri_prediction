import argparse
import datetime

from training import train
from inference import run_inference

from models.models import MODELS


def parse_args(args: list[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alzheimer's state prediction with Convolutional Neural Networks.")

    parser.add_argument("--mode", type=str, default="train", choices=["preprocess", "train", "inference"])
    parser.add_argument("--random-seed", type=int, default=42)
    
    parser.add_argument("--model-name", type=str, choices=MODELS.keys(), required=True,
                        help="Name of the model architecture to train.")

    parser.add_argument("--data-path", type=str, default="data", help="Path to the location of the data files.")
    parser.add_argument("--results-path", type=str, default="results", help="Path to results dir. Does not have to exist.")
    
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for.")
    parser.add_argument("--early-stopping-patience", type=int, default=5, help="Patience of the EarlyStopping callback.")
    

    parser.add_argument("--training-id", type=str, default=None,
                        help="ID for the training run. Ignored if mode != 'train'. "
                             "If not specified, the ID will be generated with the system date.")

    namespace = parser.parse_args(args)

    if namespace.training_id is None:
        namespace.training_id = f"training_run_{datetime.datetime.now().isoformat()}"

    return namespace

if __name__ == '__main__':
    namespace = parse_args()
    if namespace.mode == "train":
        train(namespace)
    elif namespace.mode == "inference":
        run_inference(namespace)
