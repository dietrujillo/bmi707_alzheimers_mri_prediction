import argparse
import datetime

from preprocessing import preprocess
from training import train
from inference import run_inference


def parse_args(args: list[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alzheimer's state prediction with Convolutional Neural Networks.")

    # TODO: add more arguments as we need them.

    parser.add_argument("--mode", type=str, default="train", choices=["preprocess", "train", "inference"])
    parser.add_argument("--random-seed", type=int, default=42)

    parser.add_argument("--data-path", type=str, default="data", help="Path to the location of the data files.")
    parser.add_argument("--val-data-path", type=str, default="data",
                        help="Path to the location of the validation data files.")

    parser.add_argument("--crop-min", type=list, default=[0, 0, 0], help="Minimum index to crop images", nargs=3)
    parser.add_argument("--crop-max", type=list, default=[256, 256, 198], help="Maximum index to crop images", nargs="+")


    parser.add_argument("--training-id", type=str, default=None,
                        help="ID for the training run. Ignored if mode != 'train'. "
                             "If not specified, the ID will be generated with the system date.")

    namespace = parser.parse_args(args)

    if namespace.training_id is None:
        namespace.training_id = f"training_run_{datetime.datetime.now().isoformat()}"

    return namespace

if __name__ == '__main__':
    namespace = parse_args()
    if namespace.mode == "preprocess":
        preprocess(namespace)
    elif namespace.mode == "train":
        train(namespace)
    elif namespace.mode == "inference":
        run_inference(namespace)
