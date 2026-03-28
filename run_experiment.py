import argparse
import tensorflow as tf

from config import MODEL_NAME
from train import run_all_folds, train_one_fold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL_NAME, choices=["bilstm", "transformer"])
    parser.add_argument("--mode", type=str, default="all", choices=["all", "one"])
    parser.add_argument("--fold", type=int, default=1)

    args = parser.parse_args()

    print("TensorFlow version:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))
    print("Selected model:", args.model)
    print("Mode:", args.mode)

    if args.mode == "one":
        train_one_fold(args.fold, args.model)
    else:
        run_all_folds(args.model)


if __name__ == "__main__":
    main()
