import argparse
from train import train
from validate import validate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retail Transaction Model")
    parser.add_argument("--mode", type=str, choices=["train", "validate"], required=True, help="Mode to run the script in: train or validate")
    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "validate":
        validate()
