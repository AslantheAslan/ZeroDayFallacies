from config import get_config
from training.train import run_training

if __name__ == "__main__":
    args = get_config()
    run_training(args)