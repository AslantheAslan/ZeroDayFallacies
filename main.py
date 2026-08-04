from config import get_config
from training.train import run_training
from training.train_master import run_master_training

if __name__ == "__main__":
    args = get_config()
    print("Parsed class_weight:", args.class_weight)
    if args.master_model:
        run_master_training(args)
    else:
        run_training(args)