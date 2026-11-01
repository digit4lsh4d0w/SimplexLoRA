import torch
import sys
import os

from config import parse_args

sys.path.append(os.getcwd())
from src.utils import get_run_name
import main_llm

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("~~~~~~~~~~~~~~~ GPU ~~~~~~~~~~~~~~~")
        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_name(i))
    else:
        print("~~~~~~~~~~~~~~~ USING CPU ~~~~~~~~~~~~~~~")
    args, parser = parse_args()
    args.run_name = get_run_name(args, parser)
    if args.dataset.lower() in main_llm.DATASETS:
        main_llm.main(args)