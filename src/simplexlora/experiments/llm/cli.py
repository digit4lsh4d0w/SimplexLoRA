import torch

from simplexlora.utils import get_run_name
from .config import parse_args
from . import trainer


def main() -> None:
    if torch.cuda.is_available():
        print("~~~~~~~~~~~~~~~ GPU ~~~~~~~~~~~~~~~")
        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_name(i))
    else:
        print("~~~~~~~~~~~~~~~ USING CPU ~~~~~~~~~~~~~~~")
    args, parser = parse_args()
    args.run_name = get_run_name(args, parser)
    if args.dataset.lower() in trainer.DATASETS:
        trainer.main(args)


if __name__ == "__main__":
    main()
