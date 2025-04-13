import argparse
import os
import sys
from pathlib import Path
from pprint import pformat

from absl import app, logging
import torch

from lib.layers.convir_layers import build_net
from lib.layers.utils import TrainArgs, train


def main(args: list[str]) -> None:
    args = sys.argv[1:]
    parser = argparse.ArgumentParser(description="CLI ConvIR Tool", usage="cli/__main__.py")
    subparsers = parser.add_subparsers(
        title="Available subcommands",
        dest="command",
        metavar="<command>",
        required=True,
        help="Use '%(prog)s <command> --help to get help on a specific command'",
    )

    # subcommand: Visualize
    visualize_parser = subparsers.add_parser(
        "visualize", help="Visualize images from a dataset"
    )
    visualize_parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Path to a dataset adhering to the Gopro Blur Dataset format",
    )

    # subcommand: Train
    train_parser = subparsers.add_parser("train", help="Train a ConvIR net instance")
    train_parser.add_argument('-b', '--batch_size', type=int, default=4, metavar="<n>")
    train_parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, metavar="<f_lr>")
    train_parser.add_argument('-w', '--weight_decay', type=float, default=0, metavar="<f_w>")
    train_parser.add_argument('-n', '--num_epoch', type=int, default=3000, metavar="<n>")
    train_parser.add_argument('-pn', '--print_freq', type=int, default=100, metavar="<n>")
    train_parser.add_argument('-wn', '--num_worker', type=int, default=8, metavar="<n>")
    train_parser.add_argument('-sn', '--save_freq', type=int, default=100, metavar="<n>")
    train_parser.add_argument('-vn', '--valid_freq', type=int, default=100, metavar="<n>")
    train_parser.add_argument('-rp', '--resume', type=Path, default=None, metavar="<dir>")
    train_parser.add_argument('-d', '--data_dir', type=Path, required=True, metavar="<dir>")
    train_parser.add_argument('-msd', '--model_save_dir', type=Path, default=Path.home() / ".convir", metavar="<dir>")

    # subcommand: test
    test_parser = subparsers.add_parser("test", help="Test a ConvIR net instance")

    # subcommand: eval
    eval_parser = subparsers.add_parser("eval", help="Inference with a ConvIR network")

    logging.info("Start")

    # (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables
    # If dedicated GPU memory is larger than physical GPU memory, BUT GPU committed memory isn't don't crash
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    args = parser.parse_args(args)
    print(args.__dict__)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Device: %s", str(device))

    # additional validation (call parser.exit if fails), then construct a namedtuple object and call function
    match args.command:
        case "visualize":
            pass
        case "train":
            if not args.model_save_dir.exists():
                args.model_save_dir.mkdir()
            train_args = TrainArgs(**{k: args.__dict__[k] for k in TrainArgs._fields if k in args.__dict__})
            model = build_net().to(device)
            if logging.level_info():
                logging.info("Train Args: \n%s\n", pformat(train_args._asdict()))
                train(model, train_args)
        case "test":
            pass
        case "eval":
            pass
        case _:
            parser.exit(1, "Unrecognized command.")


if __name__ == "__main__":
    app.run(main, [sys.argv[0]])
