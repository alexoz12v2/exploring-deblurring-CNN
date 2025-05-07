import argparse
import sys
from pathlib import Path
from pprint import pformat

from absl import app, logging
import torch

from lib.layers.convir_layers import build_net
from lib.layers.utils import ValidArgs, TrainArgs, TestArgs, train, test, valid


def main(args: list[str]) -> None:
    args = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="CLI ConvIR Tool", usage="cli/__main__.py"
    )
    subparsers = parser.add_subparsers(
        title="Available subcommands",
        dest="command",
        metavar="<command>",
        required=True,
        help="Use %(prog)s <command> --help to get help on a specific command'",
    )

    # fmt: off
    # subcommand: Train
    train_parser = subparsers.add_parser("train", help="Train a ConvIR net instance")
    train_parser.add_argument('-b', '--batch_size', type=int, default=4, metavar="<n>", help="batch size used during training")
    train_parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, metavar="<f_lr>", help="learning rate for the optimizer")
    train_parser.add_argument('-w', '--weight_decay', type=float, default=1e-5, metavar="<f_w>", help="weight decay for regularization")
    train_parser.add_argument('-n', '--num_epoch', type=int, default=3000, metavar="<n>", help="total number of training epochs")
    train_parser.add_argument('-pn', '--print_freq', type=int, default=100, metavar="<n>", help="frequency (every n batches within an epoch) to print training progress")
    train_parser.add_argument('-wn', '--num_worker', type=int, default=8, metavar="<n>", help="number of worker processes for the DataLoader")
    train_parser.add_argument('-sn', '--save_freq', type=int, default=100, metavar="<n>", help="frequency (in epochs) to save model checkpoints")
    train_parser.add_argument('-vn', '--valid_freq', type=int, default=100, metavar="<n>", help="frequency (in epochs) to run validation")
    train_parser.add_argument('-rp', '--resume', type=Path, default=None, metavar="<dir>", help="(optional) path to a model checkpoint to resume training from")
    train_parser.add_argument('-d', '--data_dir', type=Path, required=True, metavar="<dir>", help="path to training data directory")
    train_parser.add_argument('-msd', '--model_save_dir', type=Path, default=Path.home() / ".convir", metavar="<dir>", help="path to directory where model checkpoint will be saved (default: ~/.convir)")
    train_parser.add_argument('-agf', '--accumulate-grad-freq', type=int, default=1, metavar='<n>', help="frequency (in batch indexes) after which the cumulated gradient is transferred to the model")
    train_parser.add_argument('-rd', '--result-dir', type=Path, required=True, metavar='<dir>', help='Directory in which deblurred validation images will be stored')
    train_parser.add_argument('-cv', '--convir_version', type=str, required=True, metavar='<c>', help='which version (s, b, l) of ConvIR to train')
    train_parser.add_argument('-l', '--lambda_par', type=float, required=True, metavar='<f>', help='value of the hyperparameter lambda')

    # subcommand: test
    test_parser = subparsers.add_parser("test", help="Test a ConvIR net instance")
    test_parser.add_argument('-tm', '--test_model', type=Path, required=True, metavar="<dir>", help="file containing model checkpoint")
    test_parser.add_argument('-d', '--data_dir', type=Path, required=True, metavar="<dir>", help="path to test data")
    test_parser.add_argument('-rd', '--result_dir', type=Path, metavar="<dir>", help="if present, path in which the resulting image will be saved")
    test_parser.add_argument('-cv', '--convir_version', type=str, required=True, metavar='<c>', help='which version (s, b, l) of ConvIR to test')
    test_parser.add_argument('-sc', '--save_comparison', action='store_true', help='if present togerther with rd, it will also save the difference between the input and the output image')

    # sucommand: validate
    validation_parser = subparsers.add_parser("validate", help="Start validation of a trained model")
    validation_parser.add_argument('-tm', '--test_model', type=Path, required=True, metavar="<dir>", help="file containing model checkpoint")
    validation_parser.add_argument('-d', '--data_dir', type=Path, required=True, metavar="<dir>", help="path to test data")
    validation_parser.add_argument('-rd', '--result_dir', type=Path, metavar="<dir>", help="if present, path in which the results of the validation will be saved")
    validation_parser.add_argument('-cv', '--convir_version', type=str, required=True, metavar='<c>', help='which version (s, b, l) of ConvIR to validate')
    validation_parser.add_argument('-b', '--batch_size', type=int, default=1, metavar="<n>", help="batch size for the validation dataloader (for big models it's recommended to leave the default)")

    # fmt: on

    logging.info("Start")

    # (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables
    # If dedicated GPU memory is larger than physical GPU memory, BUT GPU committed memory isn't don't crash
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    args = parser.parse_args(args)
    print(args.__dict__)

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    logging.info("Device: %s", str(device))
    # torch.set_default_dtype(torch.bfloat16)

    # additional validation (call parser.exit if fails), then construct a namedtuple object and call function
    match args.command:
        case "train":
            if not args.model_save_dir.exists():
                args.model_save_dir.mkdir()
            train_args = TrainArgs(
                **{k: args.__dict__[k] for k in TrainArgs._fields if k in args.__dict__}
            )

            match args.convir_version:
                case 's':
                    n = 4
                case 'b':
                    n = 8
                case 'l':
                    n = 16
                case _:
                    parser.exit(1, "Invalid ConvIR version")

            model = build_net(n).to(device)

            if logging.level_info():
                logging.info("Train Args: \n%s\n", pformat(train_args._asdict()))
            train(model, device, train_args)


        case "test":
            d = {k: args.__dict__[k] for k in TestArgs._fields if k in args.__dict__}
            d["save_image"] = args.result_dir is not None
            d["store_comparison"] = args.save_comparison is not None
            test_args = TestArgs(**d)

            match args.convir_version:
                case 's':
                    n = 4
                case 'b':
                    n = 8
                case 'l':
                    n = 16
                case _:
                    parser.exit(1, "Invalid ConvIR version")
            model = build_net(n).to(device)
            model.load_state_dict(torch.load(test_args.test_model, weights_only=True)["model"])

            if logging.level_info():
                logging.info("Train Args: \n%s\n", pformat(test_args._asdict()))

            test(model, device, test_args)

        
        case "validate":
            d = {k: args.__dict__[k] for k in ValidArgs._fields if k in args.__dict__}
            valid_args = ValidArgs(**d)

            match args.convir_version:
                case 's':
                    n = 4
                case 'b':
                    n = 8
                case 'l':
                    n = 16
                case _:
                    parser.exit(1, "Invalid ConvIR version")
            model = build_net(n).to(device)
            
            valid(model, device, valid_args, 0)
            

        case _:
            parser.exit(1, "Unrecognized command.")


if __name__ == "__main__":
    app.run(main, [sys.argv[0]])
