import argparse
import sys
from pathlib import Path
from pprint import pformat

from absl import app, logging
import torch

from lib.layers.convir_layers import build_net
from lib.layers.data import open_image, save_image
from lib.layers.multiscale_layers import MultiScaleNet
from lib.layers.utils import EvalArgs, TrainArgs, motion_deblur, test_multiscale, train, test, valid


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

    # subcommand: test
    test_parser = subparsers.add_parser("test", help="Test a ConvIR net instance")
    test_parser.add_argument('-tm', '--test_model', type=Path, required=True, metavar="<dir>", help="file containing model checkpoint")
    test_parser.add_argument('-d', '--data_dir', type=Path, required=True, metavar="<dir>", help="path to test data")
    test_parser.add_argument('-rd', '--result_dir', type=Path, metavar="<dir>", help="if present, path in which the resulting image will be saved")
    test_parser.add_argument('-malt', '--alternative_multiscale_model', action='store_true', help="Use alternative model based on residual blocks")

    # subcommand: validate
    validation_parser = subparsers.add_parser("validate", help="Start validation of a trained model")
    validation_parser.add_argument('-tm', '--test_model', type=Path, required=True, metavar="<dir>", help="file containing model checkpoint")
    validation_parser.add_argument('-d', '--data_dir', type=Path, required=True, metavar="<dir>", help="path to test data")
    validation_parser.add_argument('-rd', '--result_dir', type=Path, metavar="<dir>", help="if present, path in which the results of the validation will be saved")

    # subcommand: wiener
    wiener_parser = subparsers.add_parser("wiener", help="Deblur an image with a traditional image processing algorithm (Wiener Deconvolution)")
    wiener_parser.add_argument('-d', '--data_dir', type=Path, required=True, metavar="<dir>", help="path to single image")
    wiener_parser.add_argument('-rd', '--result_dir', type=Path, required=True, metavar="<dir>", help="path in which original and deblurred image will be saved")
    wiener_parser.add_argument('--len-px', '-l', type=int, default=0, metavar="<px>", help="length, in number of pixels, of the motion blur")
    wiener_parser.add_argument('--theta', '-t', type=float, default=0.0, metavar="<rad>", help="direction, in degrees, of the blur. 0 means right, 90 means up")
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
            model = build_net(type="S").to(device)
            #model.compile()
            if logging.level_info():
                logging.info("Train Args: \n%s\n", pformat(train_args._asdict()))

            # ctrain = torch.compile(train)
            # ctrain(model, device, train_args)
            train(model, device, train_args)
        case "test":
            d = {k: args.__dict__[k] for k in EvalArgs._fields if k in args.__dict__}
            d["save_image"] = args.result_dir is not None
            test_args = EvalArgs(**d)
            no_multi = not args.alternative_multiscale_model 
            if no_multi:
                model = build_net().to(device)
                model.load_state_dict(torch.load(test_args.test_model, weights_only=True)["model"])
            else:
                model = MultiScaleNet(n_feats=64, n_resblocks=9, is_skip=True).to(device)
                model.load_state_dict(torch.load(test_args.test_model))

            # model.compile()

            if logging.level_info():
                logging.info("Train Args: \n%s\n", pformat(test_args._asdict()))

            if no_multi:
                test(model, device, test_args)
            else:
                test_multiscale(model, device, test_args)
        
        case "validate":
            d = {k: args.__dict__[k] for k in EvalArgs._fields if k in args.__dict__}
            d["save_image"] = args.result_dir is not None
            valid_args = EvalArgs(**d)
            model = build_net().to(device)
            #model.compile()
            valid(model, device, valid_args, 0)
            
        case "wiener":
            res_dir: Path = args.result_dir
            data_dir: Path = args.data_dir
            if res_dir is None or (not res_dir.is_dir() and res_dir.exists()):
                parser.exit(1, "result path should be a directory")

            if data_dir is None or not data_dir.exists() or not data_dir.is_file():
                parser.exit(1, "data dir should be an existing image path")

            res_dir.mkdir(parents=True, exist_ok=True)
            input_image = open_image(data_dir, device)
            result_image = motion_deblur(input_image)
            save_image(input_image, res_dir / (data_dir.name[:-4] + '_original.png') )
            save_image(result_image, res_dir / data_dir.name)
            # Concatenate along width (dim=2)
            combined = torch.cat([input_image, result_image], dim=2)  # (C, H, W+W)

            # Save combined image
            combined_path = res_dir / (data_dir.name[:-4] + '_combined.png')
            save_image(combined, combined_path)

        case _:
            parser.exit(1, "Unrecognized command.")


if __name__ == "__main__":
    app.run(main, [sys.argv[0]])
