import torch
import pathlib
from absl import app, flags, logging

def main(args: list[str]) -> None:
    logging.info(pathlib.Path.cwd())
    logging.info(f"torch version: {torch.__version__}")
    if torch.cuda.is_available():
        logging.info("CUDA is available")
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"Number of GPUs: {torch.cuda.device_count()}")
        logging.info(f"Current GPU: {torch.cuda.current_device()}")
        logging.info(f"GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        logging.info("CUDA is not available")

if __name__ == "__main__":
    app.run(main)