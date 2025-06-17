from pathlib import Path
from typing import NamedTuple

import torch
from absl import logging

from lib.layers import ConvIR
from lib.dataloader import valid_dataloader
from lib.utils import Adder

from torcheval.metrics.functional import peak_signal_noise_ratio
from ignite.metrics import SSIM

import json

class ValidArgs(NamedTuple):
    test_model: Path | str
    data_dir: Path
    result_dir: Path
    batch_size: int


def valid(model: ConvIR, device: torch.device, args: ValidArgs, ep: int):
    print(args)
    gopro = valid_dataloader(args.data_dir, batch_size=args.batch_size, num_workers=0)
    max_iter = len(gopro)
    model.eval()
    psnr_adder = Adder()
    ssim_adder = SSIM(device=device, data_range=1.0)

    dir_ep = args.result_dir / f"validation_metrics_{ep}.json"

    with torch.inference_mode():
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            logging.info("Start GoPro Evaluation")
            for idx, data in enumerate(gopro):
                input_img, label_img = data
                input_img = input_img.to(device=device, non_blocking=True)
                label_img = label_img.to(device=device, non_blocking=True)

                pred = model(input_img)[2]
                pred_clip = torch.clamp(pred, 0, 1)

                psnr_adder(
                    peak_signal_noise_ratio(pred_clip.squeeze(0), label_img.squeeze(0))
                )
                ssim_adder.update((pred_clip, label_img))

                psnr, ssim = psnr_adder.average(), ssim_adder.compute()

                logging.info(
                    "[Validation] Idx: %03d/%03d mean PSNR so far: %f mean SSIM so far %f",
                    idx,
                    max_iter,
                    psnr,
                    ssim,
                )

    psnr_avg, ssim_avg = psnr_adder.average(), ssim_adder.compute()
    logging.info("[Validation] End: Mean PSNR: %f Mean SSIM %f", psnr_avg, ssim_avg)
    with open(dir_ep, mode="+w") as f:
        json.dump({"psnr": psnr_avg.item(), "ssim": ssim_avg}, f)
    return psnr_adder.average()