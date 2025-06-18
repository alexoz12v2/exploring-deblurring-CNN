import time
from pathlib import Path
from typing import NamedTuple

import os

import torch
import torch.nn.functional as f
from torcheval.metrics.functional import peak_signal_noise_ratio
from absl import logging

from lib.layers import ConvIR
from lib.dataloader import (
    save_image,
    test_dataloader
)
from ignite.metrics import SSIM

import json
from lib.utils import Adder

class TestArgs(NamedTuple):
    test_model: Path
    data_dir: Path
    save_image: bool
    result_dir: Path
    save_comparison: bool
    result_name: str

def test(model: ConvIR, device: torch.device, args: TestArgs):
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    adder = Adder()
    model.eval()
    
    # Crea il path dove salvare il json con i risultati
    res_dict = {"PSNR": -1, "SSIM": -1}
    if args.result_dir and args.result_name:
        res_path = Path(os.path.dirname(args.test_model))
        res_path = res_path.joinpath(args.result_name + ".json")
    elif args.result_dir:
        dataset_name = str(args.data_dir).split('\\')[-1]
        res_path = Path(os.path.dirname(args.test_model))
        res_path = res_path.joinpath(dataset_name+".json")
    else:
        dataset_name = str(args.data_dir).split('\\')[-1]
        res_path = Path(os.path.dirname(args.test_model))
        res_path = res_path.joinpath(dataset_name + ".json")

    npar = sum(p.numel() for p in model.parameters())
    print("\n\n\nNUMERO PARAMETRI: ", npar, "\n\n\n")
        

    with torch.inference_mode():
        psnr_adder = Adder()
        ssim_adder = SSIM(device=device, data_range=1.0)
        factor = 32
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, name = data

            input_img = input_img.to(device)
            label_img = label_img.to(device)
            tm = time.time()
            
            # pad input and target images
            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')

            h, w = label_img.shape[2], label_img.shape[3]
            H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            label_img = f.pad(label_img, (0, padw, 0, padh), 'reflect')

            pred = model(input_img)[2]
            elapsed = time.time() - tm
            adder(elapsed)

            pred_clip = torch.clamp(pred, 0, 1)

            if args.save_image:
                save_image(pred_clip.squeeze(0), args.result_dir / name[0])
                if args.save_comparison:
                    save_image((torch.abs(pred_clip-input_img)*2).squeeze(0), args.result_dir / f'comparison_{name[0]}')

            psnr_adder(
                peak_signal_noise_ratio(pred_clip.squeeze(0), label_img.squeeze(0))
            )
            ssim_adder.update((pred_clip, label_img))
            logging.info(
                "%d iter avg PSNR so far: %.4f avg SSIM so far: %.4f time: %f",
                iter_idx + 1,
                psnr_adder.average(),
                ssim_adder.compute(),
                elapsed,
            )

        logging.info("==========================================================")
        logging.info("The average PSNR is %.4f dB", psnr_adder.average())
        logging.info("Average time: %f", adder.average())
    
        res_dict['PSNR'] = psnr_adder.average().item()
        res_dict['SSIM'] = ssim_adder.compute()
    
    
    with open(res_path, mode="w") as bula:
        json.dump(res_dict, bula)