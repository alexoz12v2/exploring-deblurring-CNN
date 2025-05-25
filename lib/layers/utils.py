import time
from pathlib import Path
from typing import NamedTuple

import os

import torch
import torch.nn.functional as F
from torcheval.metrics.functional import peak_signal_noise_ratio
from absl import logging
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.v2 as v2

from lib.layers.convir_layers import ConvIR
from lib.layers.data import (
    save_image,
    test_dataloader,
    train_dataloader,
    valid_dataloader,
)
from lib.layers.gradual_warmup import GradualWarmupScheduler
from ignite.metrics import SSIM

import json


# Valid -----------------------------------------------------------------------
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


# Utils -----------------------------------------------------------------------
class Adder(object):
    def __init__(self):
        self.count = 0
        self.num = float(0)

    def reset(self):
        self.count = 0
        self.num = float(0)

    def __call__(self, num):
        self.count += 1
        self.num += num

    def average(self):
        return self.num / self.count


class Timer(object):
    def __init__(self, option="s"):
        self.tm = 0
        self.option = option
        if option == "s":
            self.devider = 1
        elif option == "m":
            self.devider = 60
        else:
            self.devider = 3600

    def tic(self):
        self.tm = time.time()

    def toc(self):
        return (time.time() - self.tm) / self.devider


def check_lr(optimizer):
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group["lr"]
    return lr


# Train -----------------------------------------------------------------------
class TrainArgs(NamedTuple):
    data_dir: Path | str
    model_save_dir: Path | str
    result_dir: Path
    learning_rate: float = 1e-4
    batch_size: int = 8
    num_worker: int = 8
    num_epoch: int = 50
    resume: bool = False
    print_freq: int = 10
    valid_freq: int = 10
    save_freq: int = 10
    accumulate_grad_freq: int = 2
    lambda_par: float = 0.1
    validation_batch_size: int = 1
    freeze_layers: bool = False

def train(model: ConvIR, device: torch.device, args: TrainArgs):
    loss_dict = {"lambda": args.lambda_par, "starting_epoch": 1, "frequency":[], "content":[]}
    loss_save_path = args.result_dir.joinpath("loss.json")

    model.train()
    criterion = torch.nn.L1Loss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8
    )

    lambda_par = args.lambda_par

    dataloader = train_dataloader(args.data_dir, args.batch_size, args.num_worker)
    max_iter = len(dataloader)
    warmup_epochs = 3
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epoch - warmup_epochs, eta_min=1e-6
    )
    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=1,
        total_epoch=warmup_epochs,
        after_scheduler=scheduler_cosine,
    )
    scheduler.step()
    epoch = 1
    if args.resume:
        # Carica stato di modell, scheduler e optimizer
        state = torch.load(args.resume)
        epoch = state["epoch"]
        
        scheduler.load_state_dict(state["scheduler"])
        logging.info("Resume from %d" % epoch)
        epoch += 1

        # Riprende logging della loss
        if loss_save_path.exists():
            with open(loss_save_path, mode='r') as f:
                loss_dict = json.load(f)
        else:
            loss_dict["starting_epoch"] = epoch
            
        # Congela layer se viene passato l'argomento
        if args.freeze_layers:
            # freeze blocchi encoder
            for p in model.Encoder.parameters():
                p.requires_grad = False
            
            # freeze feat_extract
            for i in range(3):
                for p in model.feat_extract[i].parameters():
                    p.requires_grad = False
                    
            # freeze SCM
            for p in model.SCM1.parameters():
                p.requires_grad = False
            for p in model.SCM2.parameters():
                p.requires_grad = False
                
            # freeze FAM
            for p in model.FAM1.parameters():
                p.requires_grad = False
            for p in model.FAM2.parameters():
                p.requires_grad = False

            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), 
                lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8
            )

            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=warmup_epochs,
                after_scheduler=scheduler_cosine,
            )
            scheduler.step()        

        else:
            optimizer.load_state_dict(state["optimizer"])
            model.load_state_dict(state["model"])


    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))

    writer = SummaryWriter()
    epoch_pixel_adder = Adder()
    epoch_fft_adder = Adder()
    iter_pixel_adder = Adder()
    iter_fft_adder = Adder()
    epoch_timer = Timer("m")
    iter_timer = Timer("m")
    best_psnr = -1

    logging.info(
        "Autocast Enabled: %d",
        torch.amp.autocast_mode.is_autocast_available(device.type),
    )
    scaler = torch.amp.GradScaler(device=device.type)
    if device.type == 'cuda':
        low_prec_type = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        low_prec_type = torch.bfloat16

    for epoch_idx in range(epoch, args.num_epoch + 1):
        epoch_timer.tic()
        iter_timer.tic()
        for iter_idx, batch_data in enumerate(dataloader):
            with torch.amp.autocast(device_type=device.type, dtype=low_prec_type):
                input_img, label_img = batch_data
                input_img = input_img.to(device=device, non_blocking=True)
                label_img = label_img.to(device=device, non_blocking=True)
                pred_img = model(input_img)
                label_img2 = F.interpolate(label_img, scale_factor=0.5, mode="bilinear")
                label_img4 = F.interpolate(
                    label_img, scale_factor=0.25, mode="bilinear"
                )

                if device.type == "cuda":
                    stream1 = torch.cuda.Stream(device=device)
                    stream2 = torch.cuda.Stream(device=device)
                    stream3 = torch.cuda.Stream(device=device)

                    torch.cuda.synchronize()

                    with torch.cuda.stream(stream1):
                        label_fft1 = torch.fft.fft2(label_img4, dim=(-2, -1))
                        pred_fft1 = torch.fft.fft2(pred_img[0], dim=(-2, -1))
                        label_fft1 = torch.view_as_real(label_fft1)
                        pred_fft1 = torch.view_as_real(pred_fft1)

                        f1 = criterion(pred_fft1, label_fft1)

                        l1 = criterion(pred_img[0], label_img4)

                    with torch.cuda.stream(stream2):
                        label_fft2 = torch.fft.fft2(label_img2, dim=(-2, -1))
                        pred_fft2 = torch.fft.fft2(pred_img[1], dim=(-2, -1))
                        label_fft2 = torch.view_as_real(label_fft2)
                        pred_fft2 = torch.view_as_real(pred_fft2)

                        f2 = criterion(pred_fft2, label_fft2)

                        l2 = criterion(pred_img[1], label_img2)

                    with torch.cuda.stream(stream3):
                        label_fft3 = torch.fft.fft2(label_img, dim=(-2, -1))
                        pred_fft3 = torch.fft.fft2(pred_img[2], dim=(-2, -1))
                        label_fft3 = torch.view_as_real(label_fft3)
                        pred_fft3 = torch.view_as_real(pred_fft3)

                        f3 = criterion(pred_fft3, label_fft3)

                        l3 = criterion(pred_img[2], label_img)

                    # Make sure all streams are done before proceeding
                    torch.cuda.synchronize()
                else:
                    # Fallback for CPU or non-CUDA device
                    label_fft1 = torch.view_as_real(
                        torch.fft.fft2(label_img4, dim=(-2, -1))
                    )
                    pred_fft1 = torch.view_as_real(
                        torch.fft.fft2(pred_img[0], dim=(-2, -1))
                    )

                    label_fft2 = torch.view_as_real(
                        torch.fft.fft2(label_img2, dim=(-2, -1))
                    )
                    pred_fft2 = torch.view_as_real(
                        torch.fft.fft2(pred_img[1], dim=(-2, -1))
                    )

                    label_fft3 = torch.view_as_real(
                        torch.fft.fft2(label_img, dim=(-2, -1))
                    )
                    pred_fft3 = torch.view_as_real(
                        torch.fft.fft2(pred_img[2], dim=(-2, -1))
                    )

                    l1 = criterion(pred_img[0], label_img4)
                    l2 = criterion(pred_img[1], label_img2)
                    l3 = criterion(pred_img[2], label_img)

                    f1 = criterion(pred_fft1, label_fft1)
                    f2 = criterion(pred_fft2, label_fft2)
                    f3 = criterion(pred_fft3, label_fft3)

                loss_content = l1 + l2 + l3
                loss_fft = f1 + f2 + f3

                loss = loss_content + lambda_par * loss_fft

            # Autograd mixed precision gradient penalty
            scaled_grad_params = torch.autograd.grad(
                outputs=scaler.scale(loss), inputs=trainable_params, retain_graph=True
            )
            inv_scale = 1.0 / scaler.get_scale()
            grad_params = [p * inv_scale for p in scaled_grad_params]

            with torch.amp.autocast(device_type=device.type, dtype=low_prec_type):
                grad_norm = 0
                for grad in grad_params:
                    grad_norm += grad.pow(2).sum()  # L2 gradient penalty
                grad_norm = grad_norm.sqrt()
                loss = loss + grad_norm

            scaler.scale(loss).backward()

            if (iter_idx + 1) % args.accumulate_grad_freq == 0:
                scaler.unscale_(optimizer)
                # O usi gradient penalty o usi gradient clipping
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            iter_pixel_adder(loss_content.item())
            iter_fft_adder(loss_fft.item())

            epoch_pixel_adder(loss_content.item())
            epoch_fft_adder(loss_fft.item())

            if (iter_idx + 1) % args.print_freq == 0:
                logging.info(
                    "Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss content: %7.4f Loss fft: %7.4f",
                    iter_timer.toc(),
                    epoch_idx,
                    iter_idx + 1,
                    max_iter,
                    scheduler.get_lr()[0],
                    iter_pixel_adder.average(),
                    iter_fft_adder.average(),
                )
                writer.add_scalar(
                    "Pixel Loss",
                    iter_pixel_adder.average(),
                    iter_idx + (epoch_idx - 1) * max_iter,
                )
                writer.add_scalar(
                    "FFT Loss",
                    iter_fft_adder.average(),
                    iter_idx + (epoch_idx - 1) * max_iter,
                )
                iter_timer.tic()
                iter_pixel_adder.reset()
                iter_fft_adder.reset()

        # back to epoch loop
        epoch_loss = epoch_pixel_adder.average() + args.lambda_par*epoch_fft_adder.average()
        loss_dict['content'].append(epoch_pixel_adder.average())
        loss_dict['frequency'].append(epoch_fft_adder.average())

        if epoch_idx % args.save_freq == 0:
            logging.info("Saving model... (save frequency)")
            save_name = args.model_save_dir / f"model_{epoch_idx}.pkl"
            save_model(model=model, scheduler=scheduler, optimizer=optimizer, epoch=epoch_idx, save_path=save_name)

            with open(loss_save_path, mode="w") as f:
                json.dump(loss_dict, f)

        logging.info(
            "EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f, Epoch loss: %7.4f",
            epoch_idx,
            epoch_timer.toc(),
            epoch_pixel_adder.average(),
            epoch_fft_adder.average(),
            epoch_loss,
        )

        epoch_fft_adder.reset()
        epoch_pixel_adder.reset()
        scheduler.step()

        if epoch_idx % args.valid_freq == 0:
            val_args = ValidArgs(
                test_model=args.model_save_dir / f"model_{epoch_idx}.pkl",
                data_dir=args.data_dir,
                batch_size=args.validation_batch_size,
                result_dir=args.result_dir
            )
            
            val_gopro = valid(model, device, val_args, epoch_idx)
            logging.info(
                "%03d epoch \n Average GOPRO PSNR %.2f dB", epoch_idx, val_gopro
            )
            writer.add_scalar("PSNR_GOPRO", val_gopro, epoch_idx)
            if val_gopro >= best_psnr:
                logging.info("Saving model... (best validation so far)")
                save_model(model=model, scheduler=scheduler, optimizer=optimizer, epoch=epoch_idx, save_path=args.model_save_dir / "Best.pkl")

    logging.info("Saving model... (end of training)")
    save_name = args.model_save_dir / "Final.pkl"
    save_model(model=model, scheduler=scheduler, optimizer=optimizer, epoch=epoch_idx, save_path=save_name)
    with open(loss_save_path, mode="w") as f:
        json.dump(loss_dict, f)

    


# Eval ------------------------------------------------------------------------
class TestArgs(NamedTuple):
    test_model: Path
    data_dir: Path
    save_image: bool
    result_dir: Path
    store_comparison: bool
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
        

    with torch.inference_mode():
        psnr_adder = Adder()
        ssim_adder = SSIM(device=device, data_range=1.0)
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, name = data

            input_img = input_img.to(device)
            label_img = label_img.to(device)
            tm = time.time()

            pred = model(input_img)[2]
            elapsed = time.time() - tm
            adder(elapsed)

            pred_clip = torch.clamp(pred, 0, 1)

            if args.save_image:
                save_image(pred_clip.squeeze(0), args.result_dir / name[0])
                if args.store_comparison:
                    save_image((torch.abs(pred_clip-input_img)*10).squeeze(0), args.result_dir / f'comparison_{name[0]}')

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
    
    if args.result_dir:
        with open(res_path, mode="w") as bula:
            json.dump(res_dict, bula)


def save_model(model:ConvIR, scheduler:GradualWarmupScheduler, optimizer:torch.optim.Adam, epoch:int, save_path:Path):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, 
        save_path,
    )