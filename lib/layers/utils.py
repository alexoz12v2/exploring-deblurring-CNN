import math
import time
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torchvision.transforms.v2.functional import grayscale_to_rgb
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
class TrainArgs(NamedTuple):
    data_dir: Path | str
    model_save_dir: Path | str
    result_dir: Path
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_worker: int = 0
    num_epoch: int = 50
    resume: bool = False
    print_freq: int = 10
    valid_freq: int = 10
    save_freq: int = 10
    accumulate_grad_freq: int = 1


def valid(model: ConvIR, device: torch.device, args: TrainArgs, ep: int):
    print(args)
    gopro = valid_dataloader(args.data_dir, batch_size=4, num_workers=0)
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
def train(model: ConvIR, device: torch.device, args: NamedTuple):
    model.train()
    criterion = torch.nn.L1Loss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8
    )
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
        state = torch.load(args.resume)
        epoch = state["epoch"]
        optimizer.load_state_dict(state["optimizer"])
        model.load_state_dict(state["model"])
        logging.info("Resume from %d" % epoch)
        epoch += 1

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

                loss = loss_content + 0.1 * loss_fft

            # Autograd mixed precision gradient penalty
            scaled_grad_params = torch.autograd.grad(
                outputs=scaler.scale(loss), inputs=model.parameters(), retain_graph=True
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
        if epoch_idx % args.save_freq == 0:
            logging.info("Saving model... (save frequency)")
            save_name = args.model_save_dir / f"model_{epoch_idx}.pkl"
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch_idx,
                    "optimizer": optimizer.state_dict(),
                },
                save_name,
            )

        logging.info(
            "EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f",
            epoch_idx,
            epoch_timer.toc(),
            epoch_pixel_adder.average(),
            epoch_fft_adder.average(),
        )
        epoch_fft_adder.reset()
        epoch_pixel_adder.reset()
        scheduler.step()

        if epoch_idx % args.valid_freq == 0:
            val_gopro = valid(model, device, args, epoch_idx)
            logging.info(
                "%03d epoch \n Average GOPRO PSNR %.2f dB", epoch_idx, val_gopro
            )
            writer.add_scalar("PSNR_GOPRO", val_gopro, epoch_idx)
            if val_gopro >= best_psnr:
                logging.info("Saving model... (best validation so far)")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch_idx,
                        "optimizer": optimizer.state_dict(),
                    }, 
                    args.model_save_dir / "Best.pkl",
                )

    logging.info("Saving model... (end of training)")
    save_name = args.model_save_dir / "Final.pkl"
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch_idx,
            "optimizer": optimizer.state_dict(),
        }, 
        save_name,
    )


# Eval ------------------------------------------------------------------------
class EvalArgs(NamedTuple):
    test_model: Path
    data_dir: Path
    save_image: bool
    result_dir: Path


def test_multiscale(model: ConvIR, device: torch.device, args: EvalArgs):
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    adder = Adder()
    model.eval()

    with torch.inference_mode():
        psnr_adder = Adder()
        ssim_adder = SSIM(device=device, data_range=1.0)
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, name = data

            input_img = input_img.to(device)
            input_img_2 = v2.functional.resize(input_img, [input_img.shape[2] // 2, input_img.shape[3] // 2])
            input_img_3 = v2.functional.resize(input_img, [input_img.shape[2] // 4, input_img.shape[3] // 4])

            label_img = label_img.to(device)
            tm = time.time()

            pred = model((input_img, input_img_2, input_img_3))[0]

            elapsed = time.time() - tm
            adder(elapsed)

            pred_clip = torch.clamp(pred, 0, 1)

            if args.save_image:
                save_image(pred_clip.squeeze(0), args.result_dir / name[0])
                save_image(label_img.squeeze(0), args.result_dir / f'original_{name[0]}')

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


def test(model: ConvIR, device: torch.device, args: EvalArgs):
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    adder = Adder()
    model.eval()

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
                save_image(label_img.squeeze(0), args.result_dir / f'original_{name[0]}')

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

# Human like performance:
def motion_deblur(image_tensor: torch.Tensor, length: int = 0, angle: float = 0.0):
    import numpy as np
    from numpy.fft import fft2, ifft2
    from scipy.signal.windows import gaussian
    def wiener_filter(img, kernel, K):
        kernel /= np.sum(kernel)
        dummy = np.copy(img)
        dummy = fft2(dummy)
        kernel = fft2(kernel, s = img.shape)
        kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
        dummy = dummy * kernel
        dummy = np.abs(ifft2(dummy))
        return dummy

    def gaussian_kernel(kernel_size = 3):
        h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
        h = np.dot(h, h.transpose())
        h /= np.sum(h)
        return h

    def motion_blur_kernel(length, angle, size):
        import cv2
        kernel = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        # draw line in middle of kernel
        pt1 = (center - length // 2, center)
        pt2 = (center + length // 2, center)
        kernel = cv2.line(kernel, pt1, pt2, 1, thickness=1)
        # rotate kernel
        rot_mat = cv2.getRotationMatrix2D((center, center), angle, 1.0)
        kernel = cv2.warpAffine(kernel, rot_mat, (size, size))
        kernel /= kernel.sum()
        return kernel

    image_np = image_tensor.cpu().permute(2,1,0).numpy()
    if length > 0:
        # ensure it's odd and fits motion
        kernel_size = max(3, int(2 * round(length) + 1))
        kernel = motion_blur_kernel(length, angle, kernel_size)
    else:
        kernel = gaussian_kernel(3)

    filtered_channels = []
    for c in range(3):
        channel = image_np[:, :, c]
        filtered = wiener_filter(channel, kernel, K = 0.006)
        filtered_channels.append(filtered)

    result_np = np.stack(filtered_channels, axis=0)
    result_tensor = torch.from_numpy(result_np).to(image_tensor.device).clamp(0.0, 1.0).permute(0, 2, 1)
    return result_tensor
