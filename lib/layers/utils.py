import os
import time
from pathlib import Path
from typing import NamedTuple

from torchvision.transforms.functional import to_pil_image
import torch
import torch.nn.functional as F
from torcheval.metrics.functional import peak_signal_noise_ratio
from absl import logging
from torch.utils.tensorboard import SummaryWriter

from lib.layers.convir_layers import ConvIR
from lib.layers.data import test_dataloader, train_dataloader, valid_dataloader
from lib.layers.gradual_warmup import GradualWarmupScheduler
from lib.ssim import StructuralSimilarity


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
    gopro = valid_dataloader(args.data_dir, batch_size=1, num_workers=0)
    max_iter = len(gopro)
    model.eval()
    psnr_adder = Adder()

    with torch.inference_mode():
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            logging.info("Start GoPro Evaluation")
            # factor = 32
            for idx, data in enumerate(gopro):
                input_img, label_img = data
                input_img = input_img.to(device=device, non_blocking=True)
                label_img = label_img.to(device=device, non_blocking=True)

                h, w = input_img.shape[2], input_img.shape[3]
                # H, W = ((h + factor) // factor) * factor, ((w + factor) // factor * factor)
                # padh = H - h if h % factor != 0 else 0
                # padw = W - w if w % factor != 0 else 0
                # input_img = F.pad(input_img, (0, padw, 0, padh), "reflect")

                dir_ep = args.result_dir / f'{ep}'
                if not dir_ep.exists():
                    dir_ep.mkdir(parents=True)

                pred = model(input_img)[2]
                pred = pred[:, :, :h, :w]

                pred_clip = torch.clamp(pred, 0, 1)

                psnr = peak_signal_noise_ratio(pred_clip, label_img)
                ssim = StructuralSimilarity(device=device)
                ssim.update(pred_clip, label_img)

                ssim_value = ssim.compute().cpu().numpy()

                psnr_adder(psnr.cpu().numpy())
                logging.info("[Validation] Idx: %03d/%03d PSNR: %f, SSIM: %f", idx, max_iter, psnr, ssim_value)

    logging.info("\n")
    model.train()
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
    for epoch_idx in range(epoch, args.num_epoch + 1):
        epoch_timer.tic()
        iter_timer.tic()
        for iter_idx, batch_data in enumerate(dataloader):
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                input_img, label_img = batch_data
                logging.info("%s", str(input_img.shape))
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
            # logging.info("2")
            # scaled_grad_params = torch.autograd.grad(outputs=scaler.scale(loss), inputs=model.parameters(), retain_graph=True)
            # logging.info("2 after grad")
            # inv_scale = 1./scaler.get_scale()
            # grad_params = [p * inv_scale for p in scaled_grad_params]

            # with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            #     grad_norm = 0
            #     for grad in grad_params:
            #         grad_norm += grad.pow(2).sum() # L2 gradient penalty
            #     grad_norm = grad_norm.sqrt()
            #     loss = loss + grad_norm
            # logging.info("3")

            scaler.scale(loss).backward()

            if (iter_idx + 1) % args.accumulate_grad_freq == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
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
                "%03d epoch \n Average GOPRO PSNR %.2f dB" % (epoch_idx, val_gopro)
            )
            writer.add_scalar("PSNR_GOPRO", val_gopro, epoch_idx)
            if val_gopro >= best_psnr:
                logging.info("Saving model... (best validation so far)")
                torch.save(
                    {"model": model.state_dict()},
                    args.model_save_dir / "Best.pkl",
                )

    logging.info("Saving model... (end of training)")
    save_name = args.model_save_dir / "Final.pkl"
    torch.save({"model": model.state_dict()}, save_name)


# Eval ------------------------------------------------------------------------
class EvalArgs(NamedTuple):
    test_model: Path
    data_dir: Path
    save_image: bool
    result_dir: Path


def test(model: ConvIR, device: torch.device, args: EvalArgs):
    factor = 32
    state_dict = torch.load(args.test_model, weights_only=True)
    model.load_state_dict(state_dict["model"])
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    adder = Adder()
    model.eval()

    with torch.inference_mode():
        psnr_adder = Adder()
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, name = data

            input_img = input_img.to(device)
            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor * factor)
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_img = F.pad(input_img, (0, padw, 0, padh), "reflect")
            tm = time.time()

            pred = model(input_img)[2]
            pred = pred[:, :, :h, :w]
            elapsed = time.time() - tm
            adder(elapsed)

            pred_clip = torch.clamp(pred, 0, 1)
            pred_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            if args.save_image:
                save_name = args.result_dir / name[0]
                pred_clip += 0.5 / 255
                pred = to_pil_image(pred_clip.squeeze(0).cpu(), "RGB")
                pred.save(save_name)

            psnr = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
            psnr_adder(psnr)
            logging.info("%d iter PSNR: %.4f time: %f", iter_idx + 1, psnr, elapsed)

        logging.info("==========================================================")
        logging.info("The average PSNR is %.4f dB", psnr_adder.average())
        logging.info("Average time: %f", adder.average())
