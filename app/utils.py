import time
import os
import torch
from lib.layers.data import test_dataloader, train_dataloader, valid_dataloader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio
from absl import logging

from lib.layers.gradual_warmup import GradualWarmupScheduler

from multiprocessing import Value, Manager
from contextlib import contextmanager


class AtomicBool:
    def __init__(self, manager, initial=False):
        self._value = Value("b", initial)  # 'b' for boolean
        self._shutdown = Value("b", False)
        self._lock = manager.Lock()

    def load(self) -> bool:
        with self._lock:
            return self._value.value

    def store(self, value: bool) -> None:
        with self._lock:
            self._value.value = value

    def should_close(self) -> bool:
        with self._lock:
            return self._shutdown.value

    def request_close(self) -> None:
        with self._lock:
            self._shutdown.value = True

    def set(self):
        self.store(True)

    def reset(self):
        self.store(False)

    def compare_and_swap(self, expected: bool, value: bool) -> bool:
        with self._lock:
            if self._value.value == expected:
                self._value.value = value
                return True
            else:
                return False


@contextmanager
def lock_bool(mtx: AtomicBool, *, timeout_millis: float):
    start_time = time.time()
    timeout_secs = timeout_millis / 1000
    acquired = False  # Track whether the lock was successfully acquired
    try:
        while not acquired:
            if mtx.compare_and_swap(False, True):
                acquired = True
            elif time.time() - start_time > timeout_secs:
                yield False  # Lock not acquired within timeout
                return  # Exit the context manager without releasing
            else:
                time.sleep(0.0001)
        yield True  # Lock successfully acquired
    finally:
        if acquired:  # Only release the lock if it was acquired
            if not mtx.compare_and_swap(True, False):
                raise ValueError("Something went wrong while releasing the lock")


# Valid -----------------------------------------------------------------------
def _valid(model, args, ep):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gopro = valid_dataloader(args.data_dir, batch_size=1, num_workers=0)
    model.eval()
    psnr_adder = Adder()

    with torch.no_grad():
        logging.info("Start GoPro Evaluation")
        factor = 32
        for idx, data in enumerate(gopro):
            input_img, label_img = data
            input_img = input_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor * factor)
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_img = F.pad(input_img, (0, padw, 0, padh), "reflect")

            if not os.path.exists(os.path.join(args.result_dir, "%d" % (ep))):
                os.mkdir(os.path.join(args.result_dir, "%d" % (ep)))

            pred = model(input_img)[2]
            pred = pred[:, :, :h, :w]

            pred_clip = torch.clamp(pred, 0, 1)
            p_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            psnr = peak_signal_noise_ratio(p_numpy, label_numpy, data_range=1)

            psnr_adder(psnr)
            logging.info("\r%03d" % idx, end=" ")

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
def _train(model, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    for epoch_idx in range(epoch, args.num_epoch + 1):
        epoch_timer.tic()
        iter_timer.tic()
        for iter_idx, batch_data in enumerate(dataloader):
            input_img, label_img = batch_data
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            optimizer.zero_grad()
            pred_img = model(input_img)
            label_img2 = F.interpolate(label_img, scale_factor=0.5, mode="bilinear")
            label_img4 = F.interpolate(label_img, scale_factor=0.25, mode="bilinear")
            l1 = criterion(pred_img[0], label_img4)
            l2 = criterion(pred_img[1], label_img2)
            l3 = criterion(pred_img[2], label_img)
            loss_content = l1 + l2 + l3

            label_fft1 = torch.fft.fft2(label_img4, dim=(-2, -1))
            label_fft1 = torch.stack((label_fft1.real, label_fft1.imag), -1)

            pred_fft1 = torch.fft.fft2(pred_img[0], dim=(-2, -1))
            pred_fft1 = torch.stack((pred_fft1.real, pred_fft1.imag), -1)

            label_fft2 = torch.fft.fft2(label_img2, dim=(-2, -1))
            label_fft2 = torch.stack((label_fft2.real, label_fft2.imag), -1)

            pred_fft2 = torch.fft.fft2(pred_img[1], dim=(-2, -1))
            pred_fft2 = torch.stack((pred_fft2.real, pred_fft2.imag), -1)

            label_fft3 = torch.fft.fft2(label_img, dim=(-2, -1))
            label_fft3 = torch.stack((label_fft3.real, label_fft3.imag), -1)

            pred_fft3 = torch.fft.fft2(pred_img[2], dim=(-2, -1))
            pred_fft3 = torch.stack((pred_fft3.real, pred_fft3.imag), -1)

            f1 = criterion(pred_fft1, label_fft1)
            f2 = criterion(pred_fft2, label_fft2)
            f3 = criterion(pred_fft3, label_fft3)
            loss_fft = f1 + f2 + f3

            loss = loss_content + 0.1 * loss_fft
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
            optimizer.step()

            iter_pixel_adder(loss_content.item())
            iter_fft_adder(loss_fft.item())

            epoch_pixel_adder(loss_content.item())
            epoch_fft_adder(loss_fft.item())

            if (iter_idx + 1) % args.print_freq == 0:
                logging.info(
                    "Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss content: %7.4f Loss fft: %7.4f"
                    % (
                        iter_timer.toc(),
                        epoch_idx,
                        iter_idx + 1,
                        max_iter,
                        scheduler.get_lr()[0],
                        iter_pixel_adder.average(),
                        iter_fft_adder.average(),
                    )
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
        overwrite_name = os.path.join(args.model_save_dir, "model.pkl")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch_idx,
            },
            overwrite_name,
        )

        if epoch_idx % args.save_freq == 0:
            save_name = os.path.join(args.model_save_dir, "model_%d.pkl" % epoch_idx)
            torch.save({"model": model.state_dict()}, save_name)
        logging.info(
            "EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f"
            % (
                epoch_idx,
                epoch_timer.toc(),
                epoch_pixel_adder.average(),
                epoch_fft_adder.average(),
            )
        )
        epoch_fft_adder.reset()
        epoch_pixel_adder.reset()
        scheduler.step()

        if epoch_idx % args.valid_freq == 0:
            val_gopro = _valid(model, args, epoch_idx)
            logging.info(
                "%03d epoch \n Average GOPRO PSNR %.2f dB" % (epoch_idx, val_gopro)
            )
            writer.add_scalar("PSNR_GOPRO", val_gopro, epoch_idx)
            if val_gopro >= best_psnr:
                torch.save(
                    {"model": model.state_dict()},
                    os.path.join(args.model_save_dir, "Best.pkl"),
                )

    save_name = os.path.join(args.model_save_dir, "Final.pkl")
    torch.save({"model": model.state_dict()}, save_name)


# Eval ------------------------------------------------------------------------
def _eval(model, args):
    factor = 32
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    adder = Adder()
    model.eval()

    with torch.no_grad():
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
                save_name = os.path.join(args.result_dir, name[0])
                pred_clip += 0.5 / 255
                pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), "RGB")
                pred.save(save_name)

            psnr = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
            psnr_adder(psnr)
            logging.info("%d iter PSNR: %.4f time: %f" % (iter_idx + 1, psnr, elapsed))

        logging.info("==========================================================")
        logging.info("The average PSNR is %.4f dB" % (psnr_adder.average()))
        logging.info("Average time: %f" % adder.average())
