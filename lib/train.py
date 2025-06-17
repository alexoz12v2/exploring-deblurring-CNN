from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn.functional as F
from absl import logging
from torch.utils.tensorboard import SummaryWriter

from lib.layers import ConvIR
from lib.dataloader import train_dataloader
from lib.gradual_warmup_scheduler import GradualWarmupScheduler
from lib.utils import Adder, Timer, save_model
from lib.validation import valid, ValidArgs

import json

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
        # Carica stato di model, scheduler e optimizer
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
    avg_time = Adder()

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
                    "Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss content: %7.4f Loss fft: %7.4f Total Loss: %7.4f",
                    iter_timer.toc(),
                    epoch_idx,
                    iter_idx + 1,
                    max_iter,
                    scheduler.get_lr()[0],
                    iter_pixel_adder.average(),
                    iter_fft_adder.average(),
                    iter_pixel_adder.average() + lambda_par*iter_fft_adder.average()
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

        epoch_time = epoch_timer.toc()
        avg_time(epoch_time)
        logging.info(
            "EPOCH: %02d\nElapsed time: %4.2f, avg time/epoch so far: %4.2f, Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f, Epoch loss: %7.4f",
            epoch_idx,
            epoch_time,
            avg_time.average(),
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