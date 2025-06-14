import json
from pathlib import Path
from contextlib import ExitStack

with open(Path.cwd() / "validation_metrics.json", "r") as f:
    validation_dict = json.load(f)

length = min(
    len(validation_dict['epochs']), len(validation_dict['PSNR']), len(validation_dict['SSIM'])
)

# comma separated with context managers, or exit stack which supports variable length
with ExitStack() as stack:
    psnr_data, ssim_data = (
        stack.enter_context(open(Path.cwd() / "validation_metrics_psnr.txt", "w")),
        stack.enter_context(open(Path.cwd() / "validation_metrics_ssim.txt", "w")),
    )
    psnr_data.write("x y\n")
    ssim_data.write("x y\n")
    for i in range(length):
        psnr_data.write(f"{validation_dict['epochs'][i]} {validation_dict['PSNR'][i]}\n")
        ssim_data.write(f"{validation_dict['epochs'][i]} {validation_dict['SSIM'][i]}\n")
