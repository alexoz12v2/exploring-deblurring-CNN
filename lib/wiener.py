import torch
import torchvision.transforms.v2 as v2
from torchvision.io import decode_image, write_png, ImageReadMode
from lib.dataloader import NormalizeRange
from pathlib import Path

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

def open_image(path: Path, device: torch.device, transform: v2.Transform = v2.Compose([v2.ToDtype(torch.get_default_dtype()), NormalizeRange(), v2.CenterCrop(256)])) -> torch.Tensor:
    image = decode_image(path, mode=ImageReadMode.RGB).to(device)
    return transform(image)


def save_image(image_tensor: torch.Tensor, path: Path):
    # must be C x H x W
    if torch.is_floating_point(image_tensor):
        image_tensor = v2.functional.to_dtype(image_tensor * 255.0, torch.uint8)
    if image_tensor.device.type != 'cpu':
        image_tensor = image_tensor.cpu()

    write_png(image_tensor, str(path))