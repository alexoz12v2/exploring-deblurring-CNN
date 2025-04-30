load("@rules_python//python:py_library.bzl", "py_library")
load("@torch_pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "lib",
    srcs = [ 
        ":__init__.py", 
        ":layers/__init__.py",
        ":layers/convir_layers.py",
        ":layers/data.py",
        ":layers/gradual_warmup.py",
        ":layers/original/__init__.py",
        ":layers/original/convir_layers.py",
		":ssim.py",
    ],
    deps = [
		requirement("pillow"),
		requirement("scikit-image"),
		requirement("torch"),
		requirement("nvidia-cublas-cu12"),
		requirement("nvidia-cuda-cupti-cu12"),
		requirement("nvidia-cuda-nvrtc-cu12"),
		requirement("nvidia-curand-cu12"),
		requirement("nvidia-cusolver-cu12"),
		requirement("nvidia-cusparse-cu12"),
		requirement("nvidia-nvjitlink-cu12"),
		requirement("nvidia-nvtx-cu12"),
		requirement("nvidia-cuda-runtime-cu12"),
		requirement("nvidia-cudnn-cu12"),
		requirement("absl-py"),
		requirement("dearpygui"),
		requirement("numpy"),
    ],
    visibility = [ "//visibility:public" ]
)