load("@python_versions//3.10:defs.bzl", "py_binary")
load("@torch_pip_deps//:requirements.bzl", "requirement")

py_binary(
    name = "app",
    main = "app.py",
    srcs = [ ":app.py" ],
    deps = [
        "//lib",
		requirement("google-api-python-client"),
		requirement("google-auth-httplib2"),
		requirement("google-auth-oauthlib"),
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