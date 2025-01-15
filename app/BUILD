load("@python_versions//3.10:defs.bzl", "py_binary")
load("@torch_pip_deps//:requirements.bzl", "requirement", "data_requirement")

# bazel query @bazel_tools//src/conditions:all
py_binary(
    name = "app",
    main = "app.py",
    srcs = [ ":app.py", ":window.py", ":pywin32_patch.py", ":data.py", ":utils.py" ],
    deps = [
        "//lib",
		requirement("google-api-python-client"),
		requirement("google-auth-httplib2"),
		requirement("google-auth-oauthlib"),
		requirement("torch"),
		requirement("torchvision"),
		requirement("tensorboard"),
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
		requirement("kaggle"),
		requirement("portalocker"),
    ] + select({
		"@bazel_tools//src/conditions:windows": [
			requirement("pywin32")
		],
		"//conditions:default": []
	}),
	data = [
    ] + select({
		"@bazel_tools//src/conditions:windows": [
			data_requirement("pywin32")
		],
		"//conditions:default": []
	}),
	env = select({
		"@bazel_tools//src/conditions:windows": {
			"BAZEL_PYWIN_REMAP": "True", # could be anything
			"BAZEL_FIX_DIR": "True",
		},
		"//conditions:default": {
			"BAZEL_FIX_DIR": "True",
		}
	}),
    visibility = [ "//visibility:public" ]
)