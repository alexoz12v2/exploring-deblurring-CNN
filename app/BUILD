load("@python_versions//3.9:defs.bzl", "py_binary")
load("@torch_pip_deps//:requirements.bzl", "requirement")

py_binary(
    name = "app",
    main = "app.py",
    srcs = [ "app.py" ],
    deps = [
        requirement("torch"),
    ],
    visibility = [ "//visibility:public" ]
)