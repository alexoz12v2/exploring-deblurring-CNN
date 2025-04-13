load("@python_versions//3.10:defs.bzl", "py_binary")
load("@torch_pip_deps//:requirements.bzl", "requirement")

py_binary(
    name = "cli",
    main = ":__main__.py",
    srcs = [":__main__.py"],
    deps = [
        "//lib",
    ],
    visibility = ["//visibility:public"]
)