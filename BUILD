load("@bazel_gazelle//:def.bzl", "gazelle")
load("@torch_pip_deps//:requirements.bzl", "all_whl_requirements")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")
load("@rules_python_gazelle_plugin//manifest:defs.bzl", "gazelle_python_manifest")
load("@rules_python_gazelle_plugin//modules_mapping:def.bzl", "modules_mapping")


# This stanza calls a rule that generates targets for managing pip dependencies
# with pip-compile.
compile_pip_requirements(
    name = "requirements",
    src = "requirements.in",
    requirements_txt = "requirements_torch_lock.txt",
    # requirements_windows = "requirements_windows.txt",
)

# This repository rule fetches the metadata for python packages we
# depend on. That data is required for the gazelle_python_manifest
# rule to update our manifest file.
modules_mapping(
    name = "modules_map",
    exclude_patterns = [
        "^_|(\\._)+",  # This is the default.
        "(\\.tests)+",  # Add a custom one to get rid of the psutil tests.
        "^colorama",  # Get rid of colorama on Windows.
        "^tzdata",  # Get rid of tzdata on Windows.
        "^lazy_object_proxy\\.cext$",  # Get rid of this on Linux because it isn't included on Windows.
    ],
    wheels = all_whl_requirements,
)

# Gazelle python extension needs a manifest file mapping from
# an import to the installed package that provides it.
# This macro produces two targets:
# - //:gazelle_python_manifest.update can be used with `bazel run`
#   to recalculate the manifest
# - //:gazelle_python_manifest.test is a test target ensuring that
#   the manifest doesn't need to be updated
# This target updates a file called gazelle_python.yaml, and
# requires that file exist before the target is run.
# When you are using gazelle you need to run this target first.
gazelle_python_manifest(
    name = "gazelle_python_manifest",
    modules_mapping = ":modules_map",
    pip_repository_name = "torch_pip_deps",
    tags = ["exclusive"],
)

# Our gazelle target points to the python gazelle binary.
# This is the simple case where we only need one language supported.
# If you also had proto, go, or other gazelle-supported languages,
# you would also need a gazelle_binary rule.
# See https://github.com/bazelbuild/bazel-gazelle/blob/master/extend.rst#example
# This is the primary gazelle target to run, so that you can update BUILD.bazel files.
# You can execute:
# - bazel run //:gazelle update
# - bazel run //:gazelle fix
# See: https://github.com/bazelbuild/bazel-gazelle#fix-and-update
gazelle(
    name = "gazelle",
    gazelle = "@rules_python_gazelle_plugin//python:gazelle_binary",
)
