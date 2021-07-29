import importlib
import os
import subprocess
import unittest


def is_transformers_available():
    try:
        import transformers

        is_available = True
    except ImportError:
        is_available = False
    return is_available


def is_torch_available():
    return importlib.util.find_spec("torch") is not None


def requires_lib(test_case, lib: list):
    mapping = {
        "torch": is_torch_available(),
        "transformers": is_transformers_available(),
    }
    conditions = [mapping[k] for k in lib]
    if not all(conditions):
        return unittest.skip(f"{lib} NOT AVAILABLE")(test_case)
    return test_case


def try_download_file(url: str):
    file_name = os.path.abspath(os.path.split(url)[-1])
    if os.path.isfile(file_name):
        return file_name
    try:
        print(f"Downloading file from {url}", end=" ... ")
        subprocess.run(["wget", url], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("done!")
    except Exception as e:
        raise ValueError(e)
    return file_name


def if_path_exists(test_case, path: str):
    if not any([os.path.isfile(path), os.path.isdir(path)]):
        return unittest.skip(f"{path} doesn't exist")(test_case)
    return test_case
