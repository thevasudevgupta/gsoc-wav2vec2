import importlib
import unittest

def is_transformers_available():
    # return importlib.util.find_spec("transformers") is not None
    try:
        import transformers

        is_available = True
    except:
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
