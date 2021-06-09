import importlib


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


def requires_transformers(fn):
    def call(*args, **kwargs):
        if is_transformers_available():
            return fn(*args, **kwargs)
        else:
            print(f"skipping - {fn.__name__} (since `transformers` is not available)")

    return call


def requires_torch(fn):
    def call(*args, **kwargs):
        if is_torch_available():
            return fn(args, **kwargs)
        else:
            print(f"skipping - {fn.__name__} (since `torch` is not available)")

    return call
