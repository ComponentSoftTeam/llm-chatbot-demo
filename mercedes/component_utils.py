import hashlib
import pickle
import os
import matplotlib.pyplot as plt
from typing import Union

CACHE_DIR = ".cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


def cached(permanent: Union[bool, callable] = True, pred: callable = None):
    def cache(__func, __pred, *args, **kwargs):
        SEP = "$|$"
        cache_token = (
            f"{__func.__name__}{SEP}"
            f"{SEP.join(str(arg) for arg in args)}{SEP}"
            f"{SEP.join( str(key) + SEP * 2 + str(val) for key, val in kwargs.items())}"
        )

        hex_hash = hashlib.sha256(cache_token.encode()).hexdigest()
        cache_filename: str = os.path.join(CACHE_DIR, f"{__func.__name__}-{hex_hash}")

        if os.path.exists(cache_filename):
            with open(cache_filename, "rb") as cache_file:
                return pickle.load(cache_file)

        result = __func(*args, **kwargs)
        if __pred(result):
            with open(cache_filename, "wb") as cache_file:
                pickle.dump(result, cache_file)

        return result

    if callable(permanent):
        def wrapper(*args, **kwargs):
            return cache(permanent, lambda _: True, *args, **kwargs)
        return wrapper
    elif isinstance(permanent, bool) and callable(pred):
        def outer(func):
            if not permanent:
                for file in os.listdir(CACHE_DIR):
                    if file.startswith(f"{func.__name__}-"):
                        os.remove(os.path.join(CACHE_DIR, file))
            def wrapper(*args, **kwargs):
                return cache(func, pred, *args, **kwargs)
            return wrapper
        return outer
    elif isinstance(permanent, bool) and pred is None:
        def wrapper(func):
            if not permanent:
                for file in os.listdir(CACHE_DIR):
                    if file.startswith(f"{func.__name__}-"):
                        os.remove(os.path.join(CACHE_DIR, file))
            def inner(*args, **kwargs):
                return cache(func, lambda _: True, *args, **kwargs)
            return inner
        return wrapper
    else:
        raise ValueError("Invalid predicate")


def plot_embeddings(projected_dataset_embeddings, title = 'Projected Embeddings'):
    plt.figure(figsize=(8, 6))
    plt.scatter(projected_dataset_embeddings[:, 0], projected_dataset_embeddings[:, 1], s=15)
    plt.gca().set_aspect('equal', 'datalim')
    plt.title(title)
    plt.axis('off')