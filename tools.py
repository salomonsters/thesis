import datetime
import functools
import hashlib
import inspect
import os
import pickle

import numpy as np
import pandas as pd


def create_logger(verbose, prepend=""):
    def log(m):
        if verbose:
            print("{time}:{prepend} {0}".format(m, time=datetime.datetime.now(), prepend=" " + prepend))
    return log


def cache_pickle(func=None, *, cache_path="data/cache", verbose=False, log_prepend="CACHE:"):
    """Keep a cache of previous function calls in a pickle in the data/cache directory

    Usage:
    >>>@cache_picke
    def foo(a):
        pass
    >>>@cache_pickle(verbose=True)
    def foo(a):
        pass"""

    # Lines below allow decorator calls with and without arguments
    if func is None:
        return functools.partial(cache_pickle, cache_path=cache_path, verbose=verbose, log_prepend=log_prepend)

    logger = create_logger(verbose=verbose, prepend=log_prepend)

    def hasher(s):
        return hashlib.md5(str(s).encode('utf-8')).hexdigest()

    @functools.wraps(func)
    def wrapper_cache(*args, **kwargs):
        cache_key = "{0}_{1}_{2}".format(func.__name__,
                                         hasher(inspect.getsource(func)),
                                         hasher(tuple(args) + tuple(kwargs.items())))
        if cache_key not in wrapper_cache.cache_files:
            logger(cache_key + " not yet in cache, creating...")
            result = func(*args, **kwargs)
            wrapper_cache.cache_files.append(cache_key)
            with open(os.path.join(cache_path, cache_key + ".pkl"), "wb") as f:
                pickle.dump(result, f)
            logger(cache_key + " Created and put into cache")
        else:
            with open(os.path.join(cache_path, cache_key + ".pkl"), "rb") as f:
                result = pickle.load(f)
            logger(cache_key + " Retrieved from cache")
        return result
    if not os.path.isdir(cache_path):
        raise FileNotFoundError("Cache directory {0} does not exist".format(cache_path))
    wrapper_cache.cache_files = [f[:-4] for f in os.listdir(cache_path) if os.path.isfile(os.path.join(cache_path, f))
                                 and f.endswith('.pkl')]
    return wrapper_cache


def subsample_dataframe(df, seconds, group_col='fid', time_col='ts'):
    resampled = []
    for track_name, track_group in df.groupby(group_col):
        v = np.arange(track_group[time_col].min() - seconds, track_group[time_col].max() + seconds, seconds)
        df_binned = track_group.groupby(pd.cut(track_group[time_col], v))
        # Use only the first element
        first_rows = df_binned.head(1)
        resampled.append(first_rows)

    df_resampled = pd.concat(resampled)
    return df_resampled


if __name__ == "__main__":
    @cache_pickle
    def add_one(a):
        return a + 1


    @cache_pickle(verbose=True)
    def add_two(a):
        return a + 2

    print(add_one(1))
    print(add_two(3))
