import functools

from omegaconf import OmegaConf


def _search_root(cfg, root_key: str):
    for key in root_key.split("/"):
        if key == "":
            continue
        cfg = cfg[key]
    return cfg


def add_args(
    params_file: str, root_key: str = "", as_default: bool = False
) -> callable:
    @functools.wraps(add_args)
    def _decorator(f: callable) -> callable:
        @functools.wraps(f)
        def _wrapper(*args, **kwargs) -> None:
            cfg_params = OmegaConf.load(params_file)
            cfg_params = _search_root(cfg_params, root_key)
            if as_default:
                cfg_params.update(kwargs)
                kwargs = cfg_params
            else:
                kwargs.update(cfg_params)
            return f(*args, **kwargs)

        return _wrapper

    return _decorator


def from_config(params_file: str, root_key: str = "") -> callable:
    @functools.wraps(from_config)
    def _decorator(f: callable) -> callable:
        @functools.wraps(f)
        def _wrapper(*args, **kwargs) -> None:
            cfg_params = OmegaConf.load(params_file)
            cfg_params = _search_root(cfg_params, root_key)
            return f(cfg_params)

        return _wrapper

    return _decorator
