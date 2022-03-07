import inspect
from typing import Any, Callable


def get_kwarg_names(fun: Callable) -> set:
    sig = inspect.signature(fun)
    return {
        p.name
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }


def call_fun(fun: Callable, kwargs: dict) -> Any:
    return fun(**{k: v for k, v in kwargs.items() if k in get_kwarg_names(fun)})


def get_class_attrs(obj: Any, filter_fun: Callable = None) -> dict:
    classes = {
        name: getattr(obj, name)
        for name in dir(obj)
        if inspect.isclass(getattr(obj, name))
    }
    if filter_fun is not None:
        classes = {k: v for k, v in classes.items() if filter_fun(k, v)}
    return classes
