import copy
from collections.abc import Callable
from typing import Any

from cyy_naive_lib.log import log_debug, log_error, log_info
from cyy_naive_lib.reflection import get_kwarg_names

from ..dataset.util import DatasetUtil, global_dataset_util_factor
from ..ml_type import DatasetType, Factory, MachineLearningPhase


class DatasetFactory(Factory):
    def get(
        self, key: Any, case_sensitive: bool = True, default: Any = None, **kwargs: Any
    ) -> Any:
        return super().get(
            key=key, case_sensitive=case_sensitive, default=default, **kwargs
        )


__global_dataset_constructors: dict[DatasetType, list[DatasetFactory]] = {}


def __get_dataset_types(dataset_type: DatasetType | None = None) -> list[DatasetType]:
    dataset_types = []
    if dataset_type is not None:
        dataset_types.append(dataset_type)
    else:
        dataset_types = list(DatasetType)
    return dataset_types


def register_dataset_factory(
    factory: DatasetFactory, dataset_type: DatasetType | None = None
) -> None:
    for t in __get_dataset_types(dataset_type):
        if t not in __global_dataset_constructors:
            __global_dataset_constructors[t] = []
        __global_dataset_constructors[t].append(factory)


def register_dataset_constructors(
    name: str, constructor: Callable, dataset_type: DatasetType | None = None
) -> None:
    for t in __get_dataset_types(dataset_type):
        if t not in __global_dataset_constructors:
            register_dataset_factory(factory=DatasetFactory(), dataset_type=t)
        __global_dataset_constructors[t][-1].register(name, constructor)


def __prepare_dataset_kwargs(
    constructor_kwargs: set, dataset_kwargs: dict, cache_dir: str
) -> Callable:
    new_dataset_kwargs: dict = copy.deepcopy(dataset_kwargs)
    if "download" not in new_dataset_kwargs:
        new_dataset_kwargs["download"] = True

    def get_dataset_kwargs_of_phase(
        dataset_type: DatasetType, phase: MachineLearningPhase
    ) -> dict | None:
        if "cache_dir" in constructor_kwargs and "cache_dir" not in new_dataset_kwargs:
            new_dataset_kwargs["cache_dir"] = cache_dir
        if "root" in constructor_kwargs and "root" not in new_dataset_kwargs:
            new_dataset_kwargs["root"] = cache_dir
        else:
            new_dataset_kwargs["cache_dir"] = cache_dir
        if "train" in constructor_kwargs:
            # Some dataset only have train and test parts
            if phase == MachineLearningPhase.Validation:
                return None
            new_dataset_kwargs["train"] = phase == MachineLearningPhase.Training
        elif "split" in constructor_kwargs and dataset_type != DatasetType.Graph:
            if phase == MachineLearningPhase.Training:
                new_dataset_kwargs["split"] = new_dataset_kwargs.get(
                    "train_split", "train"
                )
            elif phase == MachineLearningPhase.Validation:
                new_dataset_kwargs["split"] = new_dataset_kwargs.get("val_split", "val")
            else:
                new_dataset_kwargs["split"] = new_dataset_kwargs.get(
                    "test_split", "test"
                )
        elif "subset" in constructor_kwargs:
            if phase == MachineLearningPhase.Training:
                new_dataset_kwargs["subset"] = "train"
            elif phase == MachineLearningPhase.Validation:
                new_dataset_kwargs["subset"] = "valid"
            else:
                new_dataset_kwargs["subset"] = "test"
        else:
            if phase != MachineLearningPhase.Training:
                return None
        discarded_dataset_kwargs = set()
        for k in new_dataset_kwargs:
            if k not in constructor_kwargs and "files" not in k:
                discarded_dataset_kwargs.add(k)
        if "dataset_kwargs" in constructor_kwargs:
            discarded_dataset_kwargs.clear()
            new_dataset_kwargs["dataset_kwargs"] = {
                k: new_dataset_kwargs[k] for k in discarded_dataset_kwargs
            }
        if discarded_dataset_kwargs:
            log_debug("discarded_dataset_kwargs %s", discarded_dataset_kwargs)
            for k in discarded_dataset_kwargs:
                new_dataset_kwargs.pop(k)
        return new_dataset_kwargs

    return get_dataset_kwargs_of_phase


__dataset_cache: dict = {}


def __create_dataset(
    dataset_name: str,
    dataset_type: DatasetType,
    dataset_constructor: Callable,
    dataset_kwargs: dict,
    cache_dir: str,
) -> tuple[DatasetType, dict] | None:
    dataset_kwargs_fun = __prepare_dataset_kwargs(
        constructor_kwargs=get_kwarg_names(dataset_constructor),
        dataset_kwargs=dataset_kwargs,
        cache_dir=cache_dir,
    )
    training_dataset = None
    validation_dataset = None
    test_dataset = None

    for phase in MachineLearningPhase:
        while True:
            try:
                cache_key = (dataset_name, dataset_type, phase)
                dataset = __dataset_cache.get(cache_key)
                if dataset is None:
                    processed_dataset_kwargs = dataset_kwargs_fun(
                        phase=phase, dataset_type=dataset_type
                    )
                    if processed_dataset_kwargs is None:
                        break
                    dataset = dataset_constructor(**processed_dataset_kwargs)
                    if dataset_type == DatasetType.Graph:
                        assert len(dataset) == 1
                    __dataset_cache[cache_key] = dataset
                    log_debug(
                        "create and cache dataset %s, id %s with kwargs %s",
                        cache_key,
                        id(dataset),
                        processed_dataset_kwargs,
                    )
                else:
                    log_debug(
                        "use cached dataset %s, id %s",
                        cache_key,
                        id(dataset),
                    )
                if phase == MachineLearningPhase.Training:
                    training_dataset = dataset
                elif phase == MachineLearningPhase.Validation:
                    validation_dataset = dataset
                else:
                    test_dataset = dataset
                break
            except Exception as e:
                log_debug("has exception %s", e)
                if "of splits is not supported for dataset" in str(e):
                    break
                if "for argument split. Valid values are" in str(e):
                    break
                if "Unknown split" in str(e):
                    break
                raise e

    if training_dataset is None:
        return None

    if validation_dataset is None and not dataset_kwargs.get("no_validation", False):
        validation_dataset = test_dataset
        test_dataset = None

    if validation_dataset is None and test_dataset is None:
        datasets: dict = global_dataset_util_factor.get(
            dataset_type, default=DatasetUtil
        )(training_dataset).decompose()
        if datasets is not None:
            return dataset_type, datasets
    datasets = {MachineLearningPhase.Training: training_dataset}
    if validation_dataset is not None:
        datasets[MachineLearningPhase.Validation] = validation_dataset
    if test_dataset is not None:
        datasets[MachineLearningPhase.Test] = test_dataset

    return dataset_type, datasets


def get_dataset(
    name: str, dataset_kwargs: dict, cache_dir: str
) -> None | tuple[DatasetType, dict]:
    real_dataset_type = dataset_kwargs.get("dataset_type")
    similar_names = []

    if real_dataset_type is not None:
        assert isinstance(real_dataset_type, DatasetType)
        log_info("use dataset type %s", real_dataset_type)
        assert real_dataset_type in __global_dataset_constructors
    dataset_types = __get_dataset_types(real_dataset_type)

    for dataset_type in dataset_types:
        dataset_type = DatasetType(dataset_type)
        if dataset_type not in __global_dataset_constructors:
            continue
        constructor: None | Callable = None
        for factory in __global_dataset_constructors.get(dataset_type, []):
            constructor = factory.get(
                name,
                case_sensitive=True,
                cache_dir=cache_dir,
                dataset_kwargs=dataset_kwargs,
            )
            if constructor is not None:
                break
        if constructor is not None:
            return __create_dataset(
                dataset_name=name,
                dataset_type=dataset_type,
                dataset_constructor=constructor,
                dataset_kwargs=copy.deepcopy(dataset_kwargs),
                cache_dir=cache_dir,
            )

        for factory in __global_dataset_constructors.get(dataset_type, []):
            similar_names += factory.get_similar_keys(name)
    if similar_names:
        log_error(
            "can't find dataset %s, similar datasets are %s",
            name,
            sorted(similar_names),
        )
    else:
        log_error(
            "can't find dataset %s",
            name,
        )
    return None
