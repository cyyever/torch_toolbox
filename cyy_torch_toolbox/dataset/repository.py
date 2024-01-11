import copy
from typing import Callable

from cyy_naive_lib.log import get_logger
from cyy_naive_lib.reflection import get_kwarg_names

from ..dataset.util import global_dataset_util_factor
from ..factory import Factory
from ..ml_type import DatasetType, MachineLearningPhase

global_dataset_constructors: dict[DatasetType, Factory] = {}


def register_dataset_factory(dataset_type: DatasetType, factory: Factory) -> None:
    assert dataset_type not in global_dataset_constructors
    global_dataset_constructors[dataset_type] = factory


def register_dataset_constructors(
    dataset_type: DatasetType, name: str, constructor: Callable
) -> None:
    if dataset_type not in global_dataset_constructors:
        global_dataset_constructors[dataset_type] = Factory()
    global_dataset_constructors[dataset_type].register(name, constructor)


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
                if "val_split" in new_dataset_kwargs:
                    new_dataset_kwargs["split"] = new_dataset_kwargs["val_split"]
                else:
                    if dataset_type == DatasetType.Text:
                        new_dataset_kwargs["split"] = "valid"
                    else:
                        new_dataset_kwargs["split"] = "val"
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
            if k not in constructor_kwargs:
                discarded_dataset_kwargs.add(k)
        if discarded_dataset_kwargs:
            get_logger().debug("discarded_dataset_kwargs %s", discarded_dataset_kwargs)
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
    if dataset_kwargs is None:
        dataset_kwargs = {}
    constructor_kwargs = get_kwarg_names(dataset_constructor)
    dataset_kwargs_fun = __prepare_dataset_kwargs(
        constructor_kwargs=constructor_kwargs,
        dataset_kwargs=dataset_kwargs,
        cache_dir=cache_dir,
    )
    training_dataset = None
    validation_dataset = None
    test_dataset = None

    for phase in MachineLearningPhase:
        while True:
            try:
                processed_dataset_kwargs = dataset_kwargs_fun(
                    phase=phase, dataset_type=dataset_type
                )
                if processed_dataset_kwargs is None:
                    break
                cache_key = (dataset_name, dataset_type, phase)
                dataset = __dataset_cache.get(cache_key, None)
                if dataset is None:
                    dataset = dataset_constructor(**processed_dataset_kwargs)
                    if dataset_type == DatasetType.Graph:
                        assert len(dataset) == 1
                    __dataset_cache[cache_key] = dataset
                    get_logger().debug(
                        "create and cache dataset %s, id %s with kwargs %s",
                        cache_key,
                        id(dataset),
                        processed_dataset_kwargs,
                    )
                else:
                    get_logger().debug(
                        "use cached dataset %s, id %s with kwargs %s",
                        cache_key,
                        id(dataset),
                        processed_dataset_kwargs,
                    )
                if phase == MachineLearningPhase.Training:
                    training_dataset = dataset
                elif phase == MachineLearningPhase.Validation:
                    validation_dataset = dataset
                else:
                    test_dataset = dataset
                break
            except Exception as e:
                get_logger().debug("has exception %s", e)
                if "of splits is not supported for dataset" in str(e):
                    break
                if "for argument split. Valid values are" in str(e):
                    break
                if "Unknown split" in str(e):
                    break
                raise e

    if training_dataset is None:
        return None

    if validation_dataset is None:
        validation_dataset = test_dataset
        test_dataset = None

    if validation_dataset is None and test_dataset is None:
        datasets: dict = global_dataset_util_factor.get(dataset_type)(
            training_dataset
        ).decompose()
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
    real_dataset_type = dataset_kwargs.get("dataset_type", None)
    similar_names = []

    for dataset_type in DatasetType:
        if real_dataset_type is not None and real_dataset_type != dataset_type:
            continue
        if dataset_type not in global_dataset_constructors:
            continue
        constructor = global_dataset_constructors[dataset_type].get(
            name, case_sensitive=True
        )
        if constructor is not None:
            return __create_dataset(
                dataset_name=name,
                dataset_type=dataset_type,
                dataset_constructor=constructor,
                dataset_kwargs=dataset_kwargs,
                cache_dir=cache_dir,
            )
        similar_names += global_dataset_constructors[dataset_type].get_similar_keys(
            name
        )
    if similar_names:
        get_logger().error(
            "can't find dataset %s, similar datasets are %s",
            name,
            sorted(similar_names),
        )
    else:
        get_logger().error(
            "can't find dataset %s",
            name,
        )
    return None
