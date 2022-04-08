from cyy_torch_toolbox.dataset import dataset_with_indices
from cyy_torch_toolbox.hook import Hook


class AddIndexToDataset(Hook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__raw_dataset = None

    def _before_execute(self, **kwargs):
        model_executor = kwargs["model_executor"]
        model_executor.transform_dataset(self.__change_dataset)

    def _after_execute(self, **kwargs):
        assert self.__raw_dataset is not None
        model_executor = kwargs["model_executor"]
        model_executor.transform_dataset(lambda _: self.__raw_dataset)

    def __change_dataset(self, dataset, _):
        self.__raw_dataset = dataset
        return dataset_with_indices(dataset)
