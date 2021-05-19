from hook import Callback
from dataset import dataset_with_indices


class AddIndexToDataset(Callback):
    def __init__(self):
        super().__init__()
        self.__raw_dataset = None

    def _before_execute(self, **kwargs):
        model_executor = kwargs["model_executor"]
        model_executor.transform_dataset(self.__change_dataset)

    def _after_execute(self, **kwargs):
        assert self.__raw_dataset is not None
        model_executor = kwargs["model_executor"]
        model_executor.transform_dataset(lambda dataset: self.__raw_dataset)

    def __change_dataset(self, dataset):
        self.__raw_dataset = dataset
        return dataset_with_indices(dataset)
