import torch


class dataset_filter:
    def __init__(self, dataset, filters):
        self.dataset = dataset
        self.filters = filters
        self.indices = self.__get_indices()

    def __getitem__(self, index):
        return self.dataset.__getitem__(self.indices[index])

    def __len__(self):
        return len(self.indices)

    def __get_indices(self):
        indices = []
        for index, item in enumerate(self.dataset):
            if all(f(index, item) for f in self.filters):
                indices.append(index)
        return indices


class dataset_mapper:
    def __init__(self, dataset, mappers):
        self.dataset = dataset
        self.mappers = mappers

    def __getitem__(self, index):
        item = self.dataset.__getitem__(index)
        for mapper in self.mappers:
            item = mapper(index, item)
        return item

    def __len__(self):
        return self.dataset.__len__()


class dataset_with_indices(dataset_mapper):
    def __init__(self, dataset):
        super().__init__(dataset, (lambda index, item: (index, *item)))


def split_dataset(dataset):
    return [
        torch.utils.data.Subset(
            dataset,
            index) for index in range(
            len(dataset))]


def split_dataset_by_label(dataset):
    label_map = {}
    for index, sampler in enumerate(dataset):
        label = sampler[1]
        if isinstance(label, torch.Tensor):
            label = label.data.item()
        if label not in label_map:
            label_map[label] = []
        label_map[label].append(index)
    return label_map
