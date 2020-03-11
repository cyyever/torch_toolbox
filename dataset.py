import torch
import torchvision
import torchvision.transforms as transforms


class DatasetReducer:
    def __init__(self, dataset, reducers):
        self.dataset = dataset
        self.reducers = reducers
        self.indices = self.__get_indices()

    def __getitem__(self, index):
        return self.dataset.__getitem__(self.indices[index])

    def __len__(self):
        return len(self.indices)

    def __get_indices(self):
        indices = []
        for index, item in enumerate(self.dataset):
            if all(f(index, item) for f in self.reducers):
                indices.append(index)
        return indices


class DatasetMapper:
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


class DatasetWithIndices(DatasetMapper):
    def __init__(self, dataset):
        super().__init__(dataset, [lambda index, item: (*item, index)])


def split_dataset(dataset):
    return [
        torch.utils.data.Subset(
            dataset,
            [index]) for index in range(
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


def get_dataset(name, for_train):
    if name == "MNIST":
        return torchvision.datasets.MNIST(
            root="./data/MNIST/" + str(for_train),
            train=for_train,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.1307], std=[0.3081]),
                ]
            ),
        )
    if name == "CIFAR10":
        return torchvision.datasets.CIFAR10(
            root="./data/CIFAR10/" + str(for_train),
            train=for_train,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        )
    raise NotImplementedError(name)
