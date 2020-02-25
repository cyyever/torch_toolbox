import torch


class dataset_with_indices:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset.__getitem__(index)
        return data, target, index

    def __len__(self):
        return self.dataset.__len__()


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
