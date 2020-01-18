class dataset_with_indices:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset.__getitem__(index)
        return data, target, index

    def __len__(self):
        return self.dataset.__len__()
