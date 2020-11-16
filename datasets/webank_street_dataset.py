import os
import json
import torch
from PIL import Image


class WebankStreetDataset:
    def __init__(
        self,
        root_dir: str,
        train: bool,
        transform=None,
    ):
        self.__image_dir = os.path.join(root_dir, "Images")
        if train:
            with open(os.path.join(root_dir, "train_label.json"), "rt") as f:
                self.__json = json.load(f)
        else:
            with open(os.path.join(root_dir, "test_label.json"), "rt") as f:
                self.__json = json.load(f)
        self.__labels: dict = dict()
        idx = 1
        for img_json in self.__json:
            for item in img_json["items"]:
                label = item["class"]
                if label not in self.__labels:
                    self.__labels[label] = idx
                    idx += 1
        self.__transform = transform

    def __getitem__(self, index):
        img_json = self.__json[index]
        img = Image.open(
            os.path.join(
                self.__image_dir,
                img_json["image_id"] +
                ".jpg"))
        target = {"boxes": [], "labels": []}
        for item in img_json["items"]:
            target["boxes"].append([int(a) for a in item["bbox"]])
            target["labels"].append(self.__labels[item["class"]])

        target["boxes"] = torch.FloatTensor(target["boxes"])
        target["labels"] = torch.LongTensor(target["labels"])
        if self.__transform is not None:
            img = self.__transform(img)
        return img, target

    def __len__(self):
        return len(self.__json)
