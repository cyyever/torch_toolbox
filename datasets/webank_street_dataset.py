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

        # get all classes
        labels = set()
        for json_file in ("train_label.json", "test_label.json"):
            with open(os.path.join(root_dir, json_file), "rt") as f:
                for img_json in json.load(f):
                    for item in img_json["items"]:
                        label = item["class"]
                        labels.add(label)
        if train:
            with open(os.path.join(root_dir, "train_label.json"), "rt") as f:
                self.__json = json.load(f)
        else:
            with open(os.path.join(root_dir, "test_label.json"), "rt") as f:
                self.__json = json.load(f)
        labels = sorted(list(labels))
        self.__labels: dict = dict()
        # label 0 is reserved for the background
        for idx, label in enumerate(labels):
            self.__labels[label] = idx + 1
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

    def __str__(self):
        return "WebankStreetDataset"

    def label_0_for_background(self):
        return True
