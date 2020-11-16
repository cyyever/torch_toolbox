import os
import json
import torch


class DetectionDataset:
    def __init__(self, image_dir: str, json_path: str, suffix=".jpg"):
        self.__image_dir = image_dir
        with open(json_path, "rt") as f:
            self.__json = json.load(f)
        self.__suffix = suffix
        self.__labels: dict = dict()
        idx = 0
        for img_json in self.__json:
            for item in img_json["items"]:
                label = item["class"]
                if label not in self.__labels:
                    self.__labels[label] = idx
                    idx += 1

    def __getitem__(self, index):
        img_json = self.__json[index]
        with open(
            os.path.join(self.__image_dir, img_json["image_id"] + self.__suffix),
            "rb",
        ) as f:
            target = {"boxes": [], "labels": []}
            for item in img_json["items"]:
                target["boxes"].append(item["bbox"])
                target["labels"].append(self.__labels[item["class"]])

            target["boxes"] = torch.FloatTensor(target["boxes"])
            target["labels"] = torch.Int64Tenso(target["labels"])
            return (f.read(), target)

    def __len__(self):
        return len(self.__json)
