from typing import Any

from ..ml_type import ModelType


def get_text_template(dataset_name: str, model_type: ModelType) -> list[str] | None:
    match dataset_name.lower():
        case "multi_nli":
            if model_type in (ModelType.Classification, ModelType.TextGeneration):
                return ["<cls>", " ", "{premise}", " ", "<sep>", " ", "{hypothesis}"]
    return None


def interpret_template(inputs: Any, template: list[str]) -> str:
    result: str = ""
    for item in template:
        if item.startswith("{") and item.endswith("}"):
            result += inputs[item[1:-1]]
        else:
            result += item

    return result
