import functools
import re

from cyy_naive_lib.log import log_debug, log_error

from ..data_pipeline import DataPipeline, Transform
from .collection import DatasetCollection


def format_prompt(prompt: str, tokenizer, example: str | dict) -> str | dict:
    if isinstance(example, str):
        log_debug("final input is %s", prompt + example)
        return prompt + example
    extra_kwargs: dict[str, str] = {}
    try:
        if "__if__" in prompt or "__endif__" in prompt:
            p = re.compile("__if__{__(.*)_defined__}")
            branch_condition: bool | None = None
            new_lines = []
            for line in prompt.splitlines():
                if line.strip() == "__endif__":
                    assert branch_condition is not None
                    branch_condition = None
                    continue
                m = re.fullmatch(p, line.strip())
                if m is not None:
                    k = m.group(1)
                    defined = bool(example.get(k, False))
                    assert branch_condition is None
                    branch_condition = defined
                    continue
                if branch_condition is not False:
                    new_lines.append(line)
            assert branch_condition is None
            prompt = "\n".join(new_lines)

        for k, v in example.items():
            new_k = f"__{k}_defined__"
            extra_kwargs[new_k] = "true" if v else "false"
            new_k = f"space_join_{k}"
            if new_k in prompt:
                extra_kwargs[new_k] = " ".join([str(a) for a in v])
            new_k = f"comma_join_{k}"
            if new_k in prompt:
                extra_kwargs[new_k] = ",".join([str(a) for a in v])
            if "eos_token" in prompt:
                extra_kwargs["eos_token"] = tokenizer.eos_token
        new_input = prompt.format(**example, **extra_kwargs)
        example["input"] = new_input
    except BaseException as e:
        log_error("formatting fail %s", e)
        log_error("prompt is:\n%s", prompt)
        log_error("input keys are:\n%s", example.keys())
        log_error("extra kwargs are:\n%s", extra_kwargs)
        raise e
    log_debug("final input is %s", example["input"])
    return example


class TextDatasetCollection(DatasetCollection):
    __prompt: str | None = None
    __text_pipeline: DataPipeline | None = None
    __post_prompt_text_pipeline: DataPipeline | None = None

    @property
    def prompt(self) -> str | None:
        return self.__prompt

    def set_prompt(self, prompt: str) -> None:
        assert self.__prompt is None
        self.__prompt = prompt
        assert self.__text_pipeline is None or not self.__text_pipeline.has_transform(
            "format_prompt"
        )

    def append_text_transform(self, transform: Transform) -> None:
        if self.__text_pipeline is None:
            self.__text_pipeline = DataPipeline()
        self.__text_pipeline.append(transform)

    def append_post_prompt_text_transform(self, transform: Transform) -> None:
        if self.__post_prompt_text_pipeline is None:
            self.__post_prompt_text_pipeline = DataPipeline()
        self.__post_prompt_text_pipeline.append(transform)

    def get_text_pipeline(self, tokenizer) -> DataPipeline | None:
        if self.prompt is not None:
            self.append_text_transform(
                Transform(
                    name="format_prompt",
                    cacheable=True,
                    fun=functools.partial(format_prompt, self.prompt, tokenizer),
                )
            )
            if self.__post_prompt_text_pipeline:
                for t in self.__post_prompt_text_pipeline.transforms:
                    self.append_text_transform(t)
        return self.__text_pipeline
