from collections import Counter
from typing import Callable

from ..dataset.util import TextDatasetUtil
from ..ml_type import MachineLearningPhase


def collect_tokens(
    dc,
    tokenizer: Callable,
    phase: MachineLearningPhase | None = None,
) -> Counter:
    counter: Counter = Counter()

    if phase is None:
        util_list = [dc.get_dataset_util(phase=phase) for phase in MachineLearningPhase]
    else:
        util_list = [dc.get_dataset_util(phase=phase)]
    for util in util_list:
        assert isinstance(util, TextDatasetUtil)
        for index in range(len(util)):
            input_text = util.get_sample_text(index)
            match input_text:
                case str():
                    input_text = [input_text]
                # case [*elements]:
                #     pass
                case _:
                    raise NotImplementedError(type(input_text))
            for text in input_text:
                tokens = tokenizer(text)
                counter.update(tokens)
    return counter
