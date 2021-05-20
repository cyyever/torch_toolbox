from typing import Callable, Optional

from torchaudio.datasets import SPEECHCOMMANDS


class SPEECHCOMMANDS_SIMPLIFIED(SPEECHCOMMANDS):
    classes: list = [
        "backward",
        "bed",
        "bird",
        "cat",
        "dog",
        "down",
        "eight",
        "five",
        "follow",
        "forward",
        "four",
        "go",
        "happy",
        "house",
        "learn",
        "left",
        "marvin",
        "nine",
        "no",
        "off",
        "on",
        "one",
        "right",
        "seven",
        "sheila",
        "six",
        "stop",
        "three",
        "tree",
        "two",
        "up",
        "visual",
        "wow",
        "yes",
        "zero",
    ]

    def __init__(self, transform: Optional[Callable] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.transform = transform

    def __getitem__(self, n: int):
        (
            waveform,
            sample_rate,
            label,
            speaker_id,
            utterance_number,
        ) = super().__getitem__(n)
        if self.transform is not None:
            waveform = self.transform(waveform)
        return (waveform, SPEECHCOMMANDS_SIMPLIFIED.classes.index(label))
