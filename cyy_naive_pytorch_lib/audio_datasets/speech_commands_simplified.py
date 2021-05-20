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

    def __getitem__(self, n: int):
        (
            waveform,
            sample_rate,
            label,
            speaker_id,
            utterance_number,
        ) = super().__getitem__(n)
        return (waveform, sample_rate, SPEECHCOMMANDS_SIMPLIFIED.classes.index(label))
