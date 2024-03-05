from dataclasses import dataclass


@dataclass(kw_only=True)
class MemoryInfo:
    total: int
    free: int
    used: int
