import tracemalloc

from cyy_naive_lib.log import get_logger
from hook import Hook


class MemoryTracker(Hook):
    def _after_epoch(self, **kwargs):
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")
        get_logger().warning("top 10 lines with most memory consumed")
        for stat in top_stats[:10]:
            get_logger().warning("  %s", str(stat))
