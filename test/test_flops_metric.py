import torch
import torch.nn as nn

from cyy_torch_toolbox.metrics.flops_profiler import FLOPSMetric


def test_flops_metric_manual() -> None:
    """Test FLOPSMetric via manual start/stop with a simple linear model."""
    model = nn.Linear(128, 64)
    metric = FLOPSMetric(model=model)

    metric.start()
    x = torch.randn(4, 128)
    model(x)
    metric.stop()

    flops = metric.get_total_flops()
    # Linear(128, 64): 2 * 128 * 64 * batch_size = 2 * 128 * 64 * 4 = 65536
    assert flops > 0, f"Expected positive FLOPS, got {flops}"
    assert metric._profiled


def test_flops_metric_epoch_hooks() -> None:
    """Test FLOPSMetric via _before_epoch/_after_epoch lifecycle."""
    model = nn.Linear(128, 64)
    metric = FLOPSMetric(model=model)

    # Simulate first epoch
    metric._before_epoch()
    x = torch.randn(4, 128)
    model(x)
    metric._after_epoch(epoch=1)

    flops = metric.get_total_flops()
    assert flops > 0, f"Expected positive FLOPS, got {flops}"
    assert metric.get_epoch_metric(epoch=1, name="total_flops") == flops

    # Second epoch should not re-profile
    metric._before_epoch()
    model(x)
    metric._after_epoch(epoch=2)
    assert metric.get_epoch_metric(epoch=2, name="total_flops") is None


def test_flops_metric_multi_layer() -> None:
    """Test with a multi-layer model to verify FLOPS accumulate."""
    model = nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
    )
    metric = FLOPSMetric(model=model)

    metric.start()
    x = torch.randn(8, 256)
    model(x)
    metric.stop()

    flops = metric.get_total_flops()
    # Two matmuls: should be more than a single Linear
    assert flops > 0, f"Expected positive FLOPS, got {flops}"
