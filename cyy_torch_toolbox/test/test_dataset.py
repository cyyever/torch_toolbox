from cyy_torch_toolbox.data_pipeline.dataset import subset_dp
from cyy_torch_toolbox.dataset import create_dataset_collection
from cyy_torch_toolbox.dependency import has_torchvision
from cyy_torch_toolbox.ml_type import MachineLearningPhase


def test_dataset() -> None:
    if has_torchvision:
        mnist = create_dataset_collection("MNIST")
        mnist_training = mnist.get_dataset(MachineLearningPhase.Training)
        assert (
            abs(
                len(mnist_training)
                / len(mnist.get_dataset(MachineLearningPhase.Validation))
                - 12
            )
            < 0.01
        )
        assert mnist.get_dataset_util().channel == 1
        assert len(subset_dp(mnist_training, [1])) == 1

        cifar10 = create_dataset_collection("CIFAR10")
        assert cifar10.get_dataset_util().channel == 3
        assert len(cifar10.get_dataset_util().get_labels()) == 10
        assert abs(
            len(cifar10.get_dataset(MachineLearningPhase.Test))
            - len(cifar10.get_dataset(MachineLearningPhase.Validation))
            <= 1
        )
        print("cifar10 labels are", cifar10.get_label_names())


def test_dataset_labels() -> None:
    if has_torchvision:
        for name in ("MNIST", "CIFAR10"):
            dc = create_dataset_collection(name)
            assert len(dc.get_labels()) == 10
