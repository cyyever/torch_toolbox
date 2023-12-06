from cyy_torch_toolbox import Config, MachineLearningPhase


def test_inference() -> None:
    config = Config(dataset_name="MNIST", model_name="LeNet5")
    config.hyper_parameter_config.epoch = 1
    trainer = config.create_trainer()
    inferencer = trainer.get_inferencer(MachineLearningPhase.Test)
    inferencer.inference()
    inferencer.get_sample_loss()
