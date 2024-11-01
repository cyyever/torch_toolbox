from cyy_torch_toolbox import create_dataset_collection, get_dataset_collection_sampler

try:
    import cyy_torch_vision  # noqa: F401

    def test_dataset() -> None:
        mnist = create_dataset_collection("MNIST")
        get_dataset_collection_sampler(
            name="iid", dataset_collection=mnist, part_number=3
        )
        get_dataset_collection_sampler(
            name="iid_split_and_flip",
            dataset_collection=mnist,
            part_number=3,
            flip_percent=0.5,
        )
        get_dataset_collection_sampler(
            name="random", dataset_collection=mnist, part_number=3
        )

except BaseException:
    pass
