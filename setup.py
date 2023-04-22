import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cyy_torch_toolbox",
    author="cyy",
    version="0.1",
    author_email="cyyever@outlook.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cyyever/torch_toolbox",
    packages=[
        "cyy_torch_toolbox",
        "cyy_torch_toolbox/dataset_wrapper",
        "cyy_torch_toolbox/dataset_wrapper/vision",
        "cyy_torch_toolbox/dataset_wrapper/audio",
        "cyy_torch_toolbox/model_transform",
        "cyy_torch_toolbox/model_evaluator",
        "cyy_torch_toolbox/dataset_transform",
        "cyy_torch_toolbox/metrics",
        "cyy_torch_toolbox/models",
        "cyy_torch_toolbox/dataloader",
        "cyy_torch_toolbox/hooks",
        "cyy_torch_toolbox/metric_visualizers",
        "cyy_torch_toolbox/data_structure",
        "cyy_torch_toolbox/algorithm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
