import setuptools

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cyy_torch_toolbox",
    author="cyy",
    version="0.2",
    author_email="cyyever@outlook.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cyyever/torch_toolbox",
    packages=[
        "cyy_torch_toolbox",
        "cyy_torch_toolbox/dataset",
        "cyy_torch_toolbox/hyper_parameter",
        "cyy_torch_toolbox/model",
        "cyy_torch_toolbox/metrics",
        "cyy_torch_toolbox/data_pipeline",
        "cyy_torch_toolbox/hook",
        "cyy_torch_toolbox/metric_visualizers",
        "cyy_torch_toolbox/data_structure",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
