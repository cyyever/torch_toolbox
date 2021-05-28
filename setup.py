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
        "cyy_torch_toolbox/vision_datasets",
        "cyy_torch_toolbox/audio_datasets",
        "cyy_torch_toolbox/models",
        "cyy_torch_toolbox/pipelines",
        "cyy_torch_toolbox/metrics",
        "cyy_torch_toolbox/hooks",
        "cyy_torch_toolbox/metric_visualizers",
        "cyy_torch_toolbox/data_structure",
        "cyy_torch_toolbox/algorithm",
        "cyy_torch_toolbox/algorithm/quantization",
        "cyy_torch_toolbox/algorithm/sample_gradient",
        "cyy_torch_toolbox/algorithm/video",
        "cyy_torch_toolbox/algorithm/hydra",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
