import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cyy_naive_pytorch_lib",
    author="cyy",
    version="0.1",
    author_email="cyyever@outlook.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cyyever/naive_pytorch_lib",
    packages=[
        "cyy_naive_pytorch_lib",
        "cyy_naive_pytorch_lib/datasets",
        "cyy_naive_pytorch_lib/models",
        "cyy_naive_pytorch_lib/shapely_value",
        "cyy_naive_pytorch_lib/data_structure",
        "cyy_naive_pytorch_lib/algorithm"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
