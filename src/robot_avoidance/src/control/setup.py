import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="control",
    version="0.1",
    description="This package implements a GPR evaluation with a ZMQ interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: Unix",
    ],
    install_requires=[
        "zmq==0.0.0",
        # "sklearn",
        # "scikit-learn==0.23.2"
    ],
    python_requires='>=3',
)
