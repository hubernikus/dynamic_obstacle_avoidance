import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="robot_avoidance",
    version="0.1",
    description="This package allows the obstacle avoidance for robot arms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/pypa/sampleproject",
    # },
    package_dir={
        "": "src",
    },
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: Unix",
    ],
    install_requires=[
        # "zmq==0.0.0",
        # "zmq==0.0.0"
        # "sympy===0.0.0",
    ],
    python_requires=">=3",
)
