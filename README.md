# ObstacleAvoidance Algorithm
---
[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](http://unlicense.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
---
This package contains a dynamic obstacle avoidance algorithm for concave and convex obstacles as developped in [1] and [2]. The Code is still in alpha version.
---
Requirements: python

# Description
The algorithms allows to avoid dynamic, star-shaped obstacles. It requires anlytical description of the environment. It allows to navigate within moving, expanding and static obstacles.
<p align="center">
<img src="https://raw.githubusercontent.com/epfl-lasa/dynamic_obstacle_avoidance/main/figures/description/dynamic_crowd_horizontal.gif"  width="550"></>




## Setup
To setup got to your install/code directory, and type:
```sh
git clone --recurse-submodules https://github.com/epfl-lasa/dynamic_obstacle_avoidance.git
```
(Make sure submodules are there if `various_tools` library is not installed. To initialize submodules after cloning use `git submodule update --init --recursive`.
To update all submodules `git submodule update --recursive`

Go to file directory:
```sh
cd dynamic_obstacle_avoidance
``` 

### Custom Environment
Choose your favorite python-environment. I recommend to use [virtual environment venv](https://docs.python.org/3/library/venv.html).
Setup virtual environment (use whatever compatible environment manager that you have with Python >=3.9).

``` bash
python3.10 -m venv .venv
```
with python -V >= 3.9

Activate your environment
``` sh
source .venv/bin/activate
```


### Setup Dependencies
Install all requirements:
``` bash
pip install -r requirements.txt && pip install -e .
```
make sure you also install the submodules (mainly `vartools`)

Install the sub modules:
``` bash
cd libraries/various_tools && pip install -r requirements.txt && pip install -e . && cd ../..
```

### Installation Options
If you want to be able to test/develop additionally install
``` sh
pip install -r requirements_dev.txt
```

For the use of the jupyter notebook additionally install
(found in the examples/notebook folder)
``` sh
pip install -r requirements_notebook.txt
```

## Getting Started
The `example` folder contains a jupyter notebook & general example for static & dynamic simulation in multi-obstacle environment.  

More information about the behavior of the algorithm can be found in the video below (click on the image to watch it):
[![Alt text](https://img.youtube.com/vi/WKso-wu68v8/0.jpg)](https://youtu.be/WKso-wu68v8)

### For Developers
We use pytest in this, to test the code run
``` sh
pytest
```
Code consistency is ensured by using black. Download and setup pre-commit hook for automated formatting
``` sh
pip install pre-commit
```

### 3D Plotting
In order to get nice 3D plots, additionally install mayavi (http://docs.enthought.com/mayavi/mayavi/index.html) & PyQt5

``` sh
pip install mayavi
pip install PyQt5
```


## Debug
You forgot to add the submodules, add them with:
``` sh
git submodule update --init --recursive
```



**References**     
> [1] Huber, Lukas, Aude Billard, and Jean-Jacques E. Slotine. "Avoidance of Convex and Concave Obstacles with Convergence ensured through Contraction." IEEE Robotics and Automation Letters (2019).  

> [2] L. Huber, J. -J. Slotine and A. Billard, "Fast Obstacle Avoidance Based on Real-Time Sensing," in IEEE Robotics and Automation Letters, doi: 10.1109/LRA.2022.3232271.

**Contact**: [Lukas Huber] (https://people.epfl.ch/lukas.huber?lang=en) (lukas.huber AT epfl dot ch)

**Acknowledgments**
This work was supported by EU ERC grant SAHR.

(c) hubernikus
