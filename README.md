# ObstacleAvoidance Algorithm
---
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

---
This package contains a dynamic obstacle avoidance algorithm for concave and convex obstacles as developped in [1] and [2].
---
Requirements: conda, jupyter notebook, python.

## Setup
To setup got to your install/code directory, and type:
```sh
git clone --recurse-submodules https://github.com/epfl-lasa/dynamic_obstacle_avoidance.git
```
(Make sure submodules are there if various_tools librarys is not installed.)

Go to file directory:
```sh
cd dynamic_obstacle_avoidance
``` 

## Custom Environment
Choose your favorite python-environment. I recommend to use [virtual environment venv](https://docs.python.org/3/library/venv.html).
Setup virtual environment (use whatever compatible environment manager that you have with Python >3.7).

``` bash
python3.10 -m venv .venv
```
with python -V > 3.7

Activate your environment
``` sh
source .venv/bin/activate
```

# Setup Dependencies
Install all requirements:
``` bash
pip install -r requirements.txt && python setup.py develop
```
make sure you also install the submodules (mainly `vartools`)

Install the submodules:
``` bash
cd lib/various_tools && pip install -r requirements.txt && python setup.py develop && cd ../..
```

## Installation Options
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


## For Developpers
We use pytest in this, to test the code run
``` sh
pytest
```

Download and setup pre-commit hook for automated formatting

``` sh
pip install pre-commit
```

### 3D Plotting
In order to get nice 3D plots, additionally install mayavi (http://docs.enthought.com/mayavi/mayavi/index.html) & PyQt5

``` sh
pip install mayavi
pip install PyQt5
```


# Debug
You forgot to add the submodules, add them with:
``` sh
git submodule update --init --recursive
```

## Getting Started
To run example file for a 'multiple-obstacle' environment:
```
python scripts/examples/examples_multiple_obstaces.py
```


**References**     
> [1] Huber, Lukas, Aude Billard, and Jean-Jacques E. Slotine. "Avoidance of Convex and Concave Obstacles with Convergence ensured through Contraction." IEEE Robotics and Automation Letters (2019).  

> [2] Huber, Lukas, and Slotine Aude Billard. "Avoiding Dense and Dynamic Obstacles in Enclosed Spaces: Application to Moving in a Simulated Crowd." arXiv preprint arXiv:2105.11743 (2021).  

**Contact**: [Lukas Huber] (https://people.epfl.ch/lukas.huber?lang=en) (lukas.huber AT epfl dot ch)

**Acknowledgments**
This work was funded in part by the EU project Crowdbots.

(c) hubernikus
