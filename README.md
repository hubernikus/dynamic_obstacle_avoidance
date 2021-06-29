# ObstacleAvoidance Algorithm
---
This package contains a dynamic obstacle avoidance algorithm for concave and convex obstacles as developped in [1] and [2].
---
Requirements: conda, jupyter notebook, python.

## Setup
To setup got to your install/code directory, and type:
```
git clone --recurse-submodules https://github.com/epfl-lasa/dynamic_obstacle_avoidance.git
```
(Make sure submodules are there if various_tools librarys is not installed.

```
cd dynamic_obstacle_avoidance
conda env create -f environment.yml
conda activate dynamic_obstacle_avoidance
pip install -r requirements.txt
python setup.py develop
```

Next step is to install the dependency library:
```
cd lib/various_tools/
python setup.py develop
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


