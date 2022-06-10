# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-04-23

from dynamic_obstacle_avoidance.rotational.graph_handler import GraphElement
from dynamic_obstacle_avoidance.rotational.graph_handler import GraphHandler


def test_graph_element():
    oma = GraphElement(0)
    dad = GraphElement(1)
    baby = GraphElement(2)

    dad.add_child(baby)
    assert baby.parent == dad

    dad.set_parent(oma)
    assert len(oma.children) == 1
    assert oma.children[0] == dad

    # dad.delete()

    # assert baby.parent is None
    # assert len(oma.children) == 0


if (__name__) == "__main__":
    test_graph_element()
    print("Done")
