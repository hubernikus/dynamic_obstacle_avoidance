""" Handling of graphs for tracking of data-trees. """
# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-04-23

from __future__ import annotations  # Not needed from python 3.10 onwards
import logging
from dataclasses import dataclass, field

import numpy as np

GraphType = np.ndarray


# @dataclass(slots=True)
@dataclass
class GraphElement:
    """Hirarchy Element which Tracks the parent and children of current obstacle."""

    # __slots__ = ['ID', 'parent', 'children']

    value: GraphType
    parent: GraphElement = None
    children: list[GraphElement] = field(default_factory=lambda: [])

    @property
    def number_of_children(self):
        return len(self.children)

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def set_parent(self, parent):
        self.parent = parent
        parent.children.append(self)

    def expel(self):
        """Removes all connection to all under and overlying elements."""
        logging.warn("Active deleting of graph element is not fully defined.")
        # TODO: this requires updating of 'referenceID' etc.
        for child in self.children:
            child.parent = None

        if self.parent is not None:
            self.parent.children.remove(self)


@dataclass
class GraphHandler:
    _root: int = None
    _graph: list[GraphElement] = field(default_factory=lambda: [])

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, value):
        self._root = GraphElement(value=value, parent=None)
        self._graph.append(self._root)

    def add_element_with_parent(self, child, parent):
        self._graph.append(GraphElement(value=child, parent=parent))

    def delete_element(self, element):
        """Value should the reference to the to-be-deleted element."""
        element.expel()
        self._graph.remove(element)
