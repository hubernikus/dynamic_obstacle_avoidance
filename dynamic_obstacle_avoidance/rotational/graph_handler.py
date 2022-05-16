""" Handling of graphs for tracking of data-trees. """
# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-04-23

from __future__ import annotations  # Not needed from python 3.10 onwards
import logging
from dataclasses import dataclass, field

from numpy import np

GraphType = np.nparray


# @dataclass(slots=True)
@dataclass
class GraphElement:
    """ Hirarchy Element which Tracks the parent and children of current obstacle. """
    # __slots__ = ['ID', 'parent', 'children']

    reference: GraphType
    parent: GraphElement = None
    children: list[GraphElement] = field(default_factory=lambda : [])

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def set_parent(self, parent):
        self.parent = parent
        parent.children.append(self)

    # def __delete__(self):
    # def __del__(self):
        # self.delete()
        
    def delete(self):
        logging.warn("Active deleting of graph element is not fully defined.")
        # TODO: this requires updating of 'referenceID' etc.
        for child in self.children:
            child.parent = None

        if self.parent is not None:
            self.parent.children.remove(self)


@dataclass
class GraphHandler:
    _graph: list[GraphElement] = field(default_factory=lambda : [])
    _root: int = None

    def set_root(self, reference):
        self._graph.append(
            GraphElement(reference=reference, parent=None)
        )

    def add_element_with_parent(self, reference):
        self._graph.append(
            GraphElement(reference=reference, parent=None)
        )
        
    @property
    def root(self):
        if self._root is None:
            logging.warning("The graph does not have a defined root.")
            return None
        
        return self._root.id
