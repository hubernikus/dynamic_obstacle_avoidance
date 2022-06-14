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
class BasicGraphHandler:
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


class GraphHandler:
    """Hirarchy graph which has all node-information built in."""

    def __init__(self, n_nodes: int = None, node_values: list = None):
        """Input argument has to be the number of nodes 'n_nodes' or the node_values."""
        if node_values is not None:
            self._node_values = node_values
        elif n_nodes is not None:
            self._node_values = np.arange(n_nodes).tolist()
        else:
            self._node_values = []

        self._parent_index = [None for ii in range(self.n_nodes)]
        self._children_indices = [[] for ii in range(self.n_nodes)]

    @property
    def n_nodes(self):
        return len(self._node_values)

    @property
    def roots(self):
        return [ind_par for ind_par in self._parent_index if ind_par is None]

    def get_root_indices(self):
        return [ii for ii, ind_par in enumerate(self._parent_index) if ind_par is None]

    def set_root(self, value):
        self._node_values.append(value)
        self._children_indices.append([])
        self._parent_index.append(None)

    def get_parent(self, value):
        parent_index = self._parent_index[self._node_values.index(value)]
        if parent_index is None:
            return parent_index
        return self._node_values[parent_index]

    def get_children(self, value):
        children_inds = self._children_indices[self._node_values.index(value)]
        return [self._node_values[ind] for ind in children_inds]

    def add_element_with_parent(self, value, parent_value):
        self._node_values.append(value)
        self._children_indices.append([])

        ind_node = len(self._node_values) - 1
        ind_parent = self._node_values.index(parent_value)

        # Set parent and children
        self._children_indices[ind_parent].append(ind_node)
        self._parent_index.append(ind_parent)

    def set_parent_of_element(self, value, parent_value):
        ind_node = self._node_values.index(value)
        ind_parent = self._node_values.index(parent_value)

        # Set parent and children
        self._children_indices[ind_parent].append(ind_node)
        self._parent_index[ind_node] = ind_parent

    def get_nodes_hirarchy_descending(self):
        node_indices = self.get_root_indices()

        ii = 0
        while ii < len(node_indices):
            node_indices += self.children_indices[node_indices[ii]]
            ii += 1

        return self._node_values[node_indices]

    def delete_element(self, element: GraphType):
        """Value should the reference to the to-be-deleted element."""
        raise NotImplementedError()
