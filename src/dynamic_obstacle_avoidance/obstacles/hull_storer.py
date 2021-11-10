"""
Store Hull for obstacles
"""
import warnings
from math import pi

from shapely import affinity
from shapely.geometry import Point

import numpy as np


class ObstacleHullsStorer:
    """Shapely storer which automatically deletes when having new assignments change of base.

    Attributes
    ----------
    n_options (int)
    _hull_list: Stores the hull_list of shape x
    """

    n_options = 3

    def __init__(self) -> None:
        self._hull_list = [None for ii in range(2 ** self.n_options)]

    @property
    def global_margin(self):
        """Shapely for global intersection-check."""
        return self.get(global_frame=True, margin=True, reference_extended=False)

    @property
    def local_margin(self):
        hull_ = self.get(global_frame=True, margin=True, reference_extended=True)
        if not hull_:
            hull_ = self.get(global_frame=True, margin=True, reference_extended=False)
        return hull_

    def transform_list_to_index(
        self, global_frame: bool, margin: bool, reference_extended: bool = False
    ) -> None:
        """Chosse between:
        global_frame (bool): local /global
        margin (bool): no-margin / margin
        reference_extended(bool): reference-not extended / reference_extended."""
        arg_vals = np.array([global_frame, margin, reference_extended])
        arg_base = 2 ** np.arange(arg_vals.shape[0])
        return np.sum(arg_vals * arg_base)

    def set(self, value: object, *args, **kwargs) -> None:
        index = self.transform_list_to_index(*args, **kwargs)

        self._hull_list[index] = value

        # TODO: automatically delete when updating certains (e.g. local, no-margin, no-extension

    def get(self, *args, **kwargs) -> object:
        index = self.transform_list_to_index(*args, **kwargs)

        return self._hull_list[index]

    def get_global(
        self, position: np.ndarray, orientation: float, *args, **kwargs
    ) -> object:
        """Transform 2D shapely object from local to global."""
        shapely_ = self.get(*args, **kwargs, global_frame=True)

        if shapely_ is None:
            shapely_ = self.get(*args, **kwargs, global_frame=False)

            if shapely_ is None:
                raise Exception("Value not found.")

        if orientation:
            shapely_ = affinity.rotate(
                shapely_, orientation * 180 / pi, origin=Point(0, 0)
            )
        shapely_ = affinity.translate(shapely_, position[0], position[1])

        return shapely_

    def get_local_edge_points(self) -> np.array:
        """Return an numpy-array of the edges data. First checking if the extended_reference
        exists and continuing to margin-only."""
        shapely_ = self.get(global_frame=False, margin=True, reference_extended=True)

        if shapely_ is None:
            shapely_ = self.get(
                global_frame=False, margin=True, reference_extended=False
            )

        if shapely_.geom_type == "Polygon":
            xy_data = np.array(shapely_.exterior).T
        else:
            xy_data = np.array(shapely_).T

        return xy_data
