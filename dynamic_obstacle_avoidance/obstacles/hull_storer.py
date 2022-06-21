"""
Store Hull for obstacles
"""
import warnings
from functools import lru_cache

from math import pi

from shapely import affinity
from shapely.geometry import Point

import numpy as np


class ObstacleHullsStorer:
    """Shapely storer which automatically deletes when having new assignments
    change of base.

    Note, that all setters are per default in the local frame.
    The absolut frame seeting is only happening when a specific object is retrieved
    The setters are hirarchical, i.e. if a lower level hull is updated,
    the upper level-hull is updated, too.

    We have following hirarchy:
    no_hull < hull < hull_with_reference_extension

    Attributes
    ----------
    n_options (int)
    _hull_list: Stores the hull_list of shape x
    _state: Reference to the state of the object. this is used
    _temp_state: This is the comparsion state (used in 'check_if_pose_has_updated')

    Methods
    -------
    check_if_pose_has_updated
    """

    # Maybe todo: set upper indeces to zero automatically...

    n_options = 2

    # Current implementation is for 2D only
    dimension = 2

    def __init__(self, state, margin=None) -> None:
        self._hull_list_local = [None for ii in range(self.n_options**2)]
        self._hull_list_global = [None for ii in range(self.n_options**2)]

        # TODO: instead of obstacle, pass state
        self._state = state

        if self._state.orientation is None:
            orientation = 0
        else:
            orientation = self._state.orientation

        self._temp_state = np.hstack([self._state.center_position, orientation])

    def check_if_pose_has_updated(self) -> bool:
        """Returns bool which states if the state (position/orienation) has
        changed since the last evaluation. In that case the global list is deleted."""
        if self._state.orientation is None:
            orientation = 0
        else:
            orientation = self._state.orientation

        temp_state = np.hstack([self._state.center_position, orientation])

        if np.allclose(temp_state, self._temp_state):
            return False

        # Reset state
        self.temp_state = temp_state
        self._hulls_global = [None for ii in range(self.n_options**2)]

        return True

    def transform_list_to_index(
        self, margin: bool, reference_extended: bool = False
    ) -> None:
        """Chosse between:
        margin (bool): no-margin / margin
        reference_extended(bool): reference-not extended / reference_extended."""
        return np.sum(
            np.array([margin, reference_extended]) * (2 ** np.arange(self.n_options))
        )

    def set(
        self,
        value: object,
        in_global_frame,
        index=None,
        has_moved=None,
        *args,
        **kwargs
    ) -> None:
        if index is None:
            index = self.transform_list_to_index(*args, **kwargs)

        if in_global_frame:
            if has_moved is None:
                # Make sure temp_state_ is updated
                self.check_if_pose_has_updated()

            self._hull_list_global[index] = value

        else:
            self._hull_list_local[index] = value

    def get(
        self, in_global_frame, has_moved=None, index=None, *args, **kwargs
    ) -> object:
        if index is None:
            index = self.transform_list_to_index(*args, **kwargs)

        if in_global_frame:
            if has_moved is None:
                has_moved = self.check_if_pose_has_updated()

            if not has_moved:
                value = self._hull_list_global[index]
                if value is not None:
                    # If in global is already defined, return it. Otherwise check local-list
                    return value

            value = self._hull_list_local[index]
            if value is None:
                # Cannot be transformed is none
                return value

            # Transform and store
            value = self.transform_relative2global(value)
            self._hull_list_global[index] = value

            return value

        else:
            # In local frame
            return self._hull_list_local[index]

    # TODO (maybe): iterative self.set / self.get, with automated indeces

    def transform_relative2global(self, shapely_object):
        """This is the a shapely-specific transformation."""

        if shapely_object is None:
            return None

        shapely_object = affinity.translate(
            shapely_object,
            xoff=self._state.center_position[0],
            yoff=self._state.center_position[1],
        )

        if self._state.orientation is not None:
            shapely_object = affinity.rotate(
                shapely_object, self._state.orientation, use_radians=True
            )

        return shapely_object

    def get_global_with_everything_as_array(self) -> np.ndarray:
        """points: np.ndarray of dimension (self.dimension, n_points)
        with the number of points dependent on the shape"""
        shapely_ = self.get_global_with_everything()
        return np.array(shapely_.exterior.coords.xy)

    def get_global_without_margin_as_array(self) -> np.ndarray:
        """points: np.ndarry of dimension (self.dimension, n_points)
        with the number of points dependent on the shape"""
        shapely_ = self.get_global_without_margin()
        breakpoint()
        return np.array(shapely_.exterior.coords.xy)

    def get_global_with_everything(self) -> object:
        """Shapely for global intersection-check."""
        # TODO in the future: replace
        has_moved = self.check_if_pose_has_updated()

        shapely_ = self.get(
            in_global_frame=True,
            margin=True,
            reference_extended=True,
            has_moved=has_moved,
        )

        if shapely_ is not None:
            return shapely_

        shapely_ = self.get(
            in_global_frame=True,
            margin=True,
            reference_extended=False,
            has_moved=has_moved,
        )

        if shapely_ is not None:
            return shapely_

        shapely_ = self.get(
            in_global_frame=True,
            margin=False,
            reference_extended=False,
            has_moved=has_moved,
        )

        return shapely_

    def get_global_without_margin(self):
        hull_ = self.get(
            in_global_frame=True,
            margin=False,
            reference_extended=False,
        )
        return hull_

    def get_local_with_margin_only(self):
        has_moved = self.check_if_pose_has_updated()

        hull_ = self.get(
            in_global_frame=True,
            margin=True,
            reference_extended=True,
            has_moved=has_moved,
        )
        if hull_ is not None:
            return hull_

        hull_ = self.get(
            in_global_frame=True,
            margin=True,
            reference_extended=False,
            has_moved=has_moved,
        )

        return hull_

    def set_local_with_everything(self, value, has_moved=None, reset_upper=False):
        """No reset is performed since this is the higher level."""
        if reset_upper:
            warnings.warn(
                "Reset upper is not performed since \n"
                + " we are at the top of the hirarchy."
            )

        if has_moved is None:
            has_moved = self.check_if_pose_has_updated()

        self.set(
            value=value,
            in_global_frame=False,
            margin=True,
            reference_extended=True,
            has_moved=has_moved,
        )

    def set_local_with_margin_only(self, value, has_moved=None, reset_upper=True):
        if has_moved is None:
            has_moved = self.check_if_pose_has_updated()

        hull_ = self.set(
            value=value,
            in_global_frame=False,
            margin=True,
            reference_extended=False,
            has_moved=has_moved,
        )

        if reset_upper:
            hull_ = self.set(
                value=value,
                in_global_frame=False,
                margin=True,
                reference_extended=True,
                has_moved=has_moved,
            )

    def set_global_without_margin(
        self, value, has_moved=None, reset_upper=True
    ) -> np.ndarray:

        if has_moved is None:
            has_moved = self.check_if_pose_has_updated()

        hull_ = self.set(
            value=value,
            in_global_frame=False,
            margin=False,
            reference_extended=False,
            has_moved=has_moved,
        )

        if reset_upper:
            hull_ = self.set(
                value=value,
                in_global_frame=False,
                margin=True,
                reference_extended=False,
                has_moved=has_moved,
            )

            hull_ = self.set(
                value=value,
                in_global_frame=False,
                margin=True,
                reference_extended=True,
                has_moved=has_moved,
            )

    def get_global(
        self, position: np.ndarray, orientation: float, *args, **kwargs
    ) -> object:
        """Transform 2D shapely object from local to global."""
        shapely_ = self.get(*args, **kwargs, in_global_frame=True)

        if shapely_ is None:
            shapely_ = self.get(*args, **kwargs, in_global_frame=False)

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
        shapely_ = self.get(in_global_frame=False, margin=True, reference_extended=True)

        if shapely_ is None:
            shapely_ = self.get(
                in_global_frame=False, margin=True, reference_extended=False
            )

        if shapely_.geom_type == "Polygon":
            xy_data = np.array(shapely_.exterior.coords).T
        else:
            xy_data = np.array(shapely_.coords).T

        return xy_data
