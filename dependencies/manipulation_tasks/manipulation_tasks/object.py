from manipulation_tasks.transform import Affine
from typing import List, Tuple, Protocol
import numpy as np


class SceneObject(Protocol):
    """
    Base protocol for objects that can be placed in a scene.

    Class variables:
    :var str urdf_path: path to the urdf describing the physical properties of the object
    :var int object_id: id of object from the simulation - if there is one
    :var bool static: indicates whether the object can be moved
    :var Affine pose: 6D pose of the object
    :var float min_dist: encompassing radius, for non-overlapping object placement
    :var Affine offset: offset of object origin and its base, to avoid placing object into the ground
    :var int unique_id: unique id of object that was generated while task generation. It is used in objectives.
    """
    urdf_path: str
    object_id: int
    static: bool
    pose: Affine
    min_dist: float
    offset: Affine
    unique_id: int


class ManipulationObject(SceneObject):
    """
    Protocol for objects that can be manipulated.

    For several objects, there are multiple valid gripper poses for a given manipulation primitive that would
    result in successful object manipulation. This protocol enables the retrieval of such poses and error computation
    to valid areas.
    """

    def get_valid_poses(self) -> List[Affine]:
        """
        This method returns a list of valid gripper poses relative to the object's pose.

        :return Affine: a list of valid gripper poses
        """
        ...

    def compute_pose_errors(self, gripper_pose: Affine, rotational_symmetries: int) -> List[Tuple[float, float]]:
        """
        This method computes and returns the translational and rotational errors to each valid gripper area
        (e.g. poses, lines, surfaces ...) given a gripper pose and the rotational symmetries of the gripper
        (2 for simple two-jaw parallel gripper).

        :param Affine gripper_pose: pose of the gripper
        :param int rotational_symmetries: rotational symmetries of a gripper
        :return List[Tuple[float, float]]: list of translational and rotational errors
        """
        ...


class TargetObject(SceneObject):
    """
    Protocol for objects that are used as target objects.


    For a given target object, there can be multiple valid target poses for a manipulation objet e.g. because of
    object symmetries. This protocol enables the retrieval of such poses and error computation to all valid areas.
    """
    occupied: bool

    def get_valid_poses(self) -> List[Affine]:
        """
        This method returns a list valid target poses for the object relative to the target object's pose.

        :return Affine: a list valid target poses
        """
        ...

    def compute_pose_errors(self, object_pose: Affine) -> List[Tuple[float, float]]:
        """
        This method computes and returns the translational and rotational errors to each valid
        target pose given a target object pose.

        :param Affine object_pose: target pose for the object
        :return List[Tuple[float, float]]:
        """
        ...


def is_overlapping(pose, min_dist, objects):
    for o in objects:
        d = np.linalg.norm(pose.translation[:2] - o.pose.translation[:2])
        overlap = d < (min_dist + o.min_dist)
        if overlap:
            return True
    return False
