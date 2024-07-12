from manipulation_tasks.transform import Affine
from typing import List, Dict, Any, Protocol
import numpy as np
from manipulation_tasks.object import SceneObject
# from manipulation_tasks.sensor import Camera


class Robot(Protocol):
    """
    Protocol for interacting with the robot.
    """

    def home(self) -> bool:
        """
        Resets the robot's joints to home position.
        """
        ...

    def ptp(self, pose: Affine) -> bool:
        """
        Point-to-point motion command to the given 6D cartesian pose.

        :param pose: target 6d pose
        """
        ...

    def lin(self, pose: Affine) -> bool:
        """
        Linear motion command to the given 6D cartesian pose.

        :param pose: target 6d pose
        """
        ...

    def open_gripper(self, **kwargs) -> bool:
        """
        Simple interface for opening the gripper. Arguments should be specified in the Robot and Protocol
        implementations.

        :param kwargs: dict of gripper command specifications.
        :return:
        """
        ...

    def close_gripper(self, **kwargs) -> bool:
        """
        Simple interface for closing the gripper. Arguments should be specified in the Robot and Protocol
        implementations.

        :param kwargs: dict of gripper command specifications.
        :return:
        """
        ...


class Scene(Protocol):
    """
    Protocol for interacting with a scene and its robot and cameras.

    Class variables:
    :var Robot robot: the robot in the scene
    :var List[Camera] cameras: list of cameras available in the scene
    :var np.array t_bounds: translational bounds for the workspace
    :var np.array t_bounds: rotational bounds for the workspace (it was never needed)
    """
    robot: Robot
    sensors: Dict[str, Dict[str, Any]]
    t_bounds: np.array
    r_bounds: np.array

    def get_observation(self, sensor_name: str, poses: List[Affine] = None) -> List[Dict[str, np.array]]:
        """
        The observations are organized into a dict for each camera, with the key referring to the image type,
        e.g. color. This method returns observations from each camera in a list of dicts.
        :return: List[Dict[str, np.array]]: all observations
        """
        ...

    def spawn_coordinate_frame(self, pose: Affine):
        """
        Debugging functionality: visualizes a coordinate frame at a given pose.
        :param pose: pose of the coordinate frame
        """
        ...

    def clean(self):
        """
        Debugging functionality: removes coordinate frame visualizations.
        """
        ...


class SimulatedScene(Scene):
    """
    Protocol for interacting with a simulated scene and its robot and cameras.
    """

    def add_object(self, o: SceneObject) -> int:
        """
        Add a SceneObject to the manipulation at the pose stored in the SceneObject.
        This method needs to set the object_id of the SceneObject.
        :param SceneObject o: the SceneObject
        :return: int: the id obtained from the simulation (NOT 'unique_ids')
        """
        ...

    def remove_objects(self, object_ids: List[int]):
        """
        Removes multiple objects from the simulation.
        :param List[int] object_ids: list of object ids (from the simulation, NOT 'unique_ids')
        """
        ...

    def shutdown(self):
        """
        Shuts down the simulation.
        """
        ...

    def get_object_pose(self, object_id: int) -> Affine:
        """
        Function that returns the Affine pose of an object, defined by its id.

        :param int object_id: id of the object
        :return: Affine: object pose
        """
        ...
