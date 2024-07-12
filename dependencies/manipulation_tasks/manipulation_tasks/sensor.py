import numpy as np

from manipulation_tasks.transform import Affine
from typing import Dict, Protocol, Any


class Sensor(Protocol):
    """
    Class containing information about the sensors used in the scene.

    Class variables:
    :var Affine pose: 6D pose of the sensor
    """

    pose: Affine

    def get_observation(self) -> Dict[str, np.array]:
        ...

    def get_config(self) -> Dict[str, Any]:
        ...

# class Camera(Sensor):
#     """
#     Class containing information about the cameras used in the scene.
#
#     Class variables:
#     :var Affine pose: 6D pose of the camera
#     :var Tuple[int, int] resolution: resolution of the camera
#     :var Tuple[float, float, float, float, float, float, float, float, float] intrinsics: flattened intrinsic matrix
#     :var Tuple[float, float] depth_range: depth range of the camera
#     """
#     pose: Affine
#     resolution: Tuple[int, int]
#     intrinsics = Tuple[float, float, float, float, float, float, float, float, float]
#     depth_range = Tuple[float, float]
#
#     def get_config(self) -> Dict[str, Union[
#         Affine, Tuple[int, int], Tuple[float, float, float, float, float, float, float, float, float], Tuple[
#             float, float]]]:
#         """
#         Returns the camera's configuration.
#
#         :return:  Dict[str, Union[Affine,
#                                   Tuple[int, int],
#                                   Tuple[float, float, float, float, float, float, float, float, float],
#                                   Tuple[float, float]]]: a dict containing the camera's pose, resolution, flattended
#                                                          intrinsic matrix and depth range
#         """
#         config = {
#             'pose': self.pose,
#             'resolution': self.resolution,
#             'intrinsics': self.intrinsics,
#             'depth_range': self.depth_range
#         }
#         return config
