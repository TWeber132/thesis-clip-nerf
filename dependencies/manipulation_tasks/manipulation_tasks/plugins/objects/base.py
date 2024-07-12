import random
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple
import manipulation_tasks.factory as factory
from manipulation_tasks.geometric_utils import project_point_on_plane, triangle_area
from manipulation_tasks.transform import Affine
import numpy as np
from manipulation_tasks.transform_utils.random import sample_pose_from_segment, sample_pose_from_rectangle
from manipulation_tasks.transform_utils.differences import rotation_to_line_difference, point_to_segment_distance, \
    transformation_difference

import cv2
import matplotlib.pyplot as plt

@dataclass
class SceneObject:
    """
    Base class for objects that can be placed in a scene.

    Class variables:
    :var str urdf_path: path to the urdf describing the physical properties of the object
    :var int object_id: id of object from the simulation - if there is one
    :var bool static: indicates whether the object can be moved
    :var Affine pose: 6D pose of the object
    :var float min_dist: encompassing radius, for non-overlapping object placement
    :var Affine offset: offset of object origin and its base, to avoid placing object into the ground
    :var int unique_id: unique id of object that was generated while task generation. It is used in objectives.
    """
    urdf_path: str = None
    object_id: int = -1
    static: bool = True
    pose: Affine = Affine()
    min_dist: float = 0
    offset: Affine = Affine()
    unique_id: int = -1


@dataclass
class PickObject(SceneObject):
    """
    Class for objects that can be picked. A pick configuration is required.

    For several objects, there are multiple valid gripper poses for a successful pick execution. In this case we
    restrict ourselves to planar pick actions with a 2-jaw parallel gripper. This reduces the possible pick areas
    to points and segments. We have only implemented segments, because a segment with identical endpoints
    represents a point.
    """
    static: bool = False
    pick_config: List[Dict[str, Any]] = field(default_factory=lambda: [])

    def get_valid_poses(self) -> List[Affine]:
        """
        This method samples and returns a valid gripper pose relative to the object's pose, based on the segments
        defined in the pick configuration.

        :return Affine: valid relative gripper pose
        """
        grasp_area = random.sample(self.pick_config, 1)[0]

        valid_pose = None

        # TODO match case and upgrade to python 3.10
        # TODO change 'line' to segment
        if grasp_area['type'] == 'segment':
            point_a = Affine(translation=grasp_area['point_a'])
            point_b = Affine(translation=grasp_area['point_b'])
            valid_pose = sample_pose_from_segment(point_a, point_b)
        if grasp_area['type'] == 'rectangle':
            point_a = Affine(translation=grasp_area['point_a'])
            point_b = Affine(translation=grasp_area['point_b'])
            point_c = Affine(translation=grasp_area['point_c'])
            point_d = Affine(translation=grasp_area['point_d'])
            valid_pose = sample_pose_from_rectangle(point_a, point_b, point_c, point_d)

        if valid_pose is None:
            raise Exception(f'No valid pose found for pick object {self}')
        return [valid_pose]

    def compute_pose_errors(self, gripper_pose: Affine, rotational_symmetries: int = 1) -> List[Tuple[float, float]]:
        """
        Computes translational and rotational errors to each segment defined in the pick configuration.
        Grippers that are rotationally invariant are not supported currently.
         
        :param gripper_pose: pose of the gripper
        :param rotational_symmetries: rotational symmetries of a gripper
        :return: list of translational and rotational errors
        """
        errors = []
        for grasp_area in self.pick_config:
            if grasp_area['type'] == 'segment':
                point_a = self.pose * Affine(translation=grasp_area['point_a'])
                point_b = self.pose * Affine(translation=grasp_area['point_b'])

                diff = abs(point_a.translation[2] - point_b.translation[2])
                # assert diff < 1e-10, f'Something went wrong. Horizontal lines only for pick config.'

                x_dir = np.array(point_a.translation) - np.array(point_b.translation)

                x_axis = x_dir / np.linalg.norm(x_dir)
                z_axis = np.array([0.0, 0.0, 1.0])
                y_axis = np.cross(z_axis, x_axis)

                rotation = np.vstack([x_axis, y_axis, z_axis]).T

                point_a = Affine(translation=point_a.translation, rotation=rotation)
                point_b = Affine(translation=point_b.translation, rotation=rotation)

                diff = abs(gripper_pose.translation[2] - point_b.translation[2])
                # assert diff < 1e-10, f'Something went wrong. z diff of the two poses: {diff}'
                t_error = point_to_segment_distance(gripper_pose.translation, point_a.translation,
                                                    point_b.translation)
                r_error, _ = rotation_to_line_difference(gripper_pose.quat, point_a.translation,
                                                         point_b.translation)

                max_rotation = 2 * np.pi / rotational_symmetries
                rotation_range = [-max_rotation / 2, max_rotation / 2]

                while r_error < -rotation_range[0]:
                    r_error += max_rotation
                while r_error >= rotation_range[1]:
                    r_error -= max_rotation

                errors.append((t_error, abs(r_error)))
            if grasp_area['type'] == 'rectangle':
                # print(self.pick_config)
                point_a = self.pose * Affine(translation=grasp_area['point_a'])
                point_b = self.pose * Affine(translation=grasp_area['point_b'])
                point_c = self.pose * Affine(translation=grasp_area['point_c'])
                point_d = self.pose * Affine(translation=grasp_area['point_d'])

                # print(point_a.translation, point_b.translation, point_c.translation, point_d.translation)

                # assume points are in the same plane
                diff = abs(point_a.translation[2] - point_b.translation[2])
                assert diff < 1e-10, f'Something went wrong. Horizontal lines only for pick config.'
                diff = abs(point_a.translation[2] - point_c.translation[2])
                assert diff < 1e-10, f'Something went wrong. Horizontal lines only for pick config.'
                diff = abs(point_a.translation[2] - point_d.translation[2])
                assert diff < 1e-10, f'Something went wrong. Horizontal lines only for pick config.'

                plane_normal = np.array([0.0, 0.0, 1.0])
                projection, distance = project_point_on_plane(gripper_pose.translation, point_a.translation,
                                                              plane_normal)

                # plot rectangle and projection

                # print("distance", distance)
                # assume rectangle is convex
                # compute area of triangles formed by the projection and the rectangle's points
                t_area = triangle_area(projection, point_a.translation, point_b.translation)
                t_area += triangle_area(projection, point_b.translation, point_c.translation)
                t_area += triangle_area(projection, point_c.translation, point_d.translation)
                t_area += triangle_area(projection, point_d.translation, point_a.translation)

                # print("triangle_area", t_area)
                # compute area of the rectangle
                r_area = triangle_area(point_a.translation, point_b.translation, point_c.translation)
                r_area += triangle_area(point_a.translation, point_c.translation, point_d.translation)
                # print("rectangle_area", r_area)
                # if the projection is inside the rectangle, the triangle area should not be greater than the
                # rectangle area, otherwise the projection is outside the rectangle
                # print(t_area, r_area)
                if abs(t_area - r_area) <= 0.00003:
                    t_error = abs(distance)
                else:
                    t_error = min(
                        point_to_segment_distance(gripper_pose.translation, point_a.translation, point_b.translation),
                        point_to_segment_distance(gripper_pose.translation, point_b.translation, point_c.translation),
                        point_to_segment_distance(gripper_pose.translation, point_c.translation, point_d.translation),
                        point_to_segment_distance(gripper_pose.translation, point_d.translation, point_a.translation))
                    # print("distances",
                    #       point_to_segment_distance(gripper_pose.translation, point_a.translation, point_b.translation),
                    #       point_to_segment_distance(gripper_pose.translation, point_b.translation, point_c.translation),
                    #       point_to_segment_distance(gripper_pose.translation, point_c.translation, point_d.translation),
                    #       point_to_segment_distance(gripper_pose.translation, point_d.translation, point_a.translation))
                # print(t_error)
                # print('---' * 30)
                z_axis = gripper_pose.rotation @ np.array([0.0, 0.0, 1.0])
                # determine the angle from the z axis to the plane normal
                # compute cos
                cos = np.dot(z_axis, plane_normal)
                # compute sin
                sin = np.linalg.norm(np.cross(z_axis, plane_normal))
                # compute angle
                r_error = np.arctan2(sin, cos)
                # print("error", t_error, r_error, flush=True)
                errors.append((t_error, abs(r_error)))

        #         p_a = np.array(point_a.translation[:2])
        #         p_b = np.array(point_b.translation[:2])
        #         p_c = np.array(point_c.translation[:2])
        #         p_d = np.array(point_d.translation[:2])
        #         p_p = np.array(projection[:2])
        #
        #         points = np.vstack([p_a, p_b, p_c, p_d, p_p])
        #         xs = points[:, 0]
        #         ys = points[:, 1]
        #         plt.plot(xs, ys, 'o')
        # plt.show()
        # print("=" * 50)
        # print(errors)
        # print("=" * 50)
        return errors


@dataclass
class PoseTargetObject(SceneObject):
    """
    Class for target objects that have to be placed precisely. A pose target configuration is required.

    For a given target object, there can be multiple valid target poses for a manipulation objet e.g. because of
    object symmetries. In this case we restrict ourselves to valid poses instead of areas or segments.
    """
    pose_target_config: Any = None
    occupied: bool = False

    def get_valid_poses(self) -> List[Affine]:
        """
        This method samples a valid target pose for the object relative to the target object's pose, based on the
        poses defined in the pose target configuration.

        :return Affine: valid relative target pose for the object
        """
        place_pose = random.sample(self.pose_target_config, 1)[0]
        option = Affine(**place_pose)
        return [option]

    def compute_pose_errors(self, object_pose: Affine) -> List[Tuple[float, float]]:
        """
        This method computes and returns the translational and rotational errors to each valid
        target pose in the pose target configuration given a target object pose.

        :param Affine object_pose: target pose for the object
        :return List[Tuple[float, float]]:
        """
        errors = []
        options = [Affine(**option) for option in self.pose_target_config]

        for option in options:
            target_pose = self.pose * option
            # diff = abs(target_pose.translation[2] - object_pose.translation[2])
            # assert diff < 1e-4, f'Something went wrong. z diff of the two poses: {diff}'
            t_error, r_error = transformation_difference(object_pose, target_pose)
            errors.append((t_error, r_error))
        return errors


def register() -> None:
    factory.register_object('scene-object', SceneObject)
    factory.register_object('pick', PickObject)
    factory.register_object('pose-target', PoseTargetObject)
    factory.register_object('shadow-target', PoseTargetObject)
