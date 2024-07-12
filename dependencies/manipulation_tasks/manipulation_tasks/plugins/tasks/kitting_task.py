from manipulation_tasks import loader, factory
from typing import List
import random
from manipulation_tasks.dataclasses import Objective
from manipulation_tasks.object import ManipulationObject, TargetObject, SceneObject, is_overlapping
from manipulation_tasks.transform import Affine
import numpy as np

required_plugins = ['manipulation_tasks.plugins.tasks.simple_task']


class Reset(Exception):
    pass


class KittingTaskFactory:
    def __init__(self, t_bounds, r_bounds, object_types: List[str], manipulation_type: str,
                 primitive_type: str, target_object_type: str = None, target_type: str = None,
                 kitting_board_urdf: str = None):
        self.t_bounds = t_bounds
        self.r_bounds = r_bounds
        self.object_types = object_types
        self.n_objects = 5  # TODO consider making this variable --> think about positioning
        self.manipulation_type = manipulation_type
        self.primitive_type = primitive_type
        self.target_object_type = target_object_type
        self.target_type = target_type

        # TODO what about kitting table dimensions?
        self.kitting_board_urdf = kitting_board_urdf
        self.kitting_board_dimensions = np.array([0.37, 0.235, 0.014])
        self.kitting_board_min_dist = np.sqrt(
            (self.kitting_board_dimensions[0] / 2) ** 2 + (self.kitting_board_dimensions[1] / 2) ** 2)

        self.unique_id_counter = 0

        self.relative_target_object_positions = [[-0.12, -0.0525, 0.007],
                                                 [0, -0.0525, 0.007],
                                                 [0.12, -0.0525, 0.007],
                                                 [-0.06, 0.0525, 0.007],
                                                 [0.06, 0.0525, 0.007]]

        self.max_pose_tries = 2000
        self.max_create_tries = 10

    def get_unique_id(self):
        self.unique_id_counter += 1
        return self.unique_id_counter - 1

    def create_task(self):
        task_setup = False
        create_tries = 0
        while not task_setup and create_tries < self.max_create_tries:
            self.unique_id_counter = 0
            objective_object_types = random.choices(self.object_types, k=self.n_objects)
            different_object_types = set(objective_object_types)
            sorted_object_types = {
                object_type: objective_object_types.count(object_type) for object_type in different_object_types
            }
            objectives = []
            manipulation_objects = []
            target_objects = []
            pos_ids = list(range(self.n_objects))
            try:
                random.shuffle(pos_ids)
                kitting_board = self.generate_kitting_board()
                pos_idx = 0

                for object_type, count in sorted_object_types.items():
                    object_unique_id = []
                    target_unique_ids = []
                    for _ in range(count):
                        manipulation_object = self.generate_manipulation_object(object_type,
                                                                                manipulation_objects + [kitting_board])
                        manipulation_objects.append(manipulation_object)
                        object_unique_id.append(manipulation_object.unique_id)

                        target_object = self.generate_target_object(object_type, pos_idx, kitting_board.pose)
                        pos_idx += 1
                        target_objects.append(target_object)
                        target_unique_ids.append(target_object.unique_id)

                    for object_id in object_unique_id:
                        objective = Objective(completed=False, object_unique_id=object_id,
                                              target_unique_ids=target_unique_ids)
                        objectives.append(objective)
                task_setup = True
            except Reset:
                create_tries += 1
        if not task_setup:
            raise Exception('Objects always overlap. Try to reduce number of objects in task.')

        return KittingTask(objectives, manipulation_objects, target_objects, kitting_board, self.primitive_type)

    def generate_kitting_board(self):
        new_t_bounds = np.array(self.t_bounds)
        new_t_bounds[:2, 0] = new_t_bounds[:2, 0] + self.kitting_board_min_dist
        new_t_bounds[:2, 1] = new_t_bounds[:2, 1] - self.kitting_board_min_dist
        new_t_bounds[2, :] = self.kitting_board_dimensions[2] / 2
        random_pose = Affine.random(t_bounds=new_t_bounds, r_bounds=self.r_bounds)
        object_args = {
            'urdf_path': self.kitting_board_urdf,
            'pose': random_pose,
            'min_dist': self.kitting_board_min_dist
        }
        kitting_board = factory.create_object('scene-object', object_args)
        return kitting_board

    def generate_manipulation_object(self, object_type, added_objects):
        manipulation_object = factory.create_manipulation_object(object_type, self.manipulation_type)
        object_pose = self.get_non_overlapping_pose(manipulation_object.min_dist, added_objects)
        manipulation_object.pose = manipulation_object.offset * object_pose
        manipulation_object.unique_id = self.get_unique_id()
        return manipulation_object

    def generate_target_object(self, object_type, target_index, kitting_board_pose):
        target_object = factory.create_target_object(object_type, self.target_object_type, self.target_type)
        theta = random.uniform(0, 2 * np.pi)
        pose = Affine(translation=self.relative_target_object_positions[target_index], rotation=[0, 0, theta])
        target_object.pose = kitting_board_pose * pose
        target_object.unique_id = self.get_unique_id()
        return target_object

    def get_non_overlapping_pose(self, min_dist, objects):
        overlapping = True
        # TODO timeout
        new_t_bounds = np.array(self.t_bounds)
        new_t_bounds[:2, 0] = new_t_bounds[:2, 0] + min_dist
        new_t_bounds[:2, 1] = new_t_bounds[:2, 1] - min_dist
        tries = 0
        while overlapping and tries < self.max_pose_tries:
            random_pose = Affine.random(t_bounds=new_t_bounds, r_bounds=self.r_bounds)
            overlapping = is_overlapping(random_pose, min_dist, objects)
            tries += 1
            if not overlapping:
                return random_pose
        raise Reset


class KittingTask:
    def __init__(self, objectives: List[Objective], manipulation_objects: List[ManipulationObject],
                 target_objects: List[TargetObject], kitting_board: SceneObject, primitive_type: str):
        self.primitive_type = primitive_type
        primitive_args = {
            'type': self.primitive_type
        }
        self.primitive = factory.create_primitive(primitive_args)
        self.objectives = objectives
        self.manipulation_objects = manipulation_objects
        self.target_objects = target_objects

        self.kitting_board = kitting_board

    def get_info(self):
        info = {
            'objectives': self.objectives,
            'manipulation_objects': self.manipulation_objects,
            'target_objects': self.target_objects,
            'primitive_type': self.primitive_type,
            'kitting_board': self.kitting_board,
            'type': 'kitting-task'
        }
        return info

    def execute(self, action, scene):
        self.primitive.execute(action, scene)

    def get_object_with_unique_id(self, unique_id: int):
        for o in self.manipulation_objects + self.target_objects + [self.kitting_board]:
            if o.unique_id == unique_id:
                return o
        raise RuntimeError('object id mismatch')

    def setup(self, scene):
        scene.robot.home()
        scene.add_object(self.kitting_board)
        for o in self.manipulation_objects:
            scene.add_object(o)

        for o in self.target_objects:
            if o.urdf_path is not None:
                scene.add_object(o)

    def clean(self, scene):
        object_ids = [o.object_id for o in self.manipulation_objects]
        object_ids += [o.object_id for o in self.target_objects]
        object_ids.append(self.kitting_board.object_id)
        scene.remove_objects(object_ids)


def register():
    loader.load_plugins(required_plugins)
    factory.register_task_factory('kitting-task-factory', KittingTaskFactory)
    factory.register_task('kitting-task', KittingTask)
