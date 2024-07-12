from manipulation_tasks import factory
from typing import List
import random
from manipulation_tasks.dataclasses import Objective
from manipulation_tasks.object import ManipulationObject, TargetObject, is_overlapping
from manipulation_tasks.transform import Affine
import numpy as np


class SimpleTaskFactory:
    def __init__(self, t_bounds, r_bounds, object_types: List[str], n_objects: int, manipulation_type: str,
                 primitive_type: str, target_object_type: str = None, target_type: str = None):
        self.t_bounds = t_bounds
        self.r_bounds = r_bounds
        self.object_types = object_types
        self.n_objects = n_objects
        self.manipulation_type = manipulation_type
        self.primitive_type = primitive_type
        self.target_object_type = target_object_type
        self.target_type = target_type

        self.unique_id_counter = 0

    def get_unique_id(self):
        self.unique_id_counter += 1
        return self.unique_id_counter - 1

    def create_task(self):
        self.unique_id_counter = 0
        objective_object_types = random.choices(self.object_types, k=self.n_objects)
        different_object_types = set(objective_object_types)
        sorted_object_types = {
            object_type: objective_object_types.count(object_type) for object_type in different_object_types
        }
        objectives = []
        manipulation_objects = []
        target_objects = []
        for object_type, count in sorted_object_types.items():
            object_unique_id = []
            target_unique_ids = []
            for _ in range(count):
                manipulation_object = self.generate_manipulation_object(object_type,
                                                                        manipulation_objects + target_objects)
                manipulation_objects.append(manipulation_object)
                object_unique_id.append(manipulation_object.unique_id)

                target_object = self.generate_target_object(object_type, manipulation_objects + target_objects)
                target_objects.append(target_object)
                target_unique_ids.append(target_object.unique_id)
            for object_id in object_unique_id:
                objective = Objective(completed=False, object_unique_id=object_id, target_unique_ids=target_unique_ids)
                objectives.append(objective)

        return SimpleTask(objectives, manipulation_objects, target_objects, self.primitive_type)

    def generate_manipulation_object(self, object_type, added_objects):
        manipulation_object = factory.create_manipulation_object(object_type, self.manipulation_type)
        object_pose = self.get_non_overlapping_pose(manipulation_object.min_dist, added_objects)
        manipulation_object.pose = manipulation_object.offset * object_pose
        manipulation_object.unique_id = self.get_unique_id()
        return manipulation_object

    def generate_target_object(self, object_type, added_objects):
        target_object = factory.create_target_object(object_type, self.target_object_type, self.target_type)
        target_pose = self.get_non_overlapping_pose(target_object.min_dist, added_objects)
        target_object.pose = target_object.offset * target_pose
        target_object.unique_id = self.get_unique_id()
        return target_object

    def get_non_overlapping_pose(self, min_dist, objects):
        overlapping = True
        # TODO timeout
        new_t_bounds = np.array(self.t_bounds)
        new_t_bounds[:2, 0] = new_t_bounds[:2, 0] + min_dist
        new_t_bounds[:2, 1] = new_t_bounds[:2, 1] - min_dist
        while overlapping:
            random_pose = Affine.random(t_bounds=new_t_bounds, r_bounds=self.r_bounds)
            overlapping = is_overlapping(random_pose, min_dist, objects)
        return random_pose


class SimpleTask:
    def __init__(self, objectives: List[Objective], manipulation_objects: List[ManipulationObject],
                 target_objects: List[TargetObject], primitive_type: str):
        self.primitive_type = primitive_type
        primitive_args = {
            'type': self.primitive_type
        }
        self.primitive = factory.create_primitive(primitive_args)
        self.objectives = objectives
        self.manipulation_objects = manipulation_objects
        self.target_objects = target_objects

    def get_info(self):
        info = {
            'objectives': self.objectives,
            'manipulation_objects': self.manipulation_objects,
            'target_objects': self.target_objects,
            'primitive_type': self.primitive_type,
            'type': 'simple-task'
        }
        return info

    def execute(self, action, scene):
        self.primitive.execute(action, scene)

    def get_object_with_unique_id(self, unique_id: int):
        for o in self.manipulation_objects + self.target_objects:
            if o.unique_id == unique_id:
                return o
        raise RuntimeError('object id mismatch')

    def setup(self, scene):
        scene.robot.home()
        for o in self.manipulation_objects:
            scene.add_object(o)

        for o in self.target_objects:
            if o.urdf_path is not None:
                scene.add_object(o)

    def clean(self, scene):
        object_ids = [o.object_id for o in self.manipulation_objects]
        object_ids += [o.object_id for o in self.target_objects]
        scene.remove_objects(object_ids)


def register():
    factory.register_task_factory('simple-task-factory', SimpleTaskFactory)
    factory.register_task('simple-task', SimpleTask)
