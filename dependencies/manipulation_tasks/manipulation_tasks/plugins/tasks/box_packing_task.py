import copy

from manipulation_tasks import loader, factory
from typing import List
import random
from manipulation_tasks.dataclasses import Objective
from manipulation_tasks.object import ManipulationObject, TargetObject, SceneObject
from manipulation_tasks.transform import Affine
import numpy as np
import os
from dataclasses import dataclass
import json

required_plugins = ['manipulation_tasks.plugins.tasks.simple_task',
                    'manipulation_tasks.plugins.objects.base']


class Reset(Exception):
    pass


class Block:
    def __init__(self, pose, dimensions):
        self.pose = pose
        self.dimensions = dimensions
        self.unique_id = None


class BoxPackingTaskFactory:
    def __init__(self, t_bounds, r_bounds, manipulation_type: str,
                 primitive_type: str, target_type: str = None,
                 box_template_urdf: str = None, block_template_path: str = None):
        self.t_bounds = t_bounds
        self.r_bounds = r_bounds
        self.manipulation_type = manipulation_type
        self.primitive_type = primitive_type
        self.target_type = target_type

        self.box_template_urdf = box_template_urdf
        self.block_template_path = block_template_path

        self.unique_id_counter = 0

        self.box_size_bounds = np.array([[0.05, 0.2], [0.05, 0.2]])

        self.max_pose_tries = 2000
        self.max_create_tries = 10
        self.min_object_dim = 0.04

    def get_unique_id(self):
        self.unique_id_counter += 1
        return self.unique_id_counter - 1

    def create_task(self):
        # TODO remove temp urdfs
        task_setup = False
        create_tries = 0
        while not task_setup and create_tries < self.max_create_tries:
            self.unique_id_counter = 0
            objectives = []
            try:
                box_block = self.generate_box()

                target_blocks = []

                def kd_tree(block: Block):
                    block.dimensions[2] = 0.05
                    split = block.dimensions[:2] > 2 * self.min_object_dim
                    if not split.any():
                        target_blocks.append(block)
                        return
                    split_axes = np.where(split)[0]
                    split_axis = np.random.choice(split_axes, 1)[0]

                    cut_pos = np.random.rand() * (
                            block.dimensions[split_axis] - 2 * self.min_object_dim) + self.min_object_dim

                    child_a_dim = copy.deepcopy(block.dimensions)
                    child_a_dim[split_axis] = cut_pos

                    child_a_position = copy.deepcopy(block.pose).translation
                    child_a_position[split_axis] = child_a_position[split_axis] - block.dimensions[split_axis] / 2 + \
                                                   child_a_dim[split_axis] / 2
                    child_a_pose = Affine(translation=child_a_position, rotation=block.pose.rotation)

                    child_a = Block(child_a_pose, child_a_dim)

                    child_b_dim = copy.deepcopy(block.dimensions)
                    child_b_dim[split_axis] = block.dimensions[split_axis] - cut_pos

                    child_b_position = copy.deepcopy(block.pose).translation
                    child_b_position[split_axis] = child_b_position[split_axis] + block.dimensions[split_axis] / 2 - \
                                                   child_b_dim[split_axis] / 2
                    child_b_pose = Affine(translation=child_a_position, rotation=block.pose.rotation)

                    child_b = Block(child_b_pose, child_b_dim)

                    kd_tree(child_a)
                    kd_tree(child_b)

                kd_tree(copy.deepcopy(box_block))
                pick_blocks = []

                for t in target_blocks:
                    t.unique_id = self.get_unique_id()
                    new_block = self.get_non_overlapping_block(t.dimensions, pick_blocks + [box_block])
                    new_block.unique_id = self.get_unique_id()
                    pick_blocks.append(new_block)
                    objective = Objective(object_unique_id=new_block.unique_id, target_unique_ids=[t.unique_id])
                    objectives.append(objective)

                task_setup = True
            except Reset:
                create_tries += 1
        if not task_setup:
            raise Exception('Objects always overlap. Try to reduce number of objects in task.')

        return BoxPackingTask(objectives, pick_blocks, target_blocks, box_block,
                              self.box_template_urdf, self.primitive_type, self.block_template_path)

    def generate_box(self):
        width = random.uniform(self.box_size_bounds[0, 0], self.box_size_bounds[0, 1])
        length = random.uniform(self.box_size_bounds[1, 0], self.box_size_bounds[1, 1])
        height = 0.002
        container_size = np.array([width, length, height])
        min_dist = BoxPackingTaskFactory.min_dist(container_size)
        new_t_bounds, min_dist = self.new_bounds_for_dims(container_size, min_dist)
        random_pose = Affine.random(t_bounds=new_t_bounds, r_bounds=self.r_bounds)
        return Block(random_pose, container_size)

    @staticmethod
    def min_dist(dimensions):
        min_dist = np.sqrt((dimensions[0] / 2) ** 2 + (dimensions[1] / 2) ** 2)
        return min_dist

    def new_bounds_for_dims(self, dimensions, min_dist):
        new_t_bounds = np.array(self.t_bounds)
        new_t_bounds[:2, 0] = new_t_bounds[:2, 0] + min_dist
        new_t_bounds[:2, 1] = new_t_bounds[:2, 1] - min_dist
        new_t_bounds[2, :] = dimensions[2] / 2
        print(new_t_bounds)
        return new_t_bounds, min_dist

    def get_non_overlapping_block(self, dimensions, objects):
        overlapping = True
        # TODO timeout
        min_dist = BoxPackingTaskFactory.min_dist(dimensions)
        new_t_bounds, min_dist = self.new_bounds_for_dims(dimensions, min_dist)
        tries = 0
        while overlapping and tries < self.max_pose_tries:
            random_pose = Affine.random(t_bounds=new_t_bounds, r_bounds=self.r_bounds)
            overlapping = BoxPackingTaskFactory.is_overlapping(random_pose, min_dist, objects)
            tries += 1
            if not overlapping:
                return Block(random_pose, dimensions)
        raise Reset

    @staticmethod
    def is_overlapping(pose, min_dist, objects):
        for o in objects:
            d = np.linalg.norm(pose.translation[:2] - o.pose.translation[:2])
            o_min_dis = BoxPackingTaskFactory.min_dist(o.dimensions)
            overlap = d < (min_dist + o_min_dis)
            if overlap:
                return True
        return False


class BoxPackingTask:
    def __init__(self, objectives: List[Objective], manipulation_blocks: List[Block],
                 target_blocks: List[Block], box_block: Block,
                 box_template_urdf: str, primitive_type: str, block_template_path: str):
        self.primitive_type = primitive_type
        primitive_args = {
            'type': self.primitive_type
        }
        self.primitive = factory.create_primitive(primitive_args)
        self.objectives = objectives
        self.manipulation_blocks = manipulation_blocks
        self.target_blocks = target_blocks

        self.box_block = box_block
        self.box_template_urdf = box_template_urdf
        self.block_template_path = block_template_path

        self.temp_files = []
        self.manipulation_objects = []
        self.target_objects = []
        self.box = self.create_box()
        self.create_blocks()

    def create_box(self):
        wall_offsets = self.box_block.dimensions / 2
        f_data, f_name = BoxPackingTask.fill_template(self.box_template_urdf,
                                                      replace={'DIM': self.box_block.dimensions[:2],
                                                               'HALF': wall_offsets[:2]})
        f_name = f'{f_name}_temp.urdf'
        with open(f_name, 'w') as file:
            file.write(f_data)

        self.temp_files.append(f_name)
        box_min_dist = np.sqrt(wall_offsets[0] ** 2 + wall_offsets[1] ** 2)

        object_args = {
            'urdf_path': f_name,
            'pose': self.box_block.pose,
            'min_dist': box_min_dist
        }
        box = factory.create_object('scene-object', object_args)
        return box

    def create_blocks(self):
        for m_b in self.manipulation_blocks:
            pose_offset = m_b.dimensions / 2
            urdf_data, urdf_name = BoxPackingTask.fill_template(f'{self.block_template_path}/object.urdf',
                                                                replace={'DIM': m_b.dimensions[:2]})
            urdf_name = f'{urdf_name}_temp.urdf'
            with open(urdf_name, 'w') as file:
                file.write(urdf_data)
            self.temp_files.append(urdf_name)

            grasp_dims = (m_b.dimensions[:2] - 0.025) / 2
            conf_data, conf_name = BoxPackingTask.fill_template(f'{self.block_template_path}/pick_config.json',
                                                                replace={'DIM': grasp_dims})
            conf_name = f'{conf_name}_temp.json'
            with open(conf_name, 'w') as file:
                file.write(conf_data)
            self.temp_files.append(conf_name)

            object_args = {
                'urdf_path': urdf_name,
                'pose': m_b.pose,
                'unique_id': m_b.unique_id,
            }

            with open(conf_name) as json_file:
                additional_args = json.load(json_file)
            offset = Affine(**additional_args['offset'])
            additional_args['offset'] = offset
            object_args.update(additional_args)

            min_dist = np.sqrt(pose_offset[0] ** 2 + pose_offset[1] ** 2)
            object_args['min_dist'] = min_dist

            block = factory.create_object('pick', object_args)
            self.manipulation_objects.append(block)

        for t_b in self.target_blocks:
            object_args = {
                'urdf_path': None,
                'pose': t_b.pose,
                'unique_id': t_b.unique_id,
            }
            with open(f'{self.block_template_path}/pose-target_config.json') as json_file:
                additional_args = json.load(json_file)
            offset = Affine(**additional_args['offset'])
            additional_args['offset'] = offset

            object_args.update(additional_args)

            block = factory.create_object('pose-target', object_args)
            self.target_objects.append(block)

    def get_info(self):
        info = {
            'objectives': self.objectives,
            'manipulation_blocks': self.manipulation_blocks,
            'target_blocks': self.target_blocks,
            'primitive_type': self.primitive_type,
            'box_block': self.box_block,
            'type': 'box-packing-task',
            'box_template_urdf': self.box_template_urdf,
            'block_template_path': self.block_template_path,
        }
        return info

    @staticmethod
    def fill_template(template_path, replace, filename=None):
        """Read a file and replace key strings."""
        with open(template_path, 'r') as file:
            f_data = file.read()
        for field in replace:
            for i in range(len(replace[field])):
                f_data = f_data.replace(f'{field}{i}', str(replace[field][i]))
        if not filename:
            template_filename = os.path.split(template_path)[-1].split(".")[0]
            f_name = os.path.join(os.path.split(template_path)[0],
                                  f'{template_filename}_{str(random.random()).split(".")[1]}')
        else:
            f_name = filename
        return f_data, f_name

    def execute(self, action, scene):
        self.primitive.execute(action, scene)

    def get_object_with_unique_id(self, unique_id: int):
        for o in self.manipulation_objects + self.target_objects + [self.box]:
            if o.unique_id == unique_id:
                return o
        raise RuntimeError('object id mismatch')

    def setup(self, scene):
        scene.robot.home()
        scene.add_object(self.box)
        for o in self.manipulation_objects:
            scene.add_object(o)

        for o in self.target_objects:
            if o.urdf_path is not None:
                scene.add_object(o)

    def clean(self, scene):
        object_ids = [o.object_id for o in self.manipulation_objects]
        object_ids.append(self.box.object_id)
        scene.remove_objects(object_ids)
        for f in self.temp_files:
            os.remove(f)


def register():
    loader.load_plugins(required_plugins)
    factory.register_task_factory('box-packing-task-factory', BoxPackingTaskFactory)
    factory.register_task('box-packing-task', BoxPackingTask)
