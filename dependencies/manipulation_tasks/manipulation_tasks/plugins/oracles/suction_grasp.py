import random
from manipulation_tasks.task import Task
from manipulation_tasks.scene import Scene, SimulatedScene
from manipulation_tasks.object import TargetObject, ManipulationObject
from manipulation_tasks.dataclasses import Action, Objective
from manipulation_tasks.transform import Affine
import manipulation_tasks.factory as factory
import numpy as np


class SuctionGraspOracle:
    gripper_offset: Affine
    selected_objective: Objective
    selected_object: ManipulationObject
    solution_executable: bool = False
    attention_symmetries: int = 2

    def __init__(self, gripper_offset):
        self.gripper_offset = Affine(**gripper_offset)

    def execute(self, action: Action, task: Task, scene: Scene = None):
        if not self.solution_executable:
            raise Exception('solution not executable')
        # if not test:
        self.selected_objective.completed = True
        if scene is not None:
            task.execute(action, scene)
            scene.remove_objects([self.selected_object.object_id])
        else:
            task.grasped_objects.append(self.selected_object)
        task.manipulation_objects.remove(self.selected_object)
        self.solution_executable = False

    def solve(self, task: Task):
        # select random objective
        unsolved_objectives = [o for o in task.objectives if not o.completed]
        self.selected_objective = random.sample(unsolved_objectives, 1)[0]
        # get manipulation object of objective
        self.selected_object = task.get_object_with_unique_id(self.selected_objective.object_unique_id)  # type: ignore

        # TODO attention instead of pick, maybe
        # get a gripper pose relative to object pose
        relative_pick_gripper_pose = self.selected_object.get_valid_poses()[0] * self.gripper_offset

        # compute global pick and place poses
        pick_pose = self.selected_object.pose * relative_pick_gripper_pose
        action = Action([pick_pose])

        solved = len(unsolved_objectives) - 1 <= 0
        self.solution_executable = True
        return action, solved

    def compute_attention_errors(self, task: Task, attention_pose: Affine):
        unsolved_objectives = [o for o in task.objectives if not o.completed]
        errors = []
        real_pose = attention_pose * self.gripper_offset.invert()
        for objective in unsolved_objectives:
            objective_object = task.get_object_with_unique_id(objective.object_unique_id)
            errors += objective_object.compute_pose_errors(real_pose, self.attention_symmetries)  # type: ignore

        sorted_errors = sorted(errors, key=lambda tup: tup[0])
        return sorted_errors

    def compute_transport_errors(self, task: Task, attention_pose: Affine, transport_pose: Affine):
        unsolved_objectives = [o for o in task.objectives if not o.completed]
        errors = []
        # TODO all objectives or just nearest objective?
        # TODO check whether attention pose is offset or not
        real_transport_pose = transport_pose * self.gripper_offset.invert()
        real_attention_pose = attention_pose * self.gripper_offset.invert()
        for objective in unsolved_objectives:
            objective_object = task.get_object_with_unique_id(objective.object_unique_id)
            relative_attention_pose = real_attention_pose / objective_object.pose

            # get possible target objects of objective
            targets = [task.get_object_with_unique_id(target_id) for target_id in objective.target_unique_ids]
            # select targets that are not occupied
            available_targets = [target for target in targets if not target.occupied]  # type: ignore
            for target in available_targets:
                # compute new object pose from the gripper pose and its relative pose to the object during manipulation
                object_pose = real_transport_pose * relative_attention_pose.invert()
                errors += target.compute_pose_errors(object_pose)  # type: ignore

        sorted_errors = sorted(errors, key=lambda tup: tup[0])

        return sorted_errors

    def compute_simulated_error(self, task: Task, attention_pose: Affine, scene: SimulatedScene):
        errors = []

        def distance_from_objective(objective: Objective):
            objective_object = task.get_object_with_unique_id(objective.object_unique_id)
            return np.linalg.norm(attention_pose.translation - objective_object.pose.translation)

        sorted_objectives = sorted(task.objectives, key=lambda objective: distance_from_objective(objective))
        selected_object = task.get_object_with_unique_id(sorted_objectives[0].object_unique_id)
        new_object_pose = scene.get_object_pose(selected_object.object_id)

        for target_object_id in sorted_objectives[0].target_unique_ids:
            target_object = task.get_object_with_unique_id(target_object_id)
            errors += target_object.compute_pose_errors(new_object_pose)  # type: ignore
        return errors


def register():
    factory.register_oracle('suction_grasp-oracle', SuctionGraspOracle)
