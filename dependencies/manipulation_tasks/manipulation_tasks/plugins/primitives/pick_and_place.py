from manipulation_tasks.scene import Scene
from manipulation_tasks.dataclasses import Action
from manipulation_tasks import factory
from manipulation_tasks.transform import Affine
from typing import Optional


class Pick:
    def __init__(self,
                 pre_grasp_offset: Affine = Affine(translation=[0, 0, 0.075]),
                 post_grasp_offset: Optional[Affine] = None):
        self.pre_grasp_offset = pre_grasp_offset
        if post_grasp_offset is None:
            self.post_grasp_offset = self.pre_grasp_offset
        else:
            self.post_grasp_offset = post_grasp_offset

    def execute(self, action: Action, scene: Scene):
        pre_grasp_pose = self.pre_grasp_offset * action[0]
        scene.robot.ptp(pre_grasp_pose)

        scene.robot.open_gripper()

        scene.robot.lin(action[0])

        scene.robot.close_gripper()

        post_grasp_pose = self.post_grasp_offset * action[0]
        scene.robot.lin(post_grasp_pose)


class Place:
    def __init__(self,
                 pre_place_offset: Affine = Affine(translation=[0, 0, 0.075]),
                 post_place_offset: Optional[Affine] = None):
        self.pre_place_offset = pre_place_offset
        if post_place_offset is None:
            self.post_place_offset = self.pre_place_offset
        else:
            self.post_place_offset = post_place_offset

    def execute(self, action: Action, scene: Scene):
        pre_place_pose = self.pre_place_offset * action[0]
        scene.robot.ptp(pre_place_pose)

        scene.robot.lin(action[0])

        scene.robot.open_gripper()

        post_grasp_pose = self.post_place_offset * action[0]
        scene.robot.lin(post_grasp_pose)


class PickAndPlace:
    def __init__(self, pick: Pick = Pick(), place: Place = Place()):
        self.pick = pick
        self.place = place

    def execute(self, action: Action, scene: Scene):
        self.pick.execute(Action([action[0]]), scene)
        self.place.execute(Action([action[1]]), scene)


def register() -> None:
    factory.register_primitive('pick-primitive', Pick)
    factory.register_primitive('place-primitive', Place)
    factory.register_primitive('pick-and-place-primitive', PickAndPlace)
