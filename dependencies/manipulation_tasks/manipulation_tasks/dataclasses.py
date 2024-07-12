from dataclasses import dataclass
from manipulation_tasks.transform import Affine
from typing import List


@dataclass
class Objective:
    """
    The objective of transporting a manipulation object to one of its valid target poses.

    Class variables:
    :var bool completed: indicates whether the objective was completed
    :var int object_id: unique_id of manipulation object
    :var List[int] target_ids: list of unique_ids of possible target objects
    """
    completed: bool = False
    object_unique_id: int = -1
    target_unique_ids: List[int] = None


@dataclass
class Action:
    """
    Base class for objects that can be placed in a scene.

    Class variables:
    :var List[Affine] poses: type of object, as registered in the plugin e.g. 'scene_object', 'manipulation_object'
    """
    poses: List[Affine]
    type: str = None

    def __getitem__(self, i):
        return self.poses[i]

    def __len__(self):
        return len(self.poses)

    def __iter__(self):
        return iter(self.poses)
