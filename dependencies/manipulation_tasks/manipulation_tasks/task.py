from typing import List, Dict, Any
from manipulation_tasks.object import SceneObject, ManipulationObject, TargetObject
from manipulation_tasks.dataclasses import Objective, Action
from manipulation_tasks.primitive import Primitive
from manipulation_tasks.scene import Scene, SimulatedScene


class Task:
    """
    Protocol that completely describes a manipulation task.

    Class variables:
    :var Primitive primitive: manipulation primitive
    :var List[Objective] objectives: all objectives (also completed)
    :var List[ManipulationObject] manipulation_objects: list of manipulation objects of the task
    :var List[TargetObject] target_objects: list of target objects of the task
    """
    primitive: Primitive
    objectives: List[Objective]
    manipulation_objects: List[ManipulationObject]
    target_objects: List[TargetObject]

    def get_info(self) -> Dict[str, Any]:
        """
        Returns the information required for exactly recreating the task as a dict. If used as keyword arguments, this
        dict should be sufficient to recreate the task with factory.create_task(<dict>).

        :return Dict[str, Any]:
        """
        ...

    def execute(self, action: Action, scene: Scene):
        """
        Executes the action in the scene (simulated or real). The objectives of task are not updated.

        :param Action action:
        :param Scene scene:
        """
        ...

    def get_object_with_unique_id(self, unique_id: int) -> SceneObject:
        """
        Returns the object with the given unique ID.

        :param int unique_id: unique ID of the object
        :return SceneObject: the object
        """
        ...

    def setup(self, scene: SimulatedScene):
        """
        Sets up the task in the simulated scene (e.g. adds objects to the scene).

        :param SimulatedScene scene: simulated scene
        """
        ...

    def clean(self, scene: SimulatedScene):
        """
        Removes task objects from the simulated scene.

        :param SimulatedScene scene: simulated scene
        """
        ...


class TaskFactory:
    """
    Protocol for creating Task instances.
    """
    def create_task(self) -> Task:
        """
        Creates Task instance.
        :return Task: Task instance
        """
        ...
