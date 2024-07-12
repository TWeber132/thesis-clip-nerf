from manipulation_tasks.task import Task
from manipulation_tasks.scene import Scene
from manipulation_tasks.dataclasses import Action
from manipulation_tasks.transform import Affine
from typing import List, Tuple


class Oracle:
    """
    Protocol for solving tasks and evaluating solutions considering ground truth.
    """

    def execute(self, action: Action, task: Task, scene: Scene = None):
        """
        Executes the action of the given task while also updating the poses of its manipulation and target objects and
        setting the corresponding objective to completed. If a scene is provided, the manipulation primitive is also
        executed.
        :param Action action: action, fitting the task's manipulation primitive
        :param Task task: task
        :param Scene scene: optional, real or simulated scene
        :return:
        """
        ...

    def solve(self, task: Task) -> Tuple[Action, bool]:
        """
        Returns an action that solves an unsolved objective of the given task and a bool indicating whether all
        objectives of the task have been solved.
        :param task: the task
        :return Tuple[Action, bool]: action, that solves an objective and a bool indicating whether the task was
                                     completely solved.
        """
        ...

    def compute_attention_errors(self, task: Task, attention_pose: Affine) -> List[Tuple[float, float]]:
        """
        Computes translational and rotational errors for the given attention pose in the context of the given task.
        :param Task task: task instance
        :param Affine attention_pose: attention pose
        :return List[Tuple[float, float]]: list of translational and rotational errors
        """
        ...

    def compute_transport_errors(self, task: Task, attention_pose: Affine, transport_pose: Affine) -> List[
        Tuple[float, float]]:
        """
        Computes translational and rotational errors for the given transport pose considering the given attention pose
        in the context of the given task.
        :param Task task: task instance
        :param Affine attention_pose: attention pose
        :param Affine transport_pose: transport pose
        :return List[Tuple[float, float]]: list of translational and rotational errors
        """
        ...

    def compute_simulated_error(self, task: Task, attention_pose: Affine, scene: Scene) -> List[Tuple[float, float]]:
        """

        """
        ...
