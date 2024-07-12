from manipulation_tasks.dataclasses import Action
from manipulation_tasks.scene import Scene


class Primitive:
    """
    Protocol that describes the execution (motion primitive or primitives) of the Action.
    """
    def execute(self, action: Action, scene: Scene):
        """
        Executes the action in the scene (simulated or real).

        :param Action action: the action
        :param Scene scene: the scene
        """
        ...
