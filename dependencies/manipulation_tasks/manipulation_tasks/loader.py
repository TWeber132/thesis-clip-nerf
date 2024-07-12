import importlib
import os
from typing import List, Dict
import manipulation_tasks.factory as factory


class ModuleInterface:
    """Represents a plugin interface. A plugin has a single register function."""

    @staticmethod
    def register() -> None:
        """Register the necessary items in the game character factory."""


def import_module(name: str) -> ModuleInterface:
    """Imports a module given a name."""
    return importlib.import_module(name)  # type: ignore


def load_plugins(plugins: List[str]) -> None:
    """Loads the plugins defined in the plugins list."""
    for plugin_file in plugins:
        plugin = import_module(plugin_file)
        plugin.register()


def add_available_objects(objects: Dict[str, str], root: str = None) -> None:
    for key, value in objects.items():
        if root is not None:
            value = os.path.join(root, value)
        factory.register_available_object(key, value)
