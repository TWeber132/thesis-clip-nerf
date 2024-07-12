import json
from manipulation_tasks.task import Task, TaskFactory
from manipulation_tasks.primitive import Primitive
from manipulation_tasks.scene import SimulatedScene
from manipulation_tasks.object import SceneObject
from manipulation_tasks.oracle import Oracle
from manipulation_tasks.sensor import Sensor
from typing import Any, Callable, Dict
from manipulation_tasks.transform import Affine

task_create_functions: Dict[str, Callable[..., Task]] = {}
primitive_create_functions: Dict[str, Callable[..., Primitive]] = {}
object_create_functions: Dict[str, Callable[..., SceneObject]] = {}
simulated_scene_create_functions: Dict[str, Callable[..., SimulatedScene]] = {}
oracle_create_functions: Dict[str, Callable[..., Oracle]] = {}
sensor_create_functions: Dict[str, Callable[..., Sensor]] = {}

task_factories: Dict[str, Callable[..., TaskFactory]] = {}
available_object_paths: Dict[str, str] = {}


def register_task(task_type: str, creator_fn: Callable[..., Task]) -> None:
    """Register a new task type."""
    task_create_functions[task_type] = creator_fn


def unregister_task(task_type: str) -> None:
    """Unregister a task type."""
    task_create_functions.pop(task_type, None)


def create_task(arguments: Dict[str, Any]) -> Task:
    """Create a task of a specific type"""
    args_copy = arguments.copy()
    task_type = args_copy.pop("task_type")
    try:
        creator_func = task_create_functions[task_type]
    except KeyError:
        raise ValueError(f"unknown task type {task_type!r}") from None
    return creator_func(**args_copy)


def register_oracle(oracle_type: str, creator_fn: Callable[..., Oracle]) -> None:
    """Register a new task type."""
    oracle_create_functions[oracle_type] = creator_fn


def unregister_oracle(oracle_type: str) -> None:
    """Unregister a task type."""
    oracle_create_functions.pop(oracle_type, None)


def create_oracle(arguments: Dict[str, Any]) -> Oracle:
    """Create a task of a specific type"""
    args_copy = arguments.copy()
    oracle_type = args_copy.pop("oracle_type")
    try:
        creator_func = oracle_create_functions[oracle_type]
    except KeyError:
        raise ValueError(f"unknown oracle type {oracle_type!r}") from None
    return creator_func(**args_copy)


def register_task_factory(task_factory_type: str, creator_fn: Callable[..., TaskFactory]) -> None:
    """Register a new task factory type."""
    task_factories[task_factory_type] = creator_fn


def unregister_task_factory(task_factory_type: str) -> None:
    """Unregister a task factory type."""
    task_factories.pop(task_factory_type, None)


def create_task_factory(arguments: Dict[str, Any]) -> TaskFactory:
    """Create a task factory of a specific type"""
    args_copy = arguments.copy()
    task_factory_type = args_copy.pop("task_factory_type")
    try:
        creator_func = task_factories[task_factory_type]
    except KeyError:
        raise ValueError(f"unknown task factory type {task_factory_type!r}") from None
    return creator_func(**args_copy)


def register_primitive(primitive_type: str, creator_fn: Callable[..., Primitive]) -> None:
    """Register a new primitive type."""
    primitive_create_functions[primitive_type] = creator_fn


def unregister_primitive(primitive_type: str) -> None:
    """Unregister a primitive type."""
    primitive_create_functions.pop(primitive_type, None)


def create_primitive(arguments: Dict[str, Any]) -> Primitive:
    """Create a primitive of a specific type"""
    args_copy = arguments.copy()
    primitive_type = args_copy.pop("primitive_type")
    try:
        creator_func = primitive_create_functions[primitive_type]
    except KeyError:
        raise ValueError(f"unknown primitive type {primitive_type!r}") from None
    return creator_func(**args_copy)


def register_object(object_type: str, creator_fn: Callable[..., SceneObject]) -> None:
    """Register a new object type."""
    object_create_functions[object_type] = creator_fn


def unregister_object(object_type: str) -> None:
    """Unregister an object type."""
    object_create_functions.pop(object_type, None)


# TODO naming ... o_type is not good; object_type is not good --> its an argument for SceneObject dataclass
def create_object(o_type, arguments: Dict[str, Any]) -> SceneObject:
    """Create an object of a specific type"""
    args_copy = arguments.copy()
    try:
        creator_func = object_create_functions[o_type]
    except KeyError:
        raise ValueError(f"unknown object type {o_type!r}") from None
    return creator_func(**args_copy)


def register_available_object(object_type: str, resources_path: str):
    available_object_paths[object_type] = resources_path


def unregister_available_object(object_type: str) -> None:
    """Unregister a primitive type."""
    available_object_paths.pop(object_type, None)


def create_manipulation_object(object_type, manipulation_type):
    urdf = f'{available_object_paths[object_type]}/object.urdf'
    kwargs = create_object_args_dict(manipulation_type, object_type, urdf)
    return create_object(manipulation_type, kwargs)


def create_target_object(object_type, target_object_type, target_type):
    if target_object_type is not None:
        urdf = f'{available_object_paths[object_type]}/{target_object_type}.urdf'
    else:
        urdf = None
    kwargs = create_object_args_dict(target_type, object_type, urdf)
    return create_object(target_type, kwargs)


def create_object_args_dict(manipulation_type, object_type, urdf):
    config_file = f'{available_object_paths[object_type]}/{manipulation_type}_config.json'
    with open(config_file) as json_file:
        additional_args = json.load(json_file)
    kwargs = {
        'urdf_path': urdf,
        'object_id': -1,
    }
    offset = Affine(**additional_args['offset'])
    additional_args['offset'] = offset
    kwargs.update(additional_args)
    return kwargs


def register_simulated_scene(simulated_scene_type: str, creator_fn: Callable[..., SimulatedScene]) -> None:
    """Register a new simulated scene type."""
    simulated_scene_create_functions[simulated_scene_type] = creator_fn


def unregister_simulated_scene(simulated_scene_type: str) -> None:
    """Unregister a simulated scene type."""
    simulated_scene_create_functions.pop(simulated_scene_type, None)


def create_simulated_scene(arguments: Dict[str, Any]) -> SimulatedScene:
    """Create a simulated scene of a specific type"""
    args_copy = arguments.copy()
    simulated_scene_type = args_copy.pop("scene_type")
    try:
        creator_func = simulated_scene_create_functions[simulated_scene_type]
    except KeyError:
        raise ValueError(f"unknown simulated scene type {simulated_scene_type!r}") from None
    return creator_func(**args_copy)

def register_sensor(sensor_type: str, creator_fn: Callable[..., Sensor]) -> None:
    """Register a new simulated scene type."""
    sensor_create_functions[sensor_type] = creator_fn

def unregister_sensor(sensor_type: str) -> None:
    """Unregister a simulated scene type."""
    sensor_create_functions.pop(sensor_type, None)

def create_sensor(arguments: Dict[str, Any]) -> Sensor:
    """Create a simulated scene of a specific type"""
    args_copy = arguments.copy()
    sensor_type = args_copy.pop("sensor_type")
    try:
        creator_func = sensor_create_functions[sensor_type]
    except KeyError:
        raise ValueError(f"unknown sensor type {sensor_type!r}") from None
    return creator_func(**args_copy)
