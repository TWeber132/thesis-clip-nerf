# Manipulation Tasks

We think of a manipulation task as a collection of objectives. Each objective refers to a single manipulation object and
a
set of possible target poses. This framework is aimed to help solving vision based manipulation tasks. For this reason,
we use target objects instead of target poses, as the target pose of the objective has to be visually recognizable
somehow.

**Note:** For box packing, we use target objects without a physical body, with a custom task factory plugin, that does
not add the target objects into the simulated scene.

To deal with symmetries, and other possible problems, each target object offers a set of valid target
poses (relative to the pose of the target object).

**Note:** There is a 1-to-1 mapping between manipulation object types and target object types, meaning that
multiple instances of target objects of a given type can belong to the same manipulation object of the corresponding
type. Other target object types can however not belong to this same manipulation object. Without this constraint,
target objects would need to provide valid relative poses for multiple manipulation objets. This functionality might be
included in the future

This module contains the task abstraction implementation. It provides dataclasses and protocols to help the design of
simulated robotic manipulation experiments. You can expand the module by implementing your own logic for manipulation
primitives, tasks, oracles, objects in the scene or even simulation backends as plugins. We provide some ready-to-use
implemented plugins for some basic functionality. For more information, please refer to the
[Plugins section](#example-plugins). To get an intuition for plugin based architectures, watch this
[video](https://youtu.be/iCE1bDoit9Q).

## Task abstraction

In the following we describe the intended functionality of the implemented modules.

## Utilities

### `factory`

Is used for registering and unregistering plugins and object types (SceneObjects, ManipulationObjects and
TargetObjects). The factory is also used create instances of objects (in the programming sense) defined in plugins.
TODO link to plugin example

In the following examples, the factory will be used to create objects defined in plugins, that implement this 
framework's protocols.
### `loader`

Loads plugins and objects from given paths.

```pycon
from manipulation_tasks import loader, factory

# load plugin that registers the 'pick' manipulation object in the factory
loader.load_plugins(['manipulation_tasks.plugins.objects.base'])
# define object types and the path to their description files
objects = {'cube': '/some_path/cube'}
# make objects available to the factory
loader.add_available_objects(objects)

cube = factory.create_manipulation_object('cube', 'pick')
```

### `object`

Provides protocols for objects in a scene or a task.

A sample plugin implementation for pick-and-place
tasks (`PickObject` and `PoseTargetObject`) can be found in <samp>plugins/object/base.py</samp>. The reason for
reimplementing SceneObject is that the Protocol `manipulation_tasks.object.SceneObject` only describes how a SceneObject
should look, it can not be instantiated.

#### `SceneObject`

Contains necessary information to handle objects of the task or in a scene.

`SceneObject`s typically have a corresponding URDF file, that describes their visual and physical
properties and potentially a config file, that stores object properties and describes their manipulability.
A `SceneObject` has two IDs. The `object_id` refers to the ID assigned to the object when added
to a simulation. This is required to retrieve the pose of the object after physical manipulation, or to remove
the object from the simulation. Since there is a possibility, that an object has no physical body (e.g. a target
object), or that no simulation is used, it is possible that this ID remains unset. The other ID, the `uniqe_id` is
assigned while generating the `Task` in the `TaskFactory`. This ID needs to be unique in the context of the `Task` and
is used in `Objective`s to map `ManipulationObject`s to `TargetObject`s. SceneObjects that are not relevant for the task
do not require a `unique_id`. A `SceneObject` always has a pose, an offset of its placement pose to its origin (e.g. the
center of mass of a cube and the center of its bottom side), an encompassing radius (`min_dist`), which can be used to
ensure that object placements are non-overlapping, and can either be static or not static.

The implementation of `SceneObjects` should be done via plugins. Each implemented scene object is registered in the
`factory` using a string identifier. This identifier can be used to create objects via the `factory.create_object()`
method:

```python
from manipulation_tasks import loader, factory
from manipulation_tasks.transform import Affine

# load plugin that registers 'scene-object' in the factory
loader.load_plugins(['manipulation_tasks.plugins.objects.base'])
# define specifications
some_object_urdf = '/path/to/some_object.urdf'
object_args = {
    'urdf_path': some_object_urdf,
    'pose': Affine(translation=[0.5, 0.0, 0.2]),
}
# create object using the factory
some_object = factory.create_object('scene-object', object_args)
```
**Note:** the `object_id` and the `unique_id` are not set when an object is created. They have to be set explicitly. 
#### `ManipulationObject`

Enables the manipulation of a scene object. It provides valid relative poses for the gripper for a given manipulation
primitive. Additionally, it calculates errors for a given gripper pose to the set of valid gripper poses also
accounting for rotational symmetries.

Since there is a 1-to-1 mapping between `ManipulationObject`s and `TargetObject`s, we store the files of a
`ManipulationObject` and its corresponding `TargetObject` in the same <samp>/some_path/*object_type*/</samp>
directory where <samp>object_type</samp> is a description of the object (e.g. cube). The URDF of the
manipulation is <samp>/some_path/*object_type*/object.urdf</samp>.  
A `ManipulationObject`'s task relevant affordances and its other properties are stored in
<samp>/some_path/object_type/*manipulation_type*_config.json</samp>.

The following example, <samp>/some_path/cube/pick_config.json</samp>, shows the config file of a cube with 5cm long edges for the `pick` manipulation type. 
`min_dist`refers to the encompassing radius,
for non-overlapping placement; and `offset` is the Affine transformation of its origin to its placement pose.
`pick_config` describes two segments, along which the TCP of the gripper could be positioned for a successful grasp.

```json
{ 
  "min_dist": 0.08,
  "offset": {"translation": [0.0, 0.0, 0.025],
             "rotation": [0.0, 0.0, 0.0] },
  "pick_config": [
    { "type": "segment",
      "point_a": [-0.0125, 0, 0.0251],
      "point_b": [0.0125, 0, 0.0251] },
    { "type": "segment",
      "point_a": [0, -0.0125, 0.0251],
      "point_b": [0, 0.0125, 0.0251] } 
  ]
}
```

The implementation of the functionality to interpret and use these specifications should be done
via plugins. Each implemented `ManipulationObject` is registered in the
`factory` using the manipulation_type as a string identifier. This can be then used to create objects via the 
`factory.create_manipulation_object(object_type, manipulation_type)`
method:

```python
from manipulation_tasks import loader, factory

# load plugin that registers the 'pick' manipulation object in the factory
loader.load_plugins(['manipulation_tasks.plugins.objects.base'])
# define object types and the path to their description files
objects = {'cube': '/some_path/cube'}
# make objects available to the factory
loader.add_available_objects(objects)

cube = factory.create_manipulation_object('cube', 'pick')
```
**Note:** neither `object_id`, `unique_id` nor `pose` of the object are set when an object is created. 
They have to be set explicitly. 

**TODO** is manipulation type the same as manipulation primitive? In case of one primitive per task probably ... yes.

#### `TargetObject`

Defines the possible target poses for a manipulation object. It provides valid target poses for manipulation objects and
calculates errors for a given target pose to all the valid target poses.

A `TargetObject` always has a <samp>*target_type*_config.json</samp>, that contains
relevant object information and valid relative target poses for its corresponding 
`ManipulationObject`. It can also have a URDF <samp>*target_object_type*.urdf</samp>. 
Both are in the directory of the corresponding `ManipulationObject`s directory, e.g. for a 
*fixture* as a *pose-target* and the *cube*:

```
some_path
└───cube
│   │   fixture.urdf
│   │   object.urdf
│   │   pick_config.json
│   │   pose-target_config.json
```
Following this example, 
a pose-target target type requires the target objects to be placed precisely. 
With a fixture target object type, a cube can
be placed in four different ways correctly, due to rotational symmetries.

```json
{
  "min_dist": 0.1,
  "offset": {"translation": [0.0, 0.0, 0.0035],
             "rotation": [0.0, 0.0, 0.0] },
  "pose_target_config": [
      {"translation": [0.0, 0.0, 0.0215],
        "rotation": [0.0, 0.0, 0.0] },
      {"translation": [0.0, 0.0, 0.0215],
        "rotation": [0.0, 0.0, 1.5708] },
      {"translation": [0.0, 0.0, 0.0215],
        "rotation": [0.0, 0.0, 3.1416] },
      {"translation": [0.0, 0.0, 0.0215],
        "rotation": [0.0, 0.0, -1.5708] }
    ]
}
```
Just like `ManipulationObject`, the functionality of `TargetObject`s
should be implemented
via plugins. After loading a plugin, containing a `TargetObject` and thus, registering
it with target_object_type, the `factory` can 
create objects via the 
`factory.create_target_object(object_type, target_object_type, target_type)`
method:
```python
from manipulation_tasks import loader, factory

# load plugin that registers the 'pose-target' target object in the factory
loader.load_plugins(['manipulation_tasks.plugins.objects.base'])
# define object types and the path to their description files
objects = {'cube': '/some_path/cube'}
# make objects available to the factory
loader.add_available_objects(objects)


cube = factory.create_target_object('cube', 'fixture', 'pose-target')
```
**Note:** neither `object_id`, `unique_id` nor `pose` of the object are set when an object is created. 
They have to be set explicitly. 

**Note:** For tasks like stacking, manipulation objects can also be target objects. This still needs to be addressed.

### `dataclasses`

Simple objects, that are useful for defining manipulation tasks.

#### `Objective`

Maps an `object_id` to `target_id`s which correspond to a manipulation object and its possible target objects
`unique_id` respectively (See API docs object.SceneObject TODO: link to API ref). It also indicates whether the
objective was completed, meaning that the manipulation object arrived at a valid target pose.

#### `Action`

Contains a list of poses that are required to execute a manipulation primitive. For pick-and-place actions, it would be
a list containing two poses, the pick pose and the place pose. For now, it is just a wrapper for `list`s, but it 
will be easier
to implement more complex behaviour if this list of poses is encapsulated in a class, for example if some poses
require different manipulation primitives.

### `scene`

Provides protocols to communicate with a real or a simulated scene, including getting sensor data (camera images) and
controlling a robot.

Scenes are also defined as plugins. Take a look at the sample plugin implementation in pybullet for further information TODO: link to pybullet environment

The following minimal example shows how to load and create a scene, and how to interact with it.
The functionality of the available modules is briefly described afterwards.
```python
from manipulation_tasks import loader, factory
from manipulation_tasks.transform import Affine

# load plugin that registers the 'pose-target' target object in the factory
loader.load_plugins(['pybullet_environment.scene',
                     'manipulation_tasks.plugins.objects.base'])
# define object types and the path to their description files
objects = {'cube': '/some_path/cube'}
# make objects available to the factory
loader.add_available_objects(objects)
# configure scene
scene_config = {
    "type": "pybullet",
    "resources_root": "/opt/project/assets",
    "render": True
}
# initialize a simulated scene with a robot and cameras
scene = factory.create_simulated_scene(scene_config)

# create a manipulation object
cube = factory.create_manipulation_object('cube', 'pick')
# set the pose of the object
cube.pose = Affine(translation=[0.3, 0.3, 0.1])
# add the object to the scene. this should set the object's object_id
scene.add_object(cube)

# get images from the scene's cameras
observation = scene.get_observation()
# camera configurations for data processing
camera_configs = [c.get_config() for c in scene.cameras]

# visualize a coordinate frame in the scene
scene.spawn_coordinate_frame(Affine(translation=[0.3, 0.3, 0.1]))
# command the robot
target_pose = Affine(translation=[0.3, 0.3, 0.3])
scene.robot.ptp(target_pose)
target_pose = Affine(translation=[0.3, 0.3, 0.1])
scene.robot.lin(target_pose)
scene.robot.close_gripper()

# remove object(s) from the scene
scene.remove_objects([cube.object_id])
# clean up coordinate frames
scene.clean()

# stop the simulation 
scene.shutdown()
```

#### `Camera`

Contains information about the cameras used in the scene: camera pose, resolution, intrinsic parameters and depth range.
These are required for preprocessing i.e. generating pointcloud from depth images and calculating the orthographic
projection. This is the only object that is not a protocol in the `scene` module, to have a unified `get_config()`
method, which is needed for storing datasets. We might think of some better solution.

#### `Robot`

Enables commanding the robot of the scene. It provides interfaces for resetting the robot to its home position, for
point-to-point and linear movement and to open and close its gripper.

#### `Scene`

Holds references to the robot and the cameras and defines workspace bounds.
It additionally has some debugging functionality, like visualizing coordinate frames.

#### `SimulatedScene`

Some extended functionality for a `Scene`. Can add and remove `SceneObjects`.

### `primitive`

Provides the protocol to execute manipulation primitives.

A sample plugin implementation can be found in <samp>plugins/primitives/pick_and_place.py</samp> containing three
manipulation primitives.

The example shows the usage of primitives defined in the sample implementation.
```python
from manipulation_tasks import loader, factory
from manipulation_tasks.transform import Affine
from manipulation_tasks.dataclasses import Action

# load plugin that registers the 'pose-target' target object in the factory
loader.load_plugins(['pybullet_environment.scene',
                     'manipulation_tasks.plugins.objects.base', 
                     'manipulation_tasks.plugins.primitives.pick_and_place'])
# define object types and the path to their description files
objects = {'cube': '/some_path/cube'}
# make objects available to the factory
loader.add_available_objects(objects)
# configure scene
scene_config = {
    "type": "pybullet",
    "resources_root": "/opt/project/assets",
    "render": True
}
# initialize a simulated scene with a robot and cameras
scene = factory.create_simulated_scene(scene_config)

# create a manipulation object
cube = factory.create_manipulation_object('cube', 'pick')
# set the pose of the object
cube.pose = Affine(translation=[0.3, 0.3, 0.1])
# add the object to the scene. this should set the object's object_id
scene.add_object(cube)

# configure and initialize primitive
primitive_args = {'type': 'pick-and-place-primitive'}
primitive = factory.create_primitive(primitive_args)
# define action fitting to the primitive
action = Action([Affine(translation=[0.3, 0.3, 0.1]),
                 Affine(translation=[0.3, -0.3, 0.1])])
# execute action in a simulated or real scene
primitive.execute(action, scene)

# configure and initialize a different primitive
primitive_args = {'type': 'pick-primitive'}
primitive = factory.create_primitive(primitive_args)
# define action fitting to the primitive and execute
action = Action([Affine(translation=[0.3, -0.3, 0.1])])
primitive.execute(action, scene)

# stop the simulation 
scene.shutdown()
```

#### `Primitive`

Specifies the execution of an action.

### `task`

Contains protocols to define and create manipulation tasks.

A sample plugin implementation can be found in <samp>plugins/tasks/pick_and_place.py</samp> containing three
manipulation primitives.

In this example we show you how to load and configure a TaskFactory, how to create tasks and how to execute actions. 
```python
from manipulation_tasks import loader, factory
from manipulation_tasks.transform import Affine
from manipulation_tasks.dataclasses import Action

# load plugin that registers the 'pose-target' target object in the factory
loader.load_plugins(['pybullet_environment.scene',
                     'manipulation_tasks.plugins.objects.base', 
                     'manipulation_tasks.plugins.primitives.pick_and_place',
                     "manipulation_tasks.plugins.tasks.simple_task"])
# make objects available to the factory
objects = {'cube': '/some_path/cube'}
loader.add_available_objects(objects)
# configure and initialize scene
scene_config = {
    "type": "pybullet",
    "resources_root": "/opt/project/assets",
    "render": True
}
scene = factory.create_simulated_scene(scene_config)
# configure and initialize task factors
task_factory_config = {"type": "simple-task-factory",
                       "object_types": ["cube"],
                       "n_objects": 2,
                       "manipulation_type": "pick",
                       "primitive_type": "pick-and-place-primitive",
                       "target_object_type": "fixture",
                       "target_type": "pose-target",
                       "t_bounds": scene.t_bounds,
                       "r_bounds": scene.r_bounds}
task_factory = factory.create_task_factory(task_factory_config)
# create task
task = task_factory.create_task()
# setup, e.g. placing task objects into the scene
task.setup(scene)
# get task relevant information to store in a dataset
info = task.get_info()
# define some action that fits the primitive_type
action = Action([Affine(translation=[0.3, 0.3, 0.1]),
                 Affine(translation=[0.3, -0.3, 0.1])])
# execute action in a simulated or real scene
task.execute(action, scene)
# remove task objects from the scene
task.clean(scene)
# stop the simulation 
scene.shutdown()
```
#### `Task`

Holds all task-specific information, like motion primitives, objects and objectives. Furthermore, it enables
setting up the task and the execution of actions in a scene.

The `execute()` method basically wraps the `execute()` method of the task's primitive with the possibility to implement
some higher level logic. It should not update the task's state, e.g. object poses or objective completedness.

**Note:** The `get_info()` method is used for storing task instances in datasets. We suggest, that this method returns a
dict with the keyword arguments required to recreate this task, as the recreated task is required for testing.

#### `TaskFactory`

Is used to create task instances.

### `oracles`
Oracles have access to all information in a task, thus should be able to provide valid solutions.
An oracle can be used to solve tasks, execute actions and to compute errors of actions:
```python
from manipulation_tasks import loader, factory

# load required plugins
loader.load_plugins(['pybullet_environment.scene',
                     'manipulation_tasks.plugins.objects.base', 
                     'manipulation_tasks.plugins.primitives.pick_and_place',
                     "manipulation_tasks.plugins.tasks.simple_task",
                     "manipulation_tasks.plugins.oracles.insertion"])
# make objects available to the factory
objects = {'cube': '/some_path/cube'}
loader.add_available_objects(objects)
# configure and initialize scene
scene_config = {
    "type": "pybullet",
    "resources_root": "/opt/project/assets",
    "render": True
}
scene = factory.create_simulated_scene(scene_config)
# configure and initialize task factors
task_factory_config = {"type": "simple-task-factory",
                       "object_types": ["cube"],
                       "n_objects": 2,
                       "manipulation_type": "pick",
                       "primitive_type": "pick-and-place-primitive",
                       "target_object_type": "fixture",
                       "target_type": "pose-target",
                       "t_bounds": scene.t_bounds,
                       "r_bounds": scene.r_bounds}
task_factory = factory.create_task_factory(task_factory_config)
# configure and initialize oracle
oracle_config = {"type": "insertion-oracle",
                 "gripper_offset": {
                     "translation": [0.0, 0.0, 0.0],
                     "rotation": [3.14159265359, 0.0, 1.57079632679]}}
oracle = factory.create_oracle(oracle_config)
# create task
task = task_factory.create_task()
# solve task until all objectives are completed
solved = False
while not solved:
    # setup, e.g. placing task objects into the scene
    task.setup(scene)
    # let the oracle solve the task
    action, solved = oracle.solve(task)
    # let the oracle execute the task, thus also updating task states
    oracle.execute(action, task, scene) # or oracle.execute(action, task)
    # remove task objects from the scene
    task.clean(scene)
    # compute translational and rotational errors, should be 0 for the oracles solutions
    attention_errors = oracle.compute_attention_errors(task, action[0])
    transport_errors = oracle.compute_transport_errors(task, action[0], action[1])

# stop the simulation 
scene.shutdown()
```

#### `Oracle`

The `Oracle` is used to provide and to evaluate solutions for simulated tasks.

Additionally, an `execute()` method should also be implemented, that wraps the `execute()` method of the task. This
execute method should not be called before `solve()` and opposing to the `execute()` method of task, this should
update the task's state, based on the previously computed solution and object poses. If we provide a Scene instance 
for the execute method, the primitive will be executed in the simulation. 

# Example Plugins

## Object

## Scene

## Primitive

## Task

## Oracle