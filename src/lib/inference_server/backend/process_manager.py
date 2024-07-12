import queue
import threading

from routes.grasp_process import process_optimize_poses, process_generate_trajectories

_processing_lock = threading.Lock()
_is_processing = False
_process_queue = queue.Queue()


def process():
    while not _process_queue.empty():
        task_type, byte_data, task_id = _process_queue.get()
        if task_type in ["optimize_poses"]:
            process_optimize_poses(byte_data, task_id)
        elif task_type in ["generate_trajectories"]:
            process_generate_trajectories(byte_data, task_id)
        else:
            raise ValueError(f"Unknown task type {task_type}")
    stop_processing()


def start_processing_if_not_started(background_tasks, task_function):
    global _is_processing
    with _processing_lock:
        if not _is_processing:
            _is_processing = True
            background_tasks.add_task(task_function)


def stop_processing():
    global _is_processing
    with _processing_lock:
        _is_processing = False
