import uuid
import msgpack
from fastapi import APIRouter, Request, BackgroundTasks
from starlette.responses import Response

from process_manager import start_processing_if_not_started, _process_queue, process
from shared_resources import set_result, get_result

router = APIRouter()


# @router.get("/available_optimizers/")
# async def available_optimizers():
#     optimizers = []

@router.post("/optimize_poses")
async def optimize_poses_grasp(request: Request, background_tasks: BackgroundTasks):
    byte_data = await request.body()
    task_id = str(uuid.uuid4())
    task_type = "optimize_poses"
    set_result(task_id, {"status": "pending"})
    _process_queue.put((task_type, byte_data, task_id))
    start_processing_if_not_started(background_tasks, process)

    packed_response = msgpack.packb({"task_id": task_id})
    return Response(content=packed_response, media_type="application/octet-stream")


@router.post("/generate_trajectories")
async def optimize_poses_grasp(request: Request, background_tasks: BackgroundTasks):
    byte_data = await request.body()
    task_id = str(uuid.uuid4())
    task_type = "generate_trajectories"
    set_result(task_id, {"status": "pending"})
    _process_queue.put((task_type, byte_data, task_id))
    start_processing_if_not_started(background_tasks, process)

    packed_response = msgpack.packb({"task_id": task_id})
    return Response(content=packed_response, media_type="application/octet-stream")


@router.get("/result/{task_id}")
async def get_optimization_result(task_id):
    result_data = get_result(task_id)
    if result_data is None:
        result_data = {"status": "not found"}
    packed_response = msgpack.packb(result_data)
    return Response(content=packed_response, media_type="application/octet-stream")
