from fastapi import FastAPI, Request, HTTPException
from starlette.responses import JSONResponse

from routes.grasp_routes import router as grasp_router

app = FastAPI()
app.include_router(grasp_router)


@app.exception_handler(Exception)
async def validation_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500, content={"message": "Unexpected error.", "detail": str(exc)}
    )


@app.exception_handler(HTTPException)
async def validation_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500, content={"message": "Unexpected error.", "detail": str(exc)}
    )


@app.get("/health")
def health_check():
    return {"status": "ok"}
