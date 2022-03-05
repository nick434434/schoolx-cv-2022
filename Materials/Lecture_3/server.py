import os
from json import dumps
from fastapi import FastAPI, Path, Query, Response
from config import CONFIG


# Launch server with command 'uvicorn server:app --reload
app = FastAPI()


@app.get("/camera/{camera_id}")
def camera(camera_id: int):
    file_path = os.path.join(CONFIG["camera_files_path"], str(camera_id))
    with open(file_path, "r") as f:
        base64_frame = f.readline()
    return Response(dumps({"frame": base64_frame}))
