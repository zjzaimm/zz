from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import os
import logging

from depth import depth_estimation



# 创建临时目录以存储上传的文件
if not os.path.exists("temp"):
    os.makedirs("temp")

logging.basicConfig(level=logging.INFO)


app = FastAPI()

@app.post("/predict-depth/")
async def predict_depth(file: UploadFile = File(...)):
    file_location = f"temp/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    # 调用深度估计函数
    output_image_path = await depth_estimation(file_location)
    return FileResponse(output_image_path, media_type="image/jpeg", filename="output_depth_image.jpg")

# 运行服务器
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
