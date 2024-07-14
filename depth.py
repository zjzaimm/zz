from PIL import Image
import numpy as np
import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation
from PIL import Image
import numpy as np


async def depth_estimation(image_path):
    low_cpu_mem_usage = True
    image_processor = DPTImageProcessor.from_pretrained("D:\\LLM_models\\dpt_hybrid_midas")
    model = DPTForDepthEstimation.from_pretrained("D:\\LLM_models\\dpt_hybrid_midas",
                                                  low_cpu_mem_usage=low_cpu_mem_usage)

    # 加载图像
    image = Image.open(image_path)

    # 准备图像以输入模型
    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # 插值到原始大小
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False
    )

    # 可视化预测结果
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)

    # 保存生成的深度图像到本地
    a = "output_depth_image.jpg"
    depth.save(a)
    return a


