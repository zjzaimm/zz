from PIL import Image
import numpy as np
import torch

from transformers import DPTImageProcessor, DPTForDepthEstimation

img_path = "C:\\Users\\df004\\Desktop\\haitun.jpg"


def deep(img_path):
    # Load the pre-trained models
    image_processor = DPTImageProcessor.from_pretrained("D:\\LLM_models\\dpt_hybrid_midas")
    model = DPTForDepthEstimation.from_pretrained("D:\\LLM_models\\dpt_hybrid_midas", low_cpu_mem_usage=True)

    # Local image path
    local_image_path = img_path
    image = Image.open(local_image_path)

    # Prepare image for the model
    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

    # Interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # Visualize the prediction
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)
    depth.show()
    a = depth.save("dsdsfcdf")
    return a