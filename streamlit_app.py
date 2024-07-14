import streamlit as st
import requests
from PIL import Image
from io import BytesIO

st.title("Depth Estimation App")
st.write("Upload an image to get its depth estimation.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # 将图像发送到FastAPI进行深度估计
    try:
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://127.0.0.1:8000/predict-depth/", files=files)

        if response.status_code == 200:
            output_image = Image.open(BytesIO(response.content))
            st.image(output_image, caption='Depth Estimated Image.', use_column_width=True)
        else:
            st.write(f"Error in processing the image: {response.text}")
    except Exception as e:
        st.write(f"Exception occurred: {e}")

# 使用下面的命令运行该Streamlit应用
# streamlit run your_streamlit_file_name.py
