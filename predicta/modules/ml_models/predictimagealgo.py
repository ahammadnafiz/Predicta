import streamlit as st
from .image_models import ImageModel
from ...utils.code_display import ShowCode
from ...core.logging_config import get_logger

class PredictImageAlgo:
    def __init__(self, data_path=None):
        self.data_path = data_path
    
    def algo(self):
        st.markdown("<h1 style='text-align: center; font-size: 30px;'>Image Processing Models</h1>", unsafe_allow_html=True)
        st.markdown("---")

        self._handle_single_image()

    def _handle_single_image(self):
        img_model = ImageModel(self.data_path)
            
        preprocessing_options = {
            'resize': st.sidebar.checkbox("Resize Image", value=True),
            'grayscale': st.sidebar.checkbox("Convert to Grayscale"),
            'normalize': st.sidebar.checkbox("Normalize", value=True)
        }

        algorithm_option = st.sidebar.selectbox(
            "Select Algorithm",
            ["Image Classification", "Object Detection", "Image Segmentation"]
        )

        if algorithm_option == "Image Classification":
            model_name = st.sidebar.selectbox(
                "Select Model",
                ["resnet50", "efficientnet", "vgg16"]
            )
            img_model.image_classification(
                model_name=model_name,
                preprocessing_options=preprocessing_options
            )
        elif algorithm_option == "Object Detection":
            model_name = st.sidebar.selectbox(
                "Select Model",
                ["yolov5s", "yolov5m", "yolov5l"]
            )
            confidence = st.sidebar.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5
            )
            img_model.object_detection(
                model_name=model_name,
                confidence_threshold=confidence
            )
        elif algorithm_option == "Image Segmentation":
            model_name = st.sidebar.selectbox(
                "Select Model",
                ["deeplabv3_resnet50", "fcn_resnet50"]
            )
            img_model.image_segmentation(model_name=model_name)