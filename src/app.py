from io import BytesIO
from mimetypes import init
import config
import streamlit as st
import torch
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from  image_2_image import StableDiffusionImg2ImgPipeline, preprocess
from imagegen import ImageGen
import gc
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.title("Diffusers Application")
st.sidebar.markdown("This app shows different possibilities of using Huggingface Diffusers by using DiffusionPipeline")
what_application  = st.sidebar.selectbox("Select an application", ("Select an option", "prompt2img", "img2img", "sketch2img"))

@st.cache(show_spinner=True, allow_output_mutation=True)
def load_prompt_model(model_ckpt):
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    return StableDiffusionPipeline.from_pretrained(model_ckpt, revision="fp16", scheduler=scheduler, torch_dtype=torch.float16, use_auth_token=True)
prompt_diffusion_model = load_prompt_model(config.MODEL_CKPT)


@st.cache(show_spinner=True, allow_output_mutation=True)
def load_img2img_model(model_ckpt):
    return StableDiffusionImg2ImgPipeline.from_pretrained(model_ckpt, revision="fp16",  torch_dtype=torch.float16, use_auth_token=True
)
img_diffusion_model = load_prompt_model(config.MODEL_CKPT)

torch.cuda.empty_cache()    
gc.collect()


if what_application is not None:
    if what_application == "prompt2img":
        prompt = st.text_input("Enter the prompt")
        if st.button("Generate"):
            torch.cuda.empty_cache()    
            gc.collect()
            imagegen = ImageGen(prompt_diffusion_model)
            image = imagegen.prompt2image(prompt)
            st.image(image)

    elif what_application == "img2img":
        init_image = st.file_uploader("Upload the initial image")
        prompt = st.text_input("Enter the prompt")
        if st.button("Generate"):
            st.image(init_image)
            torch.cuda.empty_cache()    
            gc.collect()
            imagegen = ImageGen(img_diffusion_model)
            init_image = init_image.read()
            init_image = Image.open(BytesIO(init_image)).convert("RGB")
            init_image = init_image.resize((768, 512))
            init_image = preprocess(init_image)
            image = imagegen.img2image(init_image, prompt)
            st.image(image)
    
    elif what_application == "sketch2img":
        drawing_mode = st.sidebar.selectbox("Drawing tool:",
        ("freedraw", "line", "rect", "circle", "transform", "polygon", "point"),
                )
        stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
        if drawing_mode == 'point':
            point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
        stroke_color = st.sidebar.color_picker("Stroke color hex: ")
        bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
        bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
        realtime_update = st.sidebar.checkbox("Update in realtime", True)

        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=Image.open(bg_image) if bg_image else None,
            update_streamlit=realtime_update,
            height=150,
            drawing_mode=drawing_mode,
            point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
            display_toolbar=st.sidebar.checkbox("Display toolbar", True),
            key="full_app",
        )

        prompt = st.text_input("Enter the prompt")
        if st.button("Generate"):
            init_image = canvas_result.image_data
            st.image(init_image)
            torch.cuda.empty_cache()    
            gc.collect()
            imagegen = ImageGen(img_diffusion_model)
            # # init_image = init_image.read()
            # init_image = Image.open(BytesIO(init_image)).convert("RGB")
            # init_image = init_image.resize((768, 512))
            # init_image = preprocess(init_image)
            image = imagegen.img2image(init_image, prompt)
            st.image(image)
            