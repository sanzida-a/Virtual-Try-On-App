# app.py
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io

from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from diffusers import StableDiffusionInpaintPipeline


@st.cache_resource
def load_segformer():
    processor = SegformerImageProcessor.from_pretrained(
        "mattmdjaga/segformer_b2_clothes",
        use_fast=True  # Use fast processor to remove warning
    )
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
    return processor, model

@st.cache_resource
def load_inpaint_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",  # More reliable on Windows
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        force_download=True  # Forces a fresh download to avoid missing files
    )
    pipe = pipe.to(device)
    return pipe


LABELS = {
    0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses",
    4: "Upper-clothes", 5: "Skirt", 6: "Pants", 7: "Dress",
    8: "Belt", 9: "Left-shoe", 10: "Right-shoe", 11: "Face",
    12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm",
    16: "Bag", 17: "Scarf"
}

def segment_clothes(image: Image.Image, region: str):
    processor, model = load_segformer()
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()

    
    if region == "Upper":
        target_ids = [4, 7]  
    else:
        target_ids = [5, 6, 12, 13] 

    mask = np.isin(pred_seg, target_ids).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask).convert("L")
    return mask_img



st.set_page_config(page_title="Virtual Try-On", layout="centered")
st.title("👕 Virtual Try-On Demo")
st.caption("Upload a photo → pick region → describe clothing → see result.")

uploaded = st.file_uploader("Upload a person photo", type=["jpg", "jpeg", "png"])
region = st.radio("Choose region to replace", ["Upper", "Lower"])
prompt = st.text_input("Enter clothing description", "A Hawaiian shirt")

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Original", use_container_width=True)

    with st.spinner("Detecting clothing region..."):
        mask = segment_clothes(image, region)
    st.image(mask, caption="Mask (white = replace)", use_container_width=True)

    if st.button("Generate Try-On"):
        pipe = load_inpaint_pipeline()
        device = pipe.device

        with st.spinner("Running inpainting..."):
            result = pipe(
                prompt=prompt,
                image=image,
                mask_image=mask,
                guidance_scale=7.5,
                num_inference_steps=30,
                generator=torch.Generator(device=device),
            ).images[0]

        st.image(result, caption="Result", use_container_width=True)

        # Download button
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        st.download_button(
            "Download Result",
            buf.getvalue(),
            file_name="tryon_result.png",
            mime="image/png"
        )

else:
    st.info("⬆️ Upload an image")
