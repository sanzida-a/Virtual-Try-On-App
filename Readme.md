#  Virtual Try-On Web App

This project implements a **Virtual Try-On system** using Streamlit.  
It allows a user to upload their own image, select a clothing region (upper or lower),  
and apply a new clothing description via text prompt. The system automatically segments the clothing region and uses inpainting to replace it.

---

##  Features
- **Image Upload**: Upload your own image of a person.  
- **Region Selection**: Choose between *Upper* (shirt, jacket, dress) or *Lower* (pants, skirt).  
- **Prompt Input**: Describe the clothing item you want (e.g., `"A Hawaiian shirt with bright flowers"`).  
- **Automatic Segmentation**: SegFormer B2 detects the clothing area to be replaced.  
- **Inpainting**: Stable Diffusion Inpainting fills the region with the desired clothing item.  
- **Results**: Displays both the segmentation mask and the final try-on image.

---

##  Models Used

### 1. Stable Diffusion v2 Inpainting
- **Source**: [Stability AI / Stable Diffusion 2](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting)  
- **Checkpoint**: `512-inpainting-ema.ckpt`  
- **Paper**:  
  > Rombach, Robin, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer.  
  > *High-Resolution Image Synthesis with Latent Diffusion Models.*  
  > CVPR 2022. [PDF](https://arxiv.org/abs/2112.10752)  
- **License**: CreativeML Open RAIL++-M  

Used here to **generate and inpaint clothing** onto the masked region based on text prompts.

---

### 2. SegFormer B2 Clothes Segmentation
- **Model**: [`mattmdjaga/segformer_b2_clothes`](https://huggingface.co/mattmdjaga/segformer_b2_clothes)  
- **Dataset**: ATR Human Parsing dataset (`mattmdjaga/human_parsing_dataset`)  
- Provides **semantic segmentation** for: upper clothes, skirt, pants, dress, etc.  
- Used to generate a **binary mask** for the clothing region.  

---

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/yourname/virtual-tryon-app.git
cd virtual-tryon-app

# Create and activate environment (Python 3.11 recommended)
python -m venv venv
venv\Scripts\activate  # on Windows
# or
source venv/bin/activate  # on Linux/Mac

# Install dependencies
pip install -r requirements.txt

streamlit run app.py

```

## Project Structure:
app.py              # Main Streamlit app (single-file implementation)
requirements.txt    # Dependencies
README.md           # Project documentation


## Limitations

Clothing segmentation may not be perfect on complex backgrounds.

Generated clothing may not always match the exact body pose.

Faces and hands may occasionally get artifacts.

Only English prompts supported.