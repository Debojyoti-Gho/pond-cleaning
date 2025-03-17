import streamlit as st
import torch
import requests
from PIL import Image
import numpy as np
import clip
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ------------------------
# Download SAM checkpoint if not exists
# ------------------------
def download_sam_checkpoint(model_dir="models", checkpoint_name="sam_vit_h_4b8939.pth"):
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, checkpoint_name)
    if not os.path.exists(checkpoint_path):
        st.info("Downloading SAM checkpoint...")
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        r = requests.get(url, stream=True)
        with open(checkpoint_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        st.success("SAM checkpoint downloaded.")
    return checkpoint_path

# ------------------------
# Load SAM and CLIP Models
# ------------------------
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = download_sam_checkpoint()
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device)
    mask_generator = SamAutomaticMaskGenerator(sam, points_per_batch=32)
    clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
    return mask_generator, clip_model, clip_preprocess, device

# ------------------------
# Generate Masks with SAM
# ------------------------
def generate_masks(image, mask_generator):
    image_np = np.array(image)
    masks = mask_generator.generate(image_np)
    return masks

# ------------------------
# Label each mask with CLIP
# ------------------------
def label_masks_with_clip(image, masks, clip_model, clip_preprocess, device):
    labels = []
    for mask in masks:
        x0, y0, x1, y1 = mask["bbox"]
        cropped = image.crop((x0, y0, x1, y1))
        if cropped.size[0] < 5 or cropped.size[1] < 5:
            labels.append("Too small")
            continue

        image_input = clip_preprocess(cropped).unsqueeze(0).to(device)

        # You can expand this list with more specific prompts
        possible_labels = [
            "a person", "a tree", "a car", "a dog", "a cat", "a building", "a chair",
            "a laptop", "a ball", "a road", "a sign", "a sky", "grass", "water", "nothing"
        ]
        text_inputs = torch.cat([clip.tokenize(label) for label in possible_labels]).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_inputs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarities = (image_features @ text_features.T).squeeze(0)
            best_label = possible_labels[similarities.argmax().item()]
            labels.append(best_label)

    return labels

# ------------------------
# Visualize masks and labels
# ------------------------
def visualize_masks(image, masks, labels):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    for mask, label in zip(masks, labels):
        x0, y0, w, h = mask["bbox"]
        rect = patches.Rectangle((x0, y0), w, h, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x0, y0 - 5, label, color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    ax.axis('off')
    st.pyplot(fig)

# ------------------------
# Streamlit UI
# ------------------------
def main():
    st.set_page_config(page_title="SAM + CLIP Visual Segmentation", layout="wide")
    st.title("🧠 Segment Anything + CLIP Labeling")
    st.caption("Upload an image, segment it with SAM, and label each segment using CLIP.")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Loading models..."):
            mask_generator, clip_model, clip_preprocess, device = load_models()

        with st.spinner("Generating masks and labels..."):
            masks = generate_masks(image, mask_generator)
            labels = label_masks_with_clip(image, masks, clip_model, clip_preprocess, device)

        st.subheader("🖼️ Segmented Image with Labels")
        visualize_masks(image, masks, labels)

if __name__ == "__main__":
    main()
