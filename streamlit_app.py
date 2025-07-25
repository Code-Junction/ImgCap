import streamlit as st
import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image, ImageDraw
import numpy as np
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import warnings
import os

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ.update({"TRANSFORMERS_VERBOSITY": "error", "TOKENIZERS_PARALLELISM": "false"})

# --- Model Loading ---
@st.cache_resource
def load_models():
    """Load both segmentation and captioning models"""
    # Segmentation model
    seg_model = maskrcnn_resnet50_fpn(weights='DEFAULT')
    seg_model.eval()
    
    # Captioning model
    cap_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    
    # Configure tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    cap_model.config.pad_token_id = tokenizer.eos_token_id
    cap_model.eval()
    
    return seg_model, cap_model, processor, tokenizer

# --- Core Functions ---

def perform_segmentation(image, model, confidence=0.5):
    """Perform segmentation and draw masks"""
    COCO_CLASSES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant','stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'earbuds', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'watch', 'dining table',
        'toilet', 'solar panel', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'spectacles', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(image)

    with torch.no_grad():
        prediction = model([img_tensor])

    img_with_masks = image.copy()
    scores, masks, labels = prediction[0]['scores'], prediction[0]['masks'], prediction[0]['labels']

    for i, score in enumerate(scores):
        if score > confidence:
            mask = masks[i, 0].mul(255).byte().cpu().numpy()
            label = COCO_CLASSES[labels[i]]
            color = np.random.randint(0, 255, size=3)
            
            mask_indices = mask > 128
            if np.any(mask_indices):
                img_array = np.array(img_with_masks)
                overlay = np.zeros_like(img_array)
                overlay[mask_indices] = color
                img_array[mask_indices] = (img_array[mask_indices] * 0.7 + overlay[mask_indices] * 0.3).astype(np.uint8)
                img_with_masks = Image.fromarray(img_array)
                
                y_coords, x_coords = np.where(mask_indices)
                if len(y_coords) > 0:
                    text_pos = (int(np.min(x_coords)), max(0, int(np.min(y_coords)) - 10))
                    draw = ImageDraw.Draw(img_with_masks)
                    draw.text(text_pos, label, fill="white", stroke_width=1, stroke_fill="black")

    return img_with_masks


def generate_caption(image, model, processor, tokenizer):
    """Generate caption for image"""
    try:
        pixel_values = processor(images=[image], return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values, max_length=16, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id
            )
        caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        return caption.lstrip("a ").lstrip("an ") or "Image analysis completed"
    except:
        return "Image contains various objects and scenes"


# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="Image AI Analysis")

st.title("🖼️ Image Segmentation & Captioning")
st.write("Upload an image to see AI-powered object detection and description!")

# Load models
with st.spinner("Loading AI models..."):
    segmentation_model, captioning_model, processor, tokenizer = load_models()
st.success("✅ Models loaded! Ready to analyze images.")

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Settings
    confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5)
    
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.subheader("Original Image")
    st.image(image, use_container_width=True)
    
    # Analyze button
    if st.button("🔍 Analyze Image", use_container_width=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Caption")
            with st.spinner("Generating description..."):
                caption = generate_caption(image, captioning_model, processor, tokenizer)
                st.write(f"**{caption.capitalize()}**")
        
        with col2:
            st.subheader("🎯 Object Detection")
            with st.spinner("Detecting objects..."):
                segmented = perform_segmentation(image, segmentation_model, confidence)
                st.image(segmented, caption="Detected objects", use_container_width=True)
