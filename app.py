import streamlit as st
import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image, ImageDraw
import numpy as np
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# --- Model Loading (Cached for performance) ---

@st.cache_resource
def load_segmentation_model():
    """Loads the pre-trained Mask R-CNN model for instance segmentation."""
    model = maskrcnn_resnet50_fpn(weights='DEFAULT')
    model.eval()
    return model

@st.cache_resource
def load_captioning_model():
    """Loads the pre-trained ViT-GPT2 model and tokenizer for image captioning."""
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    return model, feature_extractor, tokenizer

# --- Core Functions ---

def perform_segmentation(image, model, confidence_threshold=0.5):
    """
    Performs instance segmentation on an image and draws masks.
    
    Args:
        image (PIL.Image): The input image.
        model: The pre-trained Mask R-CNN model.
        confidence_threshold (float): The minimum score for a detected object to be included.
        
    Returns:
        PIL.Image: The image with segmentation masks drawn on it.
    """
    # Define the COCO class names
    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    # Convert image to tensor
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(image)

    # Get predictions
    with torch.no_grad():
        prediction = model([img_tensor])

    # Draw masks on the image
    img_with_masks = image.copy()
    draw = ImageDraw.Draw(img_with_masks, "RGBA")
    
    scores = prediction[0]['scores']
    masks = prediction[0]['masks']
    labels_idx = prediction[0]['labels']

    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            mask = masks[i, 0].mul(255).byte().cpu().numpy()
            label = COCO_INSTANCE_CATEGORY_NAMES[labels_idx[i]]
            
            # Create a color for the mask
            color = np.random.randint(0, 255, size=3)
            mask_img = Image.fromarray(mask, mode='L')
            
            # Draw the mask
            draw.bitmap((0, 0), mask_img, fill=tuple(color) + (128,)) # Semi-transparent fill
            
            # Find a position for the label
            y, x = np.where(mask > 128)
            if len(y) > 0 and len(x) > 0:
                text_pos = (np.min(x), np.min(y) - 10)
                draw.text(text_pos, label, fill="white")

    return img_with_masks


def generate_caption(image, model, feature_extractor, tokenizer):
    """
    Generates a caption for an image.
    
    Args:
        image (PIL.Image): The input image.
        model: The captioning model.
        feature_extractor: The ViT feature extractor.
        tokenizer: The GPT-2 tokenizer.
        
    Returns:
        str: The generated caption.
    """
    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
    
    # Generate caption IDs
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    
    # Decode to text
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return caption


# --- Streamlit App UI ---

st.set_page_config(layout="wide", page_title="Integrated Image Understanding")

st.title("Project: Image Segmentation & Captioning 📸📝")
st.write("This app combines two deep learning tasks: **Instance Segmentation** to identify objects and **Image Captioning** to describe the scene. Upload an image to see it in action!")

# Load models
with st.spinner("Loading AI models... This might take a moment."):
    segmentation_model = load_segmentation_model()
    captioning_model, feature_extractor, tokenizer = load_captioning_model()

st.success("Models loaded successfully! Ready to process images.")
st.markdown("---")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file).convert("RGB")
    
    st.subheader("Original Image")
    st.image(image, caption="Your uploaded image.", use_column_width=True)
    
    st.markdown("---")
    
    # Process the image upon button click
    if st.button("Analyze Image", use_container_width=True):
        
        # --- Column layout for results ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🖼️ Generated Caption")
            with st.spinner("Generating description..."):
                caption = generate_caption(image, captioning_model, feature_extractor, tokenizer)
                st.write(f"**{caption.capitalize()}**")

        with col2:
            st.subheader("🎨 Segmented Image")
            with st.spinner("Identifying objects..."):
                segmented_image = perform_segmentation(image, segmentation_model)
                st.image(segmented_image, caption="Objects identified with segmentation masks.", use_column_width=True)