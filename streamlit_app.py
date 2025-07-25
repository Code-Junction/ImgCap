import streamlit as st
import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image, ImageDraw
import numpy as np
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import warnings
import os
import time

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ.update({"TRANSFORMERS_VERBOSITY": "error", "TOKENIZERS_PARALLELISM": "false"})

# --- Model Loading ---
@st.cache_resource
def load_models():
    """Load both segmentation and captioning models with optimizations"""
    # Segmentation model with optimized settings
    seg_model = maskrcnn_resnet50_fpn(weights='DEFAULT')
    seg_model.eval()
    
    # Set to half precision for faster inference (if supported)
    if torch.cuda.is_available():
        seg_model = seg_model.half().cuda()
    
    # Captioning model with better quality settings
    cap_model = VisionEncoderDecoderModel.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    
    # Configure tokenizer for better generation
    tokenizer.pad_token = tokenizer.eos_token
    cap_model.config.pad_token_id = tokenizer.eos_token_id
    cap_model.config.decoder_start_token_id = tokenizer.bos_token_id
    
    # Enable better generation settings
    cap_model.config.max_length = 20
    cap_model.config.min_length = 5
    cap_model.config.do_sample = True
    cap_model.config.temperature = 0.7
    cap_model.eval()
    
    if torch.cuda.is_available():
        cap_model = cap_model.cuda()
    
    return seg_model, cap_model, processor, tokenizer

# --- Core Functions ---

def perform_segmentation(image, model, confidence=0.5):
    """Perform segmentation and draw masks with optimizations"""
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

    # Optimize image size for faster processing
    original_size = image.size
    max_size = 800  # Reduce from default for faster processing
    if max(original_size) > max_size:
        ratio = max_size / max(original_size)
        new_size = tuple(int(dim * ratio) for dim in original_size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(image)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda().half()

    with torch.no_grad():
        prediction = model([img_tensor])

    # Move results back to CPU for processing
    if torch.cuda.is_available():
        scores = prediction[0]['scores'].cpu()
        masks = prediction[0]['masks'].cpu()
        labels = prediction[0]['labels'].cpu()
    else:
        scores, masks, labels = prediction[0]['scores'], prediction[0]['masks'], prediction[0]['labels']

    img_with_masks = image.copy()
    
    # Optimize: Only process top N detections for speed
    top_n = 10
    top_indices = torch.argsort(scores, descending=True)[:top_n]
    
    for idx in top_indices:
        i = idx.item()
        if i >= len(scores) or scores[i] <= confidence:
            continue
            
        mask = masks[i, 0].mul(255).byte().numpy()
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
                draw.text(text_pos, label, fill="white", stroke_width=2, stroke_fill="black")

    # Resize back to original size if it was resized
    if max(original_size) > max_size:
        img_with_masks = img_with_masks.resize(original_size, Image.Resampling.LANCZOS)
    
    return img_with_masks


def generate_caption_with_quality(image, model, processor, tokenizer, quality="Balanced"):
    """Generate caption with adjustable quality settings"""
    try:
        # Smart image preprocessing based on quality setting
        original_size = image.size
        
        if quality == "Fast":
            # Fast mode - smaller image, fewer beams
            target_size = (224, 224)
            max_length = 15
            num_beams = 2
            do_sample = False
        elif quality == "High Quality":
            # High quality - larger image, more beams
            if max(original_size) > 512:
                ratio = 512 / max(original_size)
                target_size = tuple(int(dim * ratio) for dim in original_size)
            else:
                target_size = original_size
            max_length = 25
            num_beams = 5
            do_sample = True
        else:  # Balanced
            # Balanced mode - moderate settings
            if max(original_size) > 384:
                ratio = 384 / max(original_size)
                target_size = tuple(int(dim * ratio) for dim in original_size)
            else:
                target_size = original_size
            max_length = 20
            num_beams = 4
            do_sample = True
        
        # Resize image
        processed_image = image.resize(target_size, Image.Resampling.LANCZOS).convert('RGB')
        
        # Process with vision model
        pixel_values = processor(images=[processed_image], return_tensors="pt").pixel_values
        
        # Move to GPU if available
        if torch.cuda.is_available():
            pixel_values = pixel_values.cuda()
        
        # Generate caption with quality-specific settings
        generation_kwargs = {
            'max_length': max_length,
            'min_length': 5,
            'num_beams': num_beams,
            'do_sample': do_sample,
            'early_stopping': True,
            'pad_token_id': tokenizer.pad_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'no_repeat_ngram_size': 2
        }
        
        if do_sample:
            generation_kwargs.update({
                'temperature': 0.7,
                'top_k': 50,
                'top_p': 0.9
            })
        
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, **generation_kwargs)
        
        caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        
        # Clean up caption
        caption = caption.replace("a picture of", "").replace("an image of", "")
        caption = caption.replace("a photo of", "").replace("a close up of", "")
        caption = caption.strip()
        
        # Ensure caption starts properly
        if caption and not caption[0].isupper():
            caption = caption.capitalize()
        
        # Add fallback for very short captions
        if len(caption.split()) < 3:
            return "Image shows interesting visual content"
        
        return caption or "Image analysis completed"
        
    except Exception as e:
        print(f"Caption generation error: {e}")
        return "Unable to generate description for this image"


def generate_caption(image, model, processor, tokenizer):
    """Generate caption for image with optimized quality"""
    return generate_caption_with_quality(image, model, processor, tokenizer, "Balanced")


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
    st.sidebar.header("⚙️ Settings")
    confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5, 
                                   help="Higher values = more confident detections only")
    
    # Caption quality settings
    st.sidebar.subheader("📝 Caption Settings")
    caption_quality = st.sidebar.selectbox(
        "Caption Quality", 
        ["Balanced", "High Quality", "Fast"],
        index=0,
        help="Higher quality = better descriptions but slower processing"
    )
    
    # Image info
    image = Image.open(uploaded_file).convert("RGB")
    st.sidebar.header("📊 Image Info")
    st.sidebar.write(f"**Size:** {image.size[0]} x {image.size[1]} px")
    st.sidebar.write(f"**Format:** {uploaded_file.type}")
    st.sidebar.write(f"**File Size:** {len(uploaded_file.getvalue()) / 1024:.1f} KB")
    
    # Performance tips
    if max(image.size) > 1200:
        st.sidebar.warning("⚠️ Large image detected. Processing may take longer.")
    elif max(image.size) < 300:
        st.sidebar.info("ℹ️ Small image. Consider higher resolution for better detection.")
    
    # Load and display image
    st.subheader("Original Image")
    st.image(image, use_container_width=True)
    
    # Analyze button
    if st.button("🔍 Analyze Image", use_container_width=True):
        # Performance tracking
        start_time = time.time()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Caption")
            with st.spinner("Generating description..."):
                caption_start = time.time()
                caption = generate_caption_with_quality(image, captioning_model, processor, tokenizer, caption_quality)
                caption_time = time.time() - caption_start
                st.write(f"**{caption.capitalize()}**")
                st.caption(f"⏱️ Caption generated in {caption_time:.2f}s")
        
        with col2:
            st.subheader("🎯 Object Detection")
            with st.spinner("Detecting objects..."):
                detection_start = time.time()
                segmented = perform_segmentation(image, segmentation_model, confidence)
                detection_time = time.time() - detection_start
                st.image(segmented, caption="Detected objects", use_container_width=True)
                st.caption(f"⏱️ Detection completed in {detection_time:.2f}s")
        
        # Total processing time
        total_time = time.time() - start_time
        st.success(f"🚀 **Total processing time: {total_time:.2f} seconds**")
        
        # Performance tips
        with st.expander("💡 Performance Info"):
            st.write(f"""
            **Processing Statistics:**
            - Image Caption: {caption_time:.2f}s
            - Object Detection: {detection_time:.2f}s
            - Total Time: {total_time:.2f}s
            - GPU Acceleration: {'✅ Enabled' if torch.cuda.is_available() else '❌ CPU Only'}
            - Image Size: {image.size[0]} x {image.size[1]} pixels
            """)
            
            if torch.cuda.is_available():
                st.write("🔥 **GPU acceleration is active** for faster processing!")
            else:
                st.write("💻 Running on CPU. Consider GPU for 3-5x faster processing.")
