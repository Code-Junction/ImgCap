# 🖼️ Image AI Analysis App

A powerful Streamlit application that combines **AI-powered image captioning** and **object detection** using state-of-the-art deep learning models.

## 🚀 Features

- **📝 Image Captioning**: Generates descriptive captions for uploaded images using Vision Transformer + GPT-2
- **🎯 Object Detection**: Detects and segments objects in images using Mask R-CNN
- **🎛️ Interactive Controls**: Adjustable confidence threshold for detection
- **🌐 Web Interface**: User-friendly Streamlit interface
- **⚡ Real-time Processing**: Fast inference with PyTorch models

## 🛠️ Technologies Used

- **Streamlit** - Web app framework
- **PyTorch** - Deep learning framework
- **Transformers** - Hugging Face transformers library
- **OpenCV/PIL** - Image processing
- **Mask R-CNN** - Object detection and segmentation
- **Vision Transformer** - Image captioning

## 📦 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/image-ai-analysis.git
cd image-ai-analysis
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run streamlit_app.py
```

## 🎮 Usage

1. **Upload an Image**: Click "Choose an image..." and select a JPG, JPEG, or PNG file
2. **Adjust Settings**: Use the sidebar slider to set detection confidence (0.1 - 1.0)
3. **Analyze**: Click "🔍 Analyze Image" to process the image
4. **View Results**: 
   - Left panel shows AI-generated caption
   - Right panel displays detected objects with colored masks and labels

## 📁 Project Structure

```
image-ai-analysis/
├── streamlit_app.py          # Main Streamlit application
├── analysis.py               # Stock analysis and forecasting code
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
└── .gitignore               # Git ignore file
```

## 🤖 Models Used

- **Object Detection**: Mask R-CNN with ResNet-50 backbone (pre-trained on COCO dataset)
- **Image Captioning**: Vision Encoder-Decoder (ViT + GPT-2) from Hugging Face

## 🎯 Supported Objects

The app can detect 80+ object classes including:
- People, vehicles, animals
- Furniture, electronics, kitchen items
- Food items, sports equipment
- And many more from the COCO dataset

## 🚀 Deployment

### Option 1: Streamlit Cloud
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with one click!

### Option 2: Local Development
```bash
streamlit run streamlit_app.py
```

## 📊 Additional Features

This repository also includes `analysis.py` with:
- **Stock Price Analysis** using Yahoo Finance
- **Time Series Forecasting** with ARIMA, SARIMA, and Prophet
- **Interactive Gradio Interface** for stock predictions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face for pre-trained models
- Facebook Research for Mask R-CNN
- Streamlit team for the amazing framework
- COCO dataset for object detection training data

## 📞 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)

---

⭐ **Star this repository if you found it helpful!**
