# Age Detection Flask App - SSR-NET

A Flask web application for age detection using the SSR-NET (Stage-wise Regression Network) model. This application allows users to upload images and get age predictions for detected faces.

## Features

- **Modern Web Interface**: Beautiful, responsive UI with gradient design
- **Real-time Age Detection**: Uses SSR-NET model for accurate age estimation
- **Multiple Face Support**: Can detect and predict age for multiple faces in a single image
- **Face Detection**: Uses MTCNN for robust face detection
- **Image Processing**: Supports various image formats (JPG, PNG, GIF, BMP)
- **Result Visualization**: Shows both original and processed images with age annotations

## Project Structure

```
Age Detection SSR-NET/
├── app.py                 # Main Flask application
├── index.html            # Web interface (will be moved to templates/)
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── demo/                # Demo scripts and model files
│   ├── SSRNET_model.py  # SSR-NET model architecture
│   ├── TY_demo_image.py # Demo script for image processing
│   └── ...
├── pre-trained/         # Pre-trained model weights
│   ├── imdb/
│   ├── wiki/
│   ├── morph2/
│   └── ...
├── templates/           # HTML templates
│   └── index.html      # Main web interface
├── uploads/            # Uploaded images (created automatically)
└── results/            # Processed result images (created automatically)
```

## Setup Instructions

### 1. Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### 2. Installation

1. **Clone or copy the project files** to your desired location:
   ```bash
   # Copy the demo and pre-trained folders to your project directory
   # Make sure you have the following structure:
   # - demo/ (contains SSRNET_model.py and other demo files)
   # - pre-trained/ (contains model weights)
   # - app.py
   # - templates/index.html
   # - requirements.txt
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify the model files**:
   - Ensure `demo/SSRNET_model.py` exists
   - Ensure at least one pre-trained model exists in `pre-trained/` folders
   - The app will automatically search for `ssrnet_3_3_3_64_1.0_1.0.h5` files

### 3. Running the Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Access the web interface**:
   - Open your web browser
   - Navigate to `http://localhost:5000`
   - The application will load the model on startup

## Usage

1. **Upload an Image**:
   - Click the "Choose Image" button
   - Select an image file (JPG, PNG, GIF, BMP)
   - Maximum file size: 16MB

2. **View Results**:
   - The app will automatically process the image
   - Face detection will be performed using MTCNN
   - Age prediction will be done using SSR-NET
   - Results will show:
     - Original image
     - Processed image with age annotations
     - Age predictions for each detected face
     - Confidence scores

3. **Multiple Faces**:
   - The app can handle multiple faces in a single image
   - Each face will be detected and processed separately
   - Results will be displayed for each detected face

## Model Information

- **SSR-NET**: Stage-wise Regression Network for age estimation
- **MTCNN**: Multi-task Cascaded Convolutional Networks for face detection
- **Input Size**: 64x64 pixels
- **Supported Age Range**: 0-100 years
- **Model Architecture**: 3-stage regression network

## Technical Details

### Flask Routes

- `GET /`: Main page with upload interface
- `POST /upload`: Handle image upload and processing
- `GET /results/<filename>`: Serve processed result images
- `GET /uploads/<filename>`: Serve uploaded images

### File Processing

1. **Upload**: Images are saved to `uploads/` directory
2. **Processing**: Face detection → Age prediction → Result generation
3. **Output**: Processed images saved to `results/` directory
4. **Display**: Results shown in web interface with base64 encoding

### Error Handling

- Invalid file types
- No face detection
- Model loading failures
- File size limits
- Processing errors

## Troubleshooting

### Common Issues

1. **Model not loading**:
   - Check if pre-trained weights exist in `pre-trained/` folders
   - Verify the model file names match the expected pattern

2. **Face detection fails**:
   - Ensure the image contains clear, visible faces
   - Try different images with better lighting

3. **Import errors**:
   - Install all dependencies: `pip install -r requirements.txt`
   - Check Python version compatibility

4. **Memory issues**:
   - Reduce image size before uploading
   - Close other applications to free memory

### Performance Tips

- Use images with clear, well-lit faces
- Avoid very large images (resize if needed)
- The model loads once on startup for faster subsequent predictions

## Dependencies

- **Flask**: Web framework
- **OpenCV**: Image processing
- **TensorFlow/Keras**: Deep learning framework
- **MTCNN**: Face detection
- **NumPy**: Numerical computing
- **Pillow**: Image handling

## License

This project uses the SSR-NET model for age estimation. Please refer to the original SSR-NET repository for model licensing information.

## Contributing

Feel free to submit issues and enhancement requests! 