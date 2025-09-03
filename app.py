import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from werkzeug.utils import secure_filename
from mtcnn import MTCNN
from keras.models import load_model
import base64
from PIL import Image
import io
import timeit
import logging
import traceback
import sys

# Force CPU-only mode for TensorFlow (required for Render deployment)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import TensorFlow and configure for CPU-only
try:
    import tensorflow as tf
    # Configure TensorFlow to use CPU only
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    logger.info(f"TensorFlow configured for CPU-only mode (version: {tf.__version__})")
except Exception as e:
    logger.warning(f"Could not configure TensorFlow: {e}")

# Import SSR-Net model
try:
    from demo.SSRNET_model import SSR_net, SSR_net_general
    logger.info("Successfully imported SSR-NET models")
except Exception as e:
    logger.error(f"Failed to import SSR-NET models: {e}")
    sys.exit(1)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for development

# Add error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error. Please try again.'}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Global variables for model
model = None
model_gender = None
detector = None
face_cascade = None
camera = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_age_model():
    """Load the SSR-Net age estimation model"""
    global model, model_gender, detector, face_cascade
    
    try:
        if model is None:
            logger.info("Loading SSR-Net age model...")
            
            try:
                # Build model architecture
                logger.info("Building age model architecture...")
                model = SSR_net(64, [3, 3, 3], 1.0, 1.0)()
                logger.info("Age model architecture built successfully")
            except Exception as e:
                logger.error(f"Failed to create age model: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                return False
            
            # Find and load weights
            logger.info("Searching for age model weights...")
            weights_path = find_weights_file()
            if weights_path:
                try:
                    logger.info(f"Loading age weights from: {weights_path}")
                    model.load_weights(weights_path)
                    logger.info("Age model loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load age weights: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    return False
            else:
                logger.error("No age weights file found")
                return False
        
        if model_gender is None:
            logger.info("Loading SSR-Net gender model...")
            
            try:
                # Build gender model architecture
                logger.info("Building gender model architecture...")
                model_gender = SSR_net_general(64, [3, 3, 3], 1.0, 1.0)()
                logger.info("Gender model architecture built successfully")
            except Exception as e:
                logger.error(f"Failed to create gender model: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                return False
            
            # Find and load gender weights
            logger.info("Searching for gender model weights...")
            gender_weights_path = find_gender_weights_file()
            if gender_weights_path:
                try:
                    logger.info(f"Loading gender weights from: {gender_weights_path}")
                    model_gender.load_weights(gender_weights_path)
                    logger.info("Gender model loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load gender weights: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    return False
            else:
                logger.error("No gender weights file found")
                return False
        
        if detector is None:
            logger.info("Loading MTCNN face detector...")
            try:
                detector = MTCNN()
                logger.info("MTCNN detector loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load MTCNN detector: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                return False
        
        if face_cascade is None:
            logger.info("Loading LBP face cascade...")
            try:
                cascade_path = 'demo/lbpcascade_frontalface_improved.xml'
                if not os.path.exists(cascade_path):
                    logger.error(f"LBP cascade file not found: {cascade_path}")
                    return False
                face_cascade = cv2.CascadeClassifier(cascade_path)
                if face_cascade.empty():
                    logger.error("Failed to load LBP cascade classifier")
                    return False
                logger.info("LBP face cascade loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load LBP face cascade: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                return False
        
        logger.info("All models loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Unexpected error in load_age_model: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def find_weights_file():
    """Find the appropriate weights file in pre-trained folders"""
    search_folders = [
        "pre-trained/imdb/ssrnet_3_3_3_64_1.0_1.0",
        "pre-trained/wiki/ssrnet_3_3_3_64_1.0_1.0",
        "pre-trained/morph2/ssrnet_3_3_3_64_1.0_1.0",
        "pre-trained/megaface_asian/ssrnet_3_3_3_64_1.0_1.0",
    ]
    
    for folder in search_folders:
        if os.path.isdir(folder):
            for fname in os.listdir(folder):
                if (fname.lower().endswith((".h5", ".hdf5")) and 
                    "ssrnet_3_3_3_64_1.0_1.0" in fname and 
                    not fname.startswith("history_")):
                    return os.path.join(folder, fname)
    
    return None

def find_gender_weights_file():
    """Find the appropriate gender weights file in pre-trained folders"""
    search_folders = [
        "pre-trained/wiki_gender_models/ssrnet_3_3_3_64_1.0_1.0",
        "pre-trained/imdb_gender_models/ssrnet_3_3_3_64_1.0_1.0",
        "pre-trained/morph_gender_models/ssrnet_3_3_3_64_1.0_1.0",
    ]
    
    for folder in search_folders:
        if os.path.isdir(folder):
            for fname in os.listdir(folder):
                if (fname.lower().endswith((".h5", ".hdf5")) and 
                    "ssrnet_3_3_3_64_1.0_1.0" in fname and 
                    not fname.startswith("history_")):
                    return os.path.join(folder, fname)
    
    return None

def predict_age(image_path):
    """Predict age from an image file"""
    try:
        logger.info(f"Starting age prediction for image: {image_path}")
        
        # Check if image file exists
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None, "Image file not found"
        
        # Load image
        logger.info("Loading image...")
        img = cv2.imread(image_path)
        if img is None:
            logger.error("Failed to load image with OpenCV")
            return None, "Failed to load image"
        
        logger.info(f"Image loaded successfully. Shape: {img.shape}")
        
        # Check if models are loaded
        if model is None or detector is None:
            logger.error("Models not loaded")
            return None, "Models not loaded"
        
        # Detect faces
        logger.info("Detecting faces...")
        try:
            faces = detector.detect_faces(img)
            logger.info(f"Found {len(faces)} faces")
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None, f"Face detection failed: {str(e)}"
        
        if len(faces) == 0:
            logger.warning("No faces detected in the image")
            return None, "No face detected in the image"
        
        results = []
        
        for i, face in enumerate(faces):
            try:
                logger.info(f"Processing face {i+1}/{len(faces)}")
                
                x, y, w, h = face["box"]
                x, y = max(0, x), max(0, y)
                
                # Extract face region
                face_img = img[y:y+h, x:x+w]
                if face_img.size == 0:
                    logger.warning(f"Empty face region for face {i+1}")
                    continue
                
                # Preprocess face
                logger.info(f"Preprocessing face {i+1}...")
                face_resized = cv2.resize(face_img, (64, 64))
                face_normalized = face_resized.astype("float32") / 255.0
                face_input = np.expand_dims(face_normalized, axis=0)
                
                # Predict age
                logger.info(f"Predicting age for face {i+1}...")
                try:
                    pred = model.predict(face_input, verbose=0)
                    age = pred[0][0]
                    logger.info(f"Predicted age for face {i+1}: {int(age)}")
                except Exception as e:
                    logger.error(f"Age prediction failed for face {i+1}: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    continue
                
                # Draw rectangle and text on image
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, f"Age: {int(age)}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                results.append({
                    'face_id': i + 1,
                    'age': int(age),
                    'confidence': face['confidence'],
                    'bbox': [x, y, w, h]
                })
                
            except Exception as e:
                logger.error(f"Error processing face {i+1}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue
        
        if not results:
            logger.error("No faces were successfully processed")
            return None, "Failed to process any faces"
        
        # Save result image
        logger.info("Saving result image...")
        result_filename = f"result_{os.path.basename(image_path)}"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        
        try:
            cv2.imwrite(result_path, img)
            logger.info(f"Result image saved: {result_path}")
        except Exception as e:
            logger.error(f"Failed to save result image: {e}")
            return None, f"Failed to save result image: {str(e)}"
        
        logger.info(f"Age prediction completed successfully. Processed {len(results)} faces")
        return results, result_filename
        
    except Exception as e:
        logger.error(f"Unexpected error in predict_age: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None, f"Prediction failed: {str(e)}"

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
    """Draw label with background on image"""
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

def process_camera_frame(frame):
    """Process a single camera frame for age and gender detection"""
    global model, model_gender, face_cascade
    
    if model is None or model_gender is None or face_cascade is None:
        return frame
    
    img_h, img_w, _ = np.shape(frame)
    img_size = 64
    ad = 0.5
    
    # Detect faces using LBP cascade
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected = face_cascade.detectMultiScale(gray_img, 1.1)
    
    if len(detected) > 0:
        faces = np.empty((len(detected), img_size, img_size, 3))
        
        # Process each detected face
        for i, (x, y, w, h) in enumerate(detected):
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            # Expand face region
            xw1 = max(int(x1 - ad * w), 0)
            yw1 = max(int(y1 - ad * h), 0)
            xw2 = min(int(x2 + ad * w), img_w - 1)
            yw2 = min(int(y2 + ad * h), img_h - 1)
            
            # Extract and resize face
            face_img = frame[yw1:yw2 + 1, xw1:xw2 + 1, :]
            faces[i, :, :, :] = cv2.resize(face_img, (img_size, img_size))
            faces[i, :, :, :] = cv2.normalize(faces[i, :, :, :], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            
            # Draw face rectangles
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.rectangle(frame, (xw1, yw1), (xw2, yw2), (0, 0, 255), 2)
        
        # Predict ages and genders
        try:
            predicted_ages = model.predict(faces, verbose=0)
            predicted_genders = model_gender.predict(faces, verbose=0)
            
            # Draw results
            for i, (x, y, w, h) in enumerate(detected):
                x1, y1 = x, y
                
                # Determine gender
                gender_str = 'Male' if predicted_genders[i] > 0.5 else 'Female'
                
                # Create label
                label = f"{int(predicted_ages[i])}, {gender_str}"
                
                # Draw label
                draw_label(frame, (x1, y1), label)
                
        except Exception as e:
            print(f"Prediction error: {e}")
    
    return frame

def generate_camera_frames():
    """Generate camera frames for streaming"""
    global camera
    
    # For web deployment, camera won't be available
    # Return a placeholder frame with instructions
    logger.info("Camera streaming requested - returning placeholder frame")
    
    # Create a placeholder frame
    placeholder_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder_frame, 'Camera not available in web deployment', 
               (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(placeholder_frame, 'Please use image upload instead', 
               (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(placeholder_frame, 'or use WebRTC camera capture', 
               (120, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    while True:
        try:
            ret, buffer = cv2.imencode('.jpg', placeholder_frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1)  # Update every second
        except Exception as e:
            logger.error(f"Camera streaming error: {e}")
            break

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    try:
        return jsonify({
            'status': 'healthy',
            'message': 'Age Detection SSR-NET is running',
            'models_loaded': model is not None and model_gender is not None,
            'detector_loaded': detector is not None
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        logger.info("Received upload request")
        
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({'error': 'No selected file'}), 400
        
        logger.info(f"Processing file: {file.filename}")
        
        if file and allowed_file(file.filename):
            # Load model if not loaded
            logger.info("Checking if models are loaded...")
            if not load_age_model():
                logger.error("Failed to load models")
                return jsonify({'error': 'Failed to load model'}), 500
            
            # Save uploaded file
            logger.info("Saving uploaded file...")
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                logger.info(f"File saved to: {filepath}")
            except Exception as e:
                logger.error(f"Failed to save file: {e}")
                return jsonify({'error': f'Failed to save file: {str(e)}'}), 500
            
            # Predict age
            logger.info("Starting age prediction...")
            results, result_filename = predict_age(filepath)
            
            if results is None:
                logger.error(f"Age prediction failed: {result_filename}")
                return jsonify({'error': result_filename}), 400
            
            # Convert result image to base64 for display
            logger.info("Converting result image to base64...")
            try:
                result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
                with open(result_path, 'rb') as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                logger.info("Result image converted to base64 successfully")
            except Exception as e:
                logger.error(f"Failed to convert result image: {e}")
                return jsonify({'error': f'Failed to process result image: {str(e)}'}), 500
            
            logger.info("Upload and prediction completed successfully")
            return jsonify({
                'success': True,
                'results': results,
                'result_image': img_data,
                'result_filename': result_filename
            })
        
        logger.error(f"Invalid file type: {file.filename}")
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        logger.error(f"Unexpected error in upload_file: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/upload_base64', methods=['POST'])
def upload_base64():
    """Handle base64 image uploads from WebRTC camera"""
    try:
        logger.info("Received base64 image upload request")
        
        data = request.get_json()
        if not data or 'image' not in data:
            logger.error("No image data provided in base64 upload")
            return jsonify({'error': 'No image data provided'}), 400
        
        # Load model if not loaded
        logger.info("Checking if models are loaded for base64 upload...")
        if not load_age_model():
            logger.error("Failed to load models for base64 upload")
            return jsonify({'error': 'Failed to load model'}), 500
        
        # Decode base64 image
        logger.info("Decoding base64 image...")
        image_data = data['image']
        if image_data.startswith('data:image'):
            # Remove data URL prefix
            image_data = image_data.split(',')[1]
        
        # Decode and save image
        import uuid
        filename = f"webcam_{uuid.uuid4().hex[:8]}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            with open(filepath, 'wb') as f:
                f.write(base64.b64decode(image_data))
            logger.info(f"Base64 image saved to: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save base64 image: {e}")
            return jsonify({'error': f'Failed to save image: {str(e)}'}), 500
        
        # Predict age
        logger.info("Starting age prediction for base64 image...")
        results, result_filename = predict_age(filepath)
        
        if results is None:
            logger.error(f"Age prediction failed for base64 image: {result_filename}")
            return jsonify({'error': result_filename}), 400
        
        # Convert result image to base64 for display
        logger.info("Converting result image to base64...")
        try:
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            with open(result_path, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            logger.info("Base64 upload and prediction completed successfully")
        except Exception as e:
            logger.error(f"Failed to convert result image to base64: {e}")
            return jsonify({'error': f'Failed to process result image: {str(e)}'}), 500
        
        return jsonify({
            'success': True,
            'results': results,
            'result_image': img_data,
            'result_filename': result_filename
        })
        
    except Exception as e:
        logger.error(f"Unexpected error in upload_base64: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/camera')
def camera_page():
    return render_template('camera.html')

@app.route('/webrtc_camera')
def webrtc_camera_page():
    return render_template('webrtc_camera.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_camera_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera():
    """Start camera stream"""
    global camera
    
    try:
        logger.info("Camera start requested")
        # For web deployment, camera won't be available
        # Return success but with a note about limitations
        return jsonify({
            'status': 'success', 
            'message': 'Camera stream started (placeholder mode for web deployment)',
            'note': 'Live camera not available in web deployment. Use image upload or WebRTC capture.'
        })
    except Exception as e:
        logger.error(f"Camera start error: {e}")
        return jsonify({'status': 'error', 'message': f'Failed to start camera: {str(e)}'})

@app.route('/stop_camera')
def stop_camera():
    """Stop camera stream"""
    global camera
    try:
        logger.info("Camera stop requested")
        if camera is not None:
            camera.release()
            camera = None
        return jsonify({'status': 'success', 'message': 'Camera stopped'})
    except Exception as e:
        logger.error(f"Camera stop error: {e}")
        return jsonify({'status': 'error', 'message': f'Failed to stop camera: {str(e)}'})

if __name__ == '__main__':
    # Load models on startup
    logger.info("=" * 50)
    logger.info("Starting Age Detection SSR-NET Application")
    logger.info("=" * 50)
    
    logger.info("Loading models on startup...")
    if load_age_model():
        logger.info("Models loaded successfully!")
        logger.info("Starting Flask server...")
        
        # Use environment variable PORT for deployment (Render provides this)
        port = int(os.environ.get('PORT', 5000))
        logger.info(f"Server will start on port: {port}")
        logger.info("Application is ready to serve requests!")
        logger.info("=" * 50)
        
        app.run(debug=False, host='0.0.0.0', port=port)
    else:
        logger.error("Failed to load models. Exiting...")
        logger.error("=" * 50)
        sys.exit(1)


