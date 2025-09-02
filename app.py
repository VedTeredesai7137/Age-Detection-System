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

# Import SSR-Net model
from demo.SSRNET_model import SSR_net, SSR_net_general

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'

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
    
    if model is None:
        print("[INFO] Loading SSR-Net age model...")
        
        # Build model architecture
        model = SSR_net(64, [3, 3, 3], 1.0, 1.0)()
        
        # Find and load weights
        weights_path = find_weights_file()
        if weights_path:
            try:
                model.load_weights(weights_path)
                print(f"[INFO] Age model loaded successfully from {weights_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load age weights: {e}")
                return False
        else:
            print("[ERROR] No age weights file found")
            return False
    
    if model_gender is None:
        print("[INFO] Loading SSR-Net gender model...")
        
        # Build gender model architecture
        model_gender = SSR_net_general(64, [3, 3, 3], 1.0, 1.0)()
        
        # Find and load gender weights
        gender_weights_path = find_gender_weights_file()
        if gender_weights_path:
            try:
                model_gender.load_weights(gender_weights_path)
                print(f"[INFO] Gender model loaded successfully from {gender_weights_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load gender weights: {e}")
                return False
        else:
            print("[ERROR] No gender weights file found")
            return False
    
    if detector is None:
        print("[INFO] Loading MTCNN face detector...")
        detector = MTCNN()
    
    if face_cascade is None:
        print("[INFO] Loading LBP face cascade...")
        face_cascade = cv2.CascadeClassifier('demo/lbpcascade_frontalface_improved.xml')
    
    return True

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
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None, "Failed to load image"
        
        # Detect faces
        faces = detector.detect_faces(img)
        if len(faces) == 0:
            return None, "No face detected in the image"
        
        results = []
        
        for i, face in enumerate(faces):
            x, y, w, h = face["box"]
            x, y = max(0, x), max(0, y)
            
            # Extract face region
            face_img = img[y:y+h, x:x+w]
            
            # Preprocess face
            face_resized = cv2.resize(face_img, (64, 64))
            face_normalized = face_resized.astype("float32") / 255.0
            face_input = np.expand_dims(face_normalized, axis=0)
            
            # Predict age
            pred = model.predict(face_input, verbose=0)
            age = pred[0][0]
            
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
        
        # Save result image
        result_filename = f"result_{os.path.basename(image_path)}"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        cv2.imwrite(result_path, img)
        
        return results, result_filename
        
    except Exception as e:
        return None, str(e)

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
    
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Process frame for age and gender detection
        processed_frame = process_camera_frame(frame)
        
        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            continue
        
        # Yield frame as bytes
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Load model if not loaded
        if not load_age_model():
            return jsonify({'error': 'Failed to load model'}), 500
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Predict age
        results, result_filename = predict_age(filepath)
        
        if results is None:
            return jsonify({'error': result_filename}), 400
        
        # Convert result image to base64 for display
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        with open(result_path, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'results': results,
            'result_image': img_data,
            'result_filename': result_filename
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/camera')
def camera_page():
    return render_template('camera.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_camera_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera():
    """Start camera stream"""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return jsonify({'status': 'success', 'message': 'Camera started'})

@app.route('/stop_camera')
def stop_camera():
    """Stop camera stream"""
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return jsonify({'status': 'success', 'message': 'Camera stopped'})

if __name__ == '__main__':
    # Load model on startup
    if load_age_model():
        print("[INFO] Flask app starting...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("[ERROR] Failed to load model. Exiting...")
