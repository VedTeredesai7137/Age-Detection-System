import os
import sys
import argparse

import cv2
import numpy as np
from mtcnn import MTCNN
from keras.models import load_model

# Import your SSR-Net builder class
from SSRNET_model import SSR_net  # Assuming this script is in the same folder as SSRNET_model.py

def parse_args():
    p = argparse.ArgumentParser(
        description="Run SSR‑Net age estimation on a single image"
    )
    p.add_argument("--image",    "-i", required=True,
                   help="Path to input image (jpg/png)")
    p.add_argument("--weights",  "-w", default=None,
                   help="Path to SSR‑Net weights (.h5 or .hdf5). "
                        "If omitted, script will look in pre-trained folders")
    p.add_argument("--size",     "-s", type=int, default=64,
                   help="Input size, default=64")
    p.add_argument("--stages",   "-t", nargs=3, type=int, default=[3,3,3],
                   help="Stage numbers (e.g. 3 3 3)")
    p.add_argument("--lambda_l", "-l", type=float, default=0.25,
                   help="lambda_local hyperparam")
    p.add_argument("--lambda_d", "-d", type=float, default=0.25,
                   help="lambda_d hyperparam")
    p.add_argument("--model_type", "-m", choices=["age", "gender"], default="age",
                   help="Model type: age or gender prediction")
    return p.parse_args()

def locate_weights(path_arg, model_type="age"):
    # If user specified exactly, use it
    if path_arg:
        if os.path.isfile(path_arg):
            return path_arg
        else:
            print(f"[ERROR] Specified weights not found: {path_arg}")
            sys.exit(1)
    
    # Otherwise search in appropriate folders based on model type
    if model_type == "age":
        search_folders = [
            os.path.join("..", "pre-trained", "imdb", "ssrnet_3_3_3_64_1.0_1.0"),
            os.path.join("..", "pre-trained", "wiki", "ssrnet_3_3_3_64_1.0_1.0"),
            os.path.join("..", "pre-trained", "morph2", "ssrnet_3_3_3_64_1.0_1.0"),
            os.path.join("..", "pre-trained", "megaface_asian", "ssrnet_3_3_3_64_1.0_1.0"),
        ]
    else:  # gender
        search_folders = [
            os.path.join("..", "pre-trained", "imdb_gender_models", "ssrnet_3_3_3_64_1.0_1.0"),
            os.path.join("..", "pre-trained", "wiki_gender_models", "ssrnet_3_3_3_64_1.0_1.0"),
            os.path.join("..", "pre-trained", "morph_gender_models", "ssrnet_3_3_3_64_1.0_1.0"),
        ]
    
    for folder in search_folders:
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            # Only load the main model weights, not history files
            if (fname.lower().endswith((".h5", ".hdf5")) and 
                "ssrnet_3_3_3_64_1.0_1.0" in fname and 
                not fname.startswith("history_")):
                candidate = os.path.join(folder, fname)
                print(f"[INFO] Using weights: {candidate}")
                return candidate
    print(f"[ERROR] No SSR-Net {model_type} weights file found in known pre-trained folders.")
    sys.exit(1)

def main():
    args = parse_args()

    # 1) locate weights
    weights_path = locate_weights(args.weights, args.model_type)

    # 2) build model architecture
    print("[INFO] Building SSR‑Net model architecture...")
    model = SSR_net(args.size, args.stages, args.lambda_l, args.lambda_d)()

    # 3) load weights
    print(f"[INFO] Loading weights from {weights_path}...")
    try:
        model.load_weights(weights_path)
        print("[INFO] Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load weights: {e}")
        sys.exit(1)

    # 4) load and detect face
    if not os.path.isfile(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)

    img = cv2.imread(args.image)
    if img is None:
        print(f"[ERROR] Failed to load image: {args.image}")
        sys.exit(1)
    
    detector = MTCNN()
    faces = detector.detect_faces(img)
    if len(faces) == 0:
        print("[ERROR] No face detected in the image.")
        sys.exit(1)

    print(f"[INFO] Found {len(faces)} face(s) in the image.")

    # take the first detected face
    x, y, w, h = faces[0]["box"]
    x, y = max(0, x), max(0, y)
    face = img[y:y+h, x:x+w]

    # 5) preprocess face
    face = cv2.resize(face, (args.size, args.size))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)

    # 6) predict
    print(f"[INFO] Predicting {args.model_type}...")
    try:
        pred = model.predict(face, verbose=0)
        if args.model_type == "age":
            age = pred[0][0]
            print(f"[RESULT] Predicted age: {age:.2f} years")
            label = f"Age: {int(age)}"
        else:  # gender
            gender = pred[0][0]
            gender_label = "Male" if gender > 0.5 else "Female"
            confidence = abs(gender - 0.5) * 2  # Convert to 0-1 confidence
            print(f"[RESULT] Predicted gender: {gender_label} (confidence: {confidence:.2f})")
            label = f"Gender: {gender_label}"
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        sys.exit(1)

    # 7) visualize
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.putText(img, label, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    
    # Save the result image
    output_path = f"result_{args.model_type}_{os.path.basename(args.image)}"
    cv2.imwrite(output_path, img)
    print(f"[INFO] Result saved as: {output_path}")
    
    # Display the image
    cv2.imshow(f"SSR‑Net {args.model_type.title()} Estimation", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
