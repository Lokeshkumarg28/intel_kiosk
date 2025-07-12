from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Initialize face recognizer
# IMPORTANT: Ensure opencv-contrib-python is installed for cv2.face
try:
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except AttributeError:
    print("Error: cv2.face module not found. Please install opencv-contrib-python.")
    print("Run: pip uninstall opencv-python opencv-contrib-python && pip install opencv-contrib-python")
    exit() # Exit if the essential module is missing

# Database
# This will store face data in memory. In a real application,
# you would persist this data (e.g., to a database or files).
face_data = {}  # {patient_id: [face_images]}
label_map = {}  # {label: patient_id} - maps integer label to patient_id
reverse_label_map = {}  # {patient_id: label} - maps patient_id to integer label
current_label = 0 # Counter for assigning new labels

def train_model():
    """
    Trains the face recognizer with all currently registered faces.
    This function is called after each new face is added or on startup.
    """
    global face_recognizer, face_data
    if not face_data:
        # Cannot train if no face data exists
        print("No face data to train the model.")
        return

    labels = []
    images = []
    
    for patient_id, img_list in face_data.items():
        for img in img_list:
            # Ensure images are suitable for training (e.g., not empty)
            if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
                images.append(img)
                labels.append(reverse_label_map[patient_id])
            else:
                print(f"Skipping empty or invalid image for patient {patient_id}")
    
    if len(images) > 0 and len(labels) == len(images):
        try:
            face_recognizer.train(images, np.array(labels))
            print(f"Model trained with {len(images)} images.")
        except cv2.error as e:
            print(f"Error during model training: {e}")
            # This can happen if there's only one sample for a label or images are identical
            # For a production system, you'd handle this more robustly, e.g.,
            # requiring multiple samples per user.
    else:
        print("Not enough valid images or labels for training.")

@app.route('/register_face', methods=['POST'])
def register_face():
    """
    Endpoint to register a new face for a patient ID.
    Expects 'patient_id' in form data and 'photo' file.
    """
    global current_label, face_data, label_map, reverse_label_map
    
    patient_id = request.form.get('patient_id')
    file = request.files.get('photo')

    if not patient_id:
        return jsonify({'success': False, 'msg': 'Missing patient_id'}), 400
    if not file:
        return jsonify({'success': False, 'msg': 'Missing photo file'}), 400

    # Read and convert image from file bytes
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'success': False, 'msg': 'Invalid image file'}), 400

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)) # Increased minSize for better detection
    
    if len(faces) == 0:
        print(f"No face detected in registration photo for {patient_id}")
        return jsonify({'success': False, 'msg': 'No face detected'}), 400

    # Assume the largest face is the target face
    (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    face_img = gray[y:y+h, x:x+w]
    
    # Resize face image to a standard size for consistency (e.g., 100x100)
    face_img = cv2.resize(face_img, (100, 100), interpolation=cv2.INTER_AREA)

    # Add to in-memory database
    if patient_id not in face_data:
        face_data[patient_id] = []
        reverse_label_map[patient_id] = current_label
        label_map[current_label] = patient_id
        current_label += 1
        print(f"New patient {patient_id} assigned label {reverse_label_map[patient_id]}")
            
    face_data[patient_id].append(face_img)
    print(f"Added face sample for {patient_id}. Total samples: {len(face_data[patient_id])}")

    # Re-train the model immediately after adding a new face
    train_model()
    
    return jsonify({'success': True, 'msg': f'Face registered for {patient_id}'})

@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    """
    Endpoint to recognize a face from an uploaded photo.
    Expects 'photo' file.
    """
    file = request.files.get('photo')

    if not file:
        return jsonify({'success': False, 'msg': 'Missing photo file'}), 400

    # Read and convert image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'success': False, 'msg': 'Invalid image file'}), 400

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    
    if len(faces) == 0:
        print("No face detected in recognition photo.")
        return jsonify({'success': False, 'msg': 'No face detected'}), 400

    # Assuming one face, take the largest one
    (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    face_img = gray[y:y+h, x:x+w]
    
    # Resize face image to the same standard size as training images
    face_img = cv2.resize(face_img, (100, 100), interpolation=cv2.INTER_AREA)
    
    # Ensure the model has been trained before predicting
    if not face_data:
        print("No registered faces for recognition.")
        return jsonify({'success': False, 'msg': 'No registered faces'}), 404

    try:
        # Predict face
        label, confidence = face_recognizer.predict(face_img)
        
        # Confidence threshold (lower is better, adjust as needed)
        # A common threshold for LBPH is around 70-100. Lower means higher similarity.
        if confidence < 70 and label in label_map:
            patient_id = label_map[label]
            print(f"Face recognized: {patient_id} with confidence {confidence:.2f}")
            return jsonify({'success': True, 'patient_id': patient_id, 'confidence': confidence}), 200
        else:
            msg = f"No match found. Confidence: {confidence:.2f}, Label: {label}, Label Map: {label in label_map}"
            print(msg)
            return jsonify({'success': False, 'msg': 'No match found', 'confidence': confidence}), 404
    except cv2.error as e:
        print(f"Error during face prediction: {e}")
        # This can happen if the model is not trained or input image is invalid for prediction
        return jsonify({'success': False, 'msg': 'Prediction error (model might not be trained or input invalid)'}), 500

if __name__ == '__main__':
    # Initial training in case any existing data needs to be loaded (if persistence was implemented)
    # Since this is in-memory, face_data will be empty on fresh start.
    train_model() 
    app.run(host='0.0.0.0', port=5002, debug=True)
