import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session

app = Flask(__name__, template_folder='templates')
app.secret_key = "supersecretkey"  # Required for session storage

# Create necessary folders
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Get absolute path to the current directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Define paths using absolute paths
MODEL_CONFIG = os.path.join(BASE_DIR, "project", "yolov4.cfg")
MODEL_WEIGHTS = os.path.join(BASE_DIR, "project", "yolov4.weights")
CLASS_NAMES = os.path.join(BASE_DIR, "project", "coco.names")

# Check if model files exist
for path in [MODEL_CONFIG, MODEL_WEIGHTS, CLASS_NAMES]:
    if not os.path.exists(path):
        print(f"Warning: File not found - {path}")

# Load YOLO model
yolo_net = cv2.dnn.readNetFromDarknet(MODEL_CONFIG, MODEL_WEIGHTS)
yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load class labels
with open(CLASS_NAMES, "r") as f:
    classes = f.read().strip().split("\n")

def detect_animals(image_path, output_path):
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    layer_names = yolo_net.getUnconnectedOutLayersNames()
    outputs = yolo_net.forward(layer_names)
    
    boxes, confidences, class_ids = [], [], []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, w, h) = box.astype("int")
                x, y = int(centerX - w / 2), int(centerY - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indices) > 0 and indices is not None:
        indices = indices.flatten()
    else:
        indices = []

    herd_counts = {}

    for i in indices:
        animal_type = classes[class_ids[i]]
        herd_counts[animal_type] = herd_counts.get(animal_type, 0) + 1

        # Draw detection boxes & labels
        x, y, w, h = boxes[i]
        label = f"{animal_type} ({confidences[i]:.2f})"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Define text display
    herd_texts = []
    single_texts = []

    for animal, count in herd_counts.items():
        if count >= 3:
            herd_texts.append(f"Herd of {animal}s")
        else:
            single_texts.append(animal)

    herd_message = ", ".join(herd_texts) + " detected." if herd_texts else ""
    single_message = "Multiple animals detected (" + ", ".join(single_texts) + ")." if single_texts else ""

    cv2.imwrite(output_path, frame)
    
    return output_path, herd_message or single_message

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            result_path = os.path.join(app.config['RESULT_FOLDER'], file.filename)
            result_path, herd_text = detect_animals(file_path, result_path)
            session['herd_text'] = herd_text  # Store in session
            return redirect(url_for('display_result', filename=file.filename))
    return render_template('upload.html')

@app.route('/results/<filename>')
def display_result(filename):
    herd_text = session.pop('herd_text', '')  # Retrieve from session
    return render_template('result.html', filename=filename, herd_text=herd_text)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results_img/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
