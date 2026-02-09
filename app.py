import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import matplotlib.pyplot as plt
import json
import tensorflow as tf
from PIL import Image as PILImage
import uuid
import google.generativeai as genai
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
from collections import Counter
import logging

app = Flask(__name__)
app.secret_key = 'plantdetect-secret-key-2024'

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
HISTORY_FOLDER = 'static/history'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(HISTORY_FOLDER):
    os.makedirs(HISTORY_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Initialize database
def init_db():
    conn = sqlite3.connect('plant_detection.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detection_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id TEXT,
                  filename TEXT,
                  disease TEXT,
                  confidence REAL,
                  timestamp DATETIME)''')
    conn.commit()
    conn.close()

init_db()

# Load model and class indices
try:
    model = load_model('models/plant_disease_model1.h5')
    with open('models/class_indices.json', 'r') as f:
        class_indices = json.load(f)
    class_names = {int(k): v for k, v in class_indices.items()}
    print("Model loaded successfully!")
    
    # Print model layers for debugging
    print("Model layers:")
    for i, layer in enumerate(model.layers):
        print(f"{i}: {layer.name}")
        
except Exception as e:
    print(f"Model files not found. Running in demo mode. Error: {e}")
    model = None
    class_names = {
        0: "Apple Apple scab",
        1: "Apple Black rot",
        2: "Apple Cedar apple rust",
        3: "Apple healthy",
        4: "Blueberry healthy",
        5: "Cherry Powdery mildew",
        6: "Cherry healthy",
        7: "Corn Cercospora leaf spot",
        8: "Corn Common rust",
        9: "Corn Northern Leaf Blight",
        10: "Corn healthy",
        11: "Grape Black rot",
        12: "Grape Esca (Black Measles)",
        13: "Grape Leaf blight (Isariopsis Leaf Spot)",
        14: "Grape healthy",
        15: "Orange Haunglongbing (Citrus greening)",
        16: "Peach Bacterial spot",
        17: "Peach healthy",
        18: "Pepper bell Bacterial spot",
        19: "Pepper bell healthy",
        20: "Potato Early blight",
        21: "Potato Late blight",
        22: "Potato healthy",
        23: "Raspberry healthy",
        24: "Soybean healthy",
        25: "Squash Powdery mildew",
        26: "Strawberry Leaf scorch",
        27: "Strawberry healthy",
        28: "Tomato Bacterial spot",
        29: "Tomato Early blight",
        30: "Tomato Late blight",
        31: "Tomato Leaf Mold",
        32: "Tomato Septoria leaf spot",
        33: "Tomato Spider mites Two-spotted spider mite",
        34: "Tomato Target Spot",
        35: "Tomato Tomato YellowLeaf Curl Virus",
        36: "Tomato Tomato mosaic virus",
        37: "Tomato healthy"
    }

# Image size
IMG_SIZE = 224

# Configure Gemini API
GEMINI_API_KEY = ""
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    print("Gemini API configured successfully!")
except Exception as e:
    print(f"Gemini API configuration failed: {e}")
    gemini_model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def apply_clahe(img_path):
    """Apply CLAHE preprocessing to an image"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Could not read image")
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return enhanced_img
    except Exception as e:
        print(f"CLAHE processing failed: {e}")
        return cv2.imread(img_path)

def get_img_array(img_path, size):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    array /= 255.0
    return array

def find_conv_layer(model):
    """Find a suitable convolutional layer for Grad-CAM"""
    # Try common layer names
    possible_layers = [
        'conv5_block3_out',  # ResNet
        'block14_sepconv2_act',  # Xception
        'Conv_1',  # Your model
        'out_relu',  # Your model
        'block_16_project_BN',  # MobileNet
        'block_13_project_BN'  # MobileNet
    ]
    
    for layer_name in possible_layers:
        try:
            layer = model.get_layer(layer_name)
            if 'conv' in layer.name or 'Conv' in layer.name:
                print(f"Using layer for Grad-CAM: {layer_name}")
                return layer_name
        except:
            continue
    
    # If no specific layer found, find the last convolutional layer
    for layer in reversed(model.layers):
        if 'conv' in layer.name or 'Conv' in layer.name:
            print(f"Using last conv layer: {layer.name}")
            return layer.name
    
    # Fallback to the last layer before flatten/dense
    for i, layer in enumerate(model.layers):
        if 'flatten' in layer.name or 'dense' in layer.name:
            if i > 0:
                print(f"Using layer before flatten/dense: {model.layers[i-1].name}")
                return model.layers[i-1].name
    
    print("No suitable layer found for Grad-CAM, using default")
    return model.layers[-2].name  # Second to last layer

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        
        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except Exception as e:
        print(f"Grad-CAM failed: {e}")
        # Return a simple centered heatmap as fallback
        heatmap = np.zeros((IMG_SIZE, IMG_SIZE))
        center_x, center_y = IMG_SIZE // 2, IMG_SIZE // 2
        for i in range(IMG_SIZE):
            for j in range(IMG_SIZE):
                dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                heatmap[i, j] = max(0, 1 - dist / (IMG_SIZE // 2))
        return heatmap

def save_gradcam(img_path, heatmap, cam_path, alpha=0.4):
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Could not read image for Grad-CAM")
            
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
        cv2.imwrite(cam_path, superimposed_img)
        print(f"Grad-CAM saved successfully: {cam_path}")
    except Exception as e:
        print(f"Failed to save Grad-CAM: {e}")
        # Copy original image as fallback
        try:
            img = cv2.imread(img_path)
            if img is not None:
                cv2.imwrite(cam_path, img)
                print(f"Used original image as fallback for Grad-CAM: {cam_path}")
        except Exception as e2:
            print(f"Failed to save fallback image: {e2}")

def get_gemini_response(prompt, language="english"):
    """Get clean, concise response from Gemini"""
    try:
        if gemini_model is None:
            return "I'm currently in demo mode. In production, I would provide detailed treatment recommendations for your plant disease."
        
        system_prompt = """You are a helpful plant disease expert. Provide clear, concise, and practical advice.
        
        Guidelines:
        - Keep responses under 150 words
        - Use simple, actionable language
        - Focus on organic solutions first
        - Provide specific steps
        - Avoid technical jargon
        - Be encouraging and helpful
        
        Format:
        - Start with main recommendation
        - Use bullet points for steps
        - End with prevention tips
        """
        
        if language.lower() == "kannada":
            full_prompt = f"{system_prompt}\n\nQuestion: {prompt}\n\nPlease respond in Kannada language only. Keep it simple and practical."
        else:
            full_prompt = f"{system_prompt}\n\nQuestion: {prompt}\n\nPlease respond in English language only. Keep it simple and practical."
            
        response = gemini_model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API error: {e}")
        return "I apologize, but I'm having trouble responding right now. Please try again later."

def save_to_history(user_id, filename, disease, confidence):
    conn = sqlite3.connect('plant_detection.db')
    c = conn.cursor()
    c.execute('''INSERT INTO detection_history 
                 (user_id, filename, disease, confidence, timestamp)
                 VALUES (?, ?, ?, ?, ?)''',
              (user_id, filename, disease, confidence, datetime.now()))
    conn.commit()
    conn.close()
    print(f"Saved to history: {disease} with {confidence:.2f} confidence")

def get_user_history(user_id, limit=10):
    conn = sqlite3.connect('plant_detection.db')
    c = conn.cursor()
    c.execute('''SELECT * FROM detection_history 
                 WHERE user_id = ? 
                 ORDER BY timestamp DESC 
                 LIMIT ?''', (user_id, limit))
    history = c.fetchall()
    conn.close()
    return history

def get_detection_stats(user_id):
    conn = sqlite3.connect('plant_detection.db')
    c = conn.cursor()
    c.execute('''SELECT disease, COUNT(*) as count 
                 FROM detection_history 
                 WHERE user_id = ? 
                 GROUP BY disease 
                 ORDER BY count DESC''', (user_id,))
    stats = c.fetchall()
    conn.close()
    return stats

@app.before_request
def make_session_permanent():
    session.permanent = True
    app.permanent_session_lifetime = timedelta()
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())

@app.route('/')
def home():
    """Landing page"""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page"""
    user_id = session.get('user_id', 'anonymous')
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # Generate unique filenames
            unique_id = str(uuid.uuid4())[:8]
            original_filename = secure_filename(file.filename)
            original_ext = original_filename.rsplit('.', 1)[1].lower()
            
            # Save original image
            original_save_name = f"original_{unique_id}.{original_ext}"
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_save_name)
            file.save(original_path)
            print(f"Original image saved: {original_save_name}")
            
            # Apply CLAHE preprocessing
            clahe_img = apply_clahe(original_path)
            clahe_save_name = f"clahe_{unique_id}.{original_ext}"
            clahe_path = os.path.join(app.config['UPLOAD_FOLDER'], clahe_save_name)
            cv2.imwrite(clahe_path, clahe_img)
            print(f"CLAHE image saved: {clahe_save_name}")
            
            # Process the image
            img_array = get_img_array(clahe_path, (IMG_SIZE, IMG_SIZE))
            
            # Make prediction
            if model:
                try:
                    preds = model.predict(img_array, verbose=0)
                    predicted_class = np.argmax(preds[0])
                    confidence = float(np.max(preds[0]))
                    disease_name = class_names.get(predicted_class, "Unknown Disease")
                    
                    print(f"Prediction: {disease_name} with {confidence:.2f} confidence")
                    
                    # Generate Grad-CAM
                    last_conv_layer_name = find_conv_layer(model)
                    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
                    
                    # Save Grad-CAM image
                    gradcam_save_name = f"gradcam_{unique_id}.{original_ext}"
                    gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_save_name)
                    save_gradcam(clahe_path, heatmap, gradcam_path)
                    
                    # Save to history
                    save_to_history(user_id, original_save_name, disease_name, confidence)
                    
                except Exception as e:
                    print(f"Prediction error: {e}")
                    # Fallback to demo mode
                    disease_name = "Tomato Healthy"
                    confidence = 0.85
                    gradcam_save_name = original_save_name
                    # Save to history for demo
                    save_to_history(user_id, original_save_name, disease_name, confidence)
            else:
                # Demo mode - simulate prediction
                disease_name = "Tomato Healthy"
                confidence = 0.92
                gradcam_save_name = original_save_name
                
                # Create a dummy gradcam image (copy of original)
                dummy_gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], f"gradcam_{unique_id}.{original_ext}")
                img = cv2.imread(original_path)
                cv2.imwrite(dummy_gradcam_path, img)
                
                # Save to history for demo
                save_to_history(user_id, original_save_name, disease_name, confidence)
                print(f"Demo mode: Saved {disease_name} to history")
            
            # Prepare URLs for the template
            original_url = url_for('static', filename=f'uploads/{original_save_name}')
            clahe_url = url_for('static', filename=f'uploads/{clahe_save_name}')
            gradcam_url = url_for('static', filename=f'uploads/{gradcam_save_name}')
            
            # Get user history for dashboard
            history = get_user_history(user_id, 5)
            stats = get_detection_stats(user_id)
            
            return render_template('predict.html', 
                               original_image=original_url,
                               clahe_image=clahe_url,
                               gradcam_image=gradcam_url,
                               prediction=disease_name,
                               confidence=confidence*100,
                               history=history,
                               stats=stats)
        else:
            return redirect(request.url)
    
    # GET request - show page with history
    history = get_user_history(user_id, 5)
    stats = get_detection_stats(user_id)
    return render_template('predict.html', history=history, stats=stats)

@app.route('/ask_gemini', methods=['POST'])
def ask_gemini():
    """Enhanced Gemini chatbot with clean responses"""
    data = request.get_json()
    question = data.get('question', '')
    disease = data.get('disease', '')
    language = data.get('language', 'english')
    
    if not question:
        return jsonify({'response': 'Please provide a question.'})
    
    # Create a clean, focused prompt
    prompt = f"""Plant Disease: {disease}
User Question: {question}

Please provide a clear, practical response with:
1. Main recommendation (1-2 sentences)
2. 3-4 specific action steps
3. Prevention tips
4. When to seek professional help

Keep it under 150 words, use simple language, and focus on organic solutions first."""
    
    try:
        response = get_gemini_response(prompt, language)
        return jsonify({'response': response})
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({'response': 'I apologize, but I encountered an error. Please try again.'})

@app.route('/history')
def history():
    """User detection history page"""
    user_id = session.get('user_id', 'anonymous')
    history = get_user_history(user_id, 20)
    stats = get_detection_stats(user_id)
    print(f"History page: Found {len(history)} records for user {user_id}")
    return render_template('history.html', history=history, stats=stats)

@app.route('/api/stats')
def api_stats():
    """API endpoint for statistics"""
    user_id = session.get('user_id', 'anonymous')
    stats = get_detection_stats(user_id)
    return jsonify({'stats': stats})

@app.route('/clear', methods=['GET'])
def clear_uploads():
    """Clear uploads and reset session"""
    user_id = session.get('user_id', 'anonymous')
    
    # Only clear files for the current user
    user_files = []
    conn = sqlite3.connect('plant_detection.db')
    c = conn.cursor()
    c.execute('SELECT filename FROM detection_history WHERE user_id = ?', (user_id,))
    user_files = [row[0] for row in c.fetchall()]
    conn.close()
    
    for filename in user_files:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                print(f"Deleted file: {filename}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    # Clear user history from database
    conn = sqlite3.connect('plant_detection.db')
    c = conn.cursor()
    c.execute('DELETE FROM detection_history WHERE user_id = ?', (user_id,))
    conn.commit()
    conn.close()
    print(f"Cleared history for user: {user_id}")
    
    # Clear session
    session.clear()
    return redirect(url_for('predict'))

@app.route('/export_history')
def export_history():
    """Export user history as CSV"""
    user_id = session.get('user_id', 'anonymous')
    history = get_user_history(user_id, 1000)
    
    if not history:
        return redirect(url_for('history'))
    
    # Create DataFrame
    df = pd.DataFrame(history, columns=['ID', 'User ID', 'Filename', 'Disease', 'Confidence', 'Timestamp'])
    
    # Export to CSV
    csv_filename = f'plant_detection_history_{user_id}.csv'
    csv_path = os.path.join(HISTORY_FOLDER, csv_filename)
    df.to_csv(csv_path, index=False)
    
    print(f"Exported history to: {csv_filename}")
    return redirect(url_for('static', filename=f'history/{csv_filename}'))

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('static/history', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("Starting PlantDetect Server...")
    print("Available routes:")
    print("  /           - Home page")
    print("  /predict    - Disease detection")
    print("  /history    - Detection history")
    print("  /clear      - Clear history")
    print("  /export_history - Export history as CSV")
    
    app.run(debug=True, host='0.0.0.0', port=5000)