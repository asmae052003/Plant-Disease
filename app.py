
import os
import torch
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
from model import AdvancedLocalGlobalNet, Config, get_transforms, extract_random_patches

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Model
# Placeholder class names - REPLACE THESE WITH ACTUAL CLASS NAMES
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

num_classes = len(CLASS_NAMES) # Should match the model's output size (38)

model = AdvancedLocalGlobalNet(num_classes=num_classes, use_attention=Config.use_attention)
try:
    checkpoint = torch.load('best_model_final.pth', map_location=device, weights_only=False)
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict):
         # Try loading directly if keys match, otherwise might need adjustment
        try:
            model.load_state_dict(checkpoint)
        except RuntimeError as e:
            print(f"Error loading dictionary: {e}")
            # Fallback or specific key handling if needed
    else:
        model = checkpoint # If entire model was saved
    
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure 'best_model_final.pth' is in the correct directory and matches the architecture.")

# Transforms
val_transform, local_transform = get_transforms()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_url = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_url = url_for('static', filename=f'uploads/{filename}')
            
            # Prediction
            try:
                img = Image.open(filepath).convert('RGB')
                
                # Global image
                img_global = val_transform(img).unsqueeze(0).to(device)
                
                # Local patches
                patches = extract_random_patches(img, num_patches=Config.num_patches, patch_size=Config.img_size_local)
                local_tensors = [local_transform(p) for p in patches]
                patches_tensor = torch.stack(local_tensors, dim=0).unsqueeze(0).to(device) # Add batch dim
                
                with torch.no_grad():
                    logits, _ = model(img_global, patches_tensor)
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    confidence, predicted_idx = torch.max(probs, 1)
                    
                    predicted_class = CLASS_NAMES[predicted_idx.item()] if predicted_idx.item() < len(CLASS_NAMES) else f"Class {predicted_idx.item()}"
                    confidence_score = confidence.item() * 100
                    
                    prediction = {
                        'class': predicted_class,
                        'confidence': f"{confidence_score:.2f}%"
                    }
            except Exception as e:
                print(f"Prediction error: {e}")
                prediction = {'error': str(e)}

    return render_template('index.html', prediction=prediction, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)
