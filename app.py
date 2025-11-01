from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

# ============================================================
# ⚙️ Flask setup
# ============================================================
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ============================================================
# 🧠 Load Model and Labels
# ============================================================
print("🔹 Loading plant disease model...")
model = tf.keras.models.load_model('plant_disease_model.h5')
print("✅ Model loaded successfully!")

# Load class labels
if os.path.exists("class_labels.txt"):
    with open("class_labels.txt", "r") as f:
        class_labels = [line.strip() for line in f.readlines()]
else:
    class_labels = sorted(os.listdir('dataset/train'))

print("✅ Class labels loaded:", class_labels)


# ============================================================
# 🧩 Label Cleaner Function
# ============================================================
def clean_label(label):
    """Clean folder names to make them readable and consistent."""
    name = label.replace("___", " ").replace("__", " ").replace("_", " ").title()

    # Fix duplicate crop names like 'Tomato Tomato'
    parts = name.split()
    cleaned_parts = []
    for part in parts:
        if not cleaned_parts or cleaned_parts[-1].lower() != part.lower():
            cleaned_parts.append(part)
    return " ".join(cleaned_parts)


# ============================================================
# 🌿 Disease Information (Detailed)
# ============================================================
disease_info = {
    "Pepper Bell Bacterial Spot": {
        "description": (
            "A bacterial disease that begins as small, dark, water-soaked spots on pepper leaves and fruits. "
            "As the infection progresses, the spots enlarge and cause leaves to drop. "
            "It spreads easily in warm, humid conditions and may also infect fruits, reducing quality."
        ),
        "treatment": [
            "1. Remove all infected leaves and fruits immediately to prevent further spread.",
            "2. Avoid overhead watering and keep foliage dry whenever possible.",
            "3. Apply copper-based fungicides weekly. Clean tools and use disease-free seeds for prevention."
        ]
    },
    "Pepper Bell Healthy": {
        "description": (
            "The pepper plant appears green, strong, and healthy with no visible infection. "
            "Leaves are smooth and firm, showing no signs of spots or yellowing."
        ),
        "treatment": [
            "1. Continue balanced watering and avoid excessive fertilizer usage.",
            "2. Inspect plants weekly for any early disease symptoms.",
            "3. Ensure proper sunlight and spacing between plants for airflow."
        ]
    },
    "Potato Early Blight": {
        "description": (
            "A fungal disease caused by Alternaria solani, producing brown spots with concentric rings on older leaves. "
            "It thrives in warm, humid conditions and weakens plants gradually. "
            "Severe cases lead to leaf drop and reduced tuber yield."
        ),
        "treatment": [
            "1. Spray fungicides like mancozeb or chlorothalonil every 7–10 days during humid weather.",
            "2. Remove and destroy infected leaves to reduce spore spread.",
            "3. Rotate crops annually and avoid planting potatoes near tomato fields."
        ]
    },
    "Potato Late Blight": {
        "description": (
            "A destructive fungal disease caused by Phytophthora infestans that creates dark, water-soaked lesions. "
            "It affects both leaves and tubers and spreads rapidly in cool, wet conditions."
        ),
        "treatment": [
            "1. Remove affected plants and dispose of them away from the field.",
            "2. Apply copper-based fungicides at regular intervals.",
            "3. Avoid reusing infected soil and maintain good drainage around potato beds."
        ]
    },
    "Potato Healthy": {
        "description": (
            "The potato plant shows no signs of disease or stress. "
            "Leaves are vibrant green, and tuber development is healthy."
        ),
        "treatment": [
            "1. Continue regular watering, ensuring the soil is moist but not waterlogged.",
            "2. Inspect leaves weekly for early signs of pests or disease.",
            "3. Use organic compost to promote soil fertility and healthy growth."
        ]
    },
    "Tomato Early Blight": {
        "description": (
            "A common fungal disease that begins on older leaves as brown spots with concentric rings. "
            "Leaves turn yellow and drop prematurely, reducing fruit yield. "
            "It spreads easily through rain splash and wind."
        ),
        "treatment": [
            "1. Remove infected leaves and avoid watering from above.",
            "2. Apply chlorothalonil-based fungicides every 7 days during humid weather.",
            "3. Improve air circulation by proper pruning and spacing between plants."
        ]
    },
    "Tomato Late Blight": {
        "description": (
            "A serious fungal infection that causes large, dark, greasy-looking lesions on leaves and fruits. "
            "It can rapidly destroy tomato crops under moist conditions."
        ),
        "treatment": [
            "1. Destroy infected plants and apply copper fungicide immediately.",
            "2. Avoid overhead irrigation and overcrowding.",
            "3. Monitor weather conditions and spray preventively during wet seasons."
        ]
    },
    "Tomato Leaf Mold": {
        "description": (
            "A fungal disease that forms yellow patches on upper leaf surfaces and gray mold underneath. "
            "It thrives in poorly ventilated and humid greenhouses."
        ),
        "treatment": [
            "1. Ensure proper airflow by pruning dense foliage and using fans in greenhouses.",
            "2. Spray sulfur or copper-based fungicides weekly until symptoms disappear.",
            "3. Avoid overhead watering to keep leaves dry."
        ]
    },
    "Tomato Target Spot": {
        "description": (
            "A fungal disease producing brown circular lesions with pale centers on leaves and fruits. "
            "In severe cases, it leads to premature leaf drop and smaller fruits."
        ),
        "treatment": [
            "1. Apply mancozeb fungicide regularly during humid conditions.",
            "2. Remove damaged leaves and ensure good airflow around plants.",
            "3. Use crop rotation and avoid excess nitrogen fertilizer."
        ]
    },
    "Tomato Septoria Leaf Spot": {
        "description": (
            "A fungal leaf disease that starts as small, dark spots with light centers. "
            "It spreads rapidly under wet conditions, leading to significant defoliation."
        ),
        "treatment": [
            "1. Remove affected leaves and apply fungicides containing copper or chlorothalonil.",
            "2. Avoid wetting foliage and ensure good air circulation.",
            "3. Disinfect garden tools regularly to prevent spread."
        ]
    },
    "Tomato Yellowleaf Curl Virus": {
        "description": (
            "A viral infection spread by whiteflies, causing yellowing and curling of leaves. "
            "The plant becomes stunted and fruit production drops drastically."
        ),
        "treatment": [
            "1. Use insecticidal soap or neem oil to control whiteflies.",
            "2. Remove and destroy infected plants immediately to prevent spread.",
            "3. Use virus-resistant tomato varieties for long-term protection."
        ]
    },
    "Tomato Mosaic Virus": {
        "description": (
            "A viral disease causing mottled and curled leaves, leading to stunted plant growth. "
            "It spreads through contact with contaminated tools, hands, or tobacco products."
        ),
        "treatment": [
            "1. Remove infected plants and disinfect all garden tools.",
            "2. Avoid handling plants after using tobacco products.",
            "3. Grow resistant varieties and maintain field hygiene."
        ]
    },
    "Tomato Bacterial Spot": {
        "description": (
            "A bacterial infection producing black, raised spots on leaves and fruit. "
            "It can severely reduce market quality and cause leaf drop in humid conditions."
        ),
        "treatment": [
            "1. Use copper-based bactericides every week during high humidity.",
            "2. Avoid working with wet plants and disinfect equipment frequently.",
            "3. Practice crop rotation and use certified disease-free seeds."
        ]
    },
    "Tomato Spider Mites Two Spotted Spider Mite": {
        "description": (
            "Tiny spider mites that feed on leaves, causing yellow stippling and webbing. "
            "Severe infestations can lead to leaf drop and reduced photosynthesis."
        ),
        "treatment": [
            "1. Spray plants with water or insecticidal soap to dislodge mites.",
            "2. Maintain humidity, as mites thrive in dry conditions.",
            "3. Introduce natural predators like predatory mites for biological control."
        ]
    },
    "Tomato Healthy": {
        "description": (
            "The tomato plant appears healthy and vigorous with no visible disease. "
            "Leaves are green and smooth, and fruits develop evenly."
        ),
        "treatment": [
            "1. Maintain regular watering and fertilization.",
            "2. Inspect plants weekly for early pest signs.",
            "3. Support plants with stakes and remove yellow leaves regularly."
        ]
    }
}


# ============================================================
# 🌍 ROUTES
# ============================================================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze')
def analyze():
    return render_template('analyze.html')


# ============================================================
# 🌿 Image Upload and Analysis
# ============================================================
@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'})

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'})

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print("✅ Image saved:", filepath)

        # Preprocess uploaded image
        img = image.load_img(filepath, target_size=(64, 64))
        img_array = image.img_to_array(img)
        if img_array.shape[-1] == 1:
            img_array = np.repeat(img_array, 3, axis=-1)
        elif img_array.shape[-1] == 4:
            img_array = img_array[..., :3]
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        preds = model.predict(img_array)
        predicted_class = np.argmax(preds[0])
        confidence = round(float(np.max(preds[0])) * 100, 2)
        disease_raw = class_labels[predicted_class]
        disease_clean = clean_label(disease_raw)

        print(f"🌿 Predicted: {disease_clean} | Confidence: {confidence}%")

        # Fetch disease info safely
        info = disease_info.get(disease_clean, {
            "description": "No detailed information available for this disease.",
            "treatment": ["Follow general crop care and hygiene practices."]
        })

        # Format description with newlines
        description_sentences = info['description'].split(". ")
        description_formatted = "\n".join([f"{s.strip()}." for s in description_sentences if s])

        # Build report
        report = f"{description_formatted}"

        # Build recommended actions separately
        recommended_actions = ""
        for t in info["treatment"]:
            recommended_actions += f"{t}\n"

        # Return both separately in the response
        return jsonify({
            "disease": disease_clean,
            "confidence": f"{confidence}%",
            "ai_report": report,
            "recommended_actions": recommended_actions
        })

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({'error': '⚠️ Could not analyze the image. Please try again.'})


# ============================================================
# 🚀 Run App
# ============================================================
if __name__ == '__main__':
    app.run(debug=True)
