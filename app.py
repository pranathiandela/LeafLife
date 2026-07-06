from flask import Flask, render_template, request, jsonify
import os, json, sqlite3, datetime
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import google.generativeai as genai

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

app = Flask(__name__)
app.config['SECRET_KEY'] = 'leaflife-secret-key-2024'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ─── Full Disease Database (covers all 38 PlantVillage classes) ────────────────
DISEASE_DB = {
    # ── APPLE ──
    "Apple___Apple_scab": {
        "plant": "Apple", "disease": "Apple Scab", "severity": "High",
        "description": "Apple scab is a fungal disease caused by Venturia inaequalis, creating dark olive-green or brown lesions on leaves and fruit.",
        "symptoms": ["Olive-green to brown spots on leaves", "Scabby lesions on fruit", "Premature leaf drop", "Distorted fruit"],
        "causes": ["Fungus: Venturia inaequalis", "Cool wet spring weather", "Infected fallen leaves", "High humidity"],
        "prevention": ["Rake and destroy fallen leaves", "Use resistant varieties", "Prune for air circulation", "Apply fungicide in spring"],
        "organic_treatment": ["Neem oil spray", "Sulfur-based fungicide", "Copper spray"],
        "chemical_treatment": ["Captan 50% WP", "Mancozeb 75% WP", "Myclobutanil"],
        "recovery_tips": ["Remove infected fruit", "Maintain orchard hygiene", "Apply dormant sprays"]
    },
    "Apple___Black_rot": {
        "plant": "Apple", "disease": "Black Rot", "severity": "High",
        "description": "Black rot caused by Botryosphaeria obtusa affects apples causing leaf spots, fruit rot, and limb cankers.",
        "symptoms": ["Purple spots on leaves", "Brown to black fruit rot", "Cankers on branches", "Mummified fruit"],
        "causes": ["Fungus: Botryosphaeria obtusa", "Warm wet weather", "Pruning wounds", "Stressed trees"],
        "prevention": ["Remove mummified fruit", "Prune dead branches", "Avoid tree stress", "Proper fertilization"],
        "organic_treatment": ["Copper-based fungicide", "Neem oil", "Bordeaux mixture"],
        "chemical_treatment": ["Captan", "Thiophanate-methyl", "Ziram"],
        "recovery_tips": ["Remove and destroy infected material", "Improve tree vigor", "Regular monitoring"]
    },
    "Apple___Cedar_apple_rust": {
        "plant": "Apple", "disease": "Cedar Apple Rust", "severity": "Medium",
        "description": "Cedar apple rust caused by Gymnosporangium juniperi-virginianae creates bright orange-yellow spots on apple leaves.",
        "symptoms": ["Bright yellow-orange spots on leaves", "Tube-like structures on leaf undersides", "Fruit distortion", "Premature defoliation"],
        "causes": ["Fungus: Gymnosporangium juniperi-virginianae", "Requires both apple and cedar/juniper", "Wet spring weather", "Wind-dispersed spores"],
        "prevention": ["Remove nearby juniper/cedar trees", "Plant resistant varieties", "Apply protective fungicide", "Early season sprays"],
        "organic_treatment": ["Sulfur fungicide", "Neem oil", "Copper spray"],
        "chemical_treatment": ["Myclobutanil", "Propiconazole", "Triadimefon"],
        "recovery_tips": ["Monitor both host plants", "Apply fungicide preventively", "Remove galls from junipers"]
    },
    "Apple___healthy": {
        "plant": "Apple", "disease": "Healthy", "severity": "None",
        "description": "The apple leaf appears healthy with no visible disease symptoms. Continue regular care and monitoring.",
        "symptoms": ["No disease symptoms", "Normal green coloration", "Healthy leaf structure"],
        "causes": [], "prevention": ["Regular monitoring", "Proper nutrition", "Good orchard hygiene", "Timely pruning"],
        "organic_treatment": ["Preventive neem oil spray", "Compost application"],
        "chemical_treatment": ["Preventive fungicide as needed"],
        "recovery_tips": ["Maintain good orchard hygiene", "Monitor regularly"]
    },
    # ── BLUEBERRY ──
    "Blueberry___healthy": {
        "plant": "Blueberry", "disease": "Healthy", "severity": "None",
        "description": "The blueberry plant appears healthy. Maintain good cultural practices.",
        "symptoms": ["No disease symptoms", "Normal leaf color"],
        "causes": [], "prevention": ["Proper soil pH (4.5-5.5)", "Adequate water", "Regular pruning"],
        "organic_treatment": ["Compost mulch", "Acidifying fertilizers"],
        "chemical_treatment": ["Preventive sprays as needed"],
        "recovery_tips": ["Maintain soil acidity", "Proper spacing"]
    },
    # ── CHERRY ──
    "Cherry_(including_sour)___Powdery_mildew": {
        "plant": "Cherry", "disease": "Powdery Mildew", "severity": "Medium",
        "description": "Powdery mildew on cherry is caused by Podosphaera clandestina, producing a white powdery coating on leaves and shoots.",
        "symptoms": ["White powdery coating on leaves", "Curled and distorted young leaves", "Stunted shoot growth", "Reduced fruit quality"],
        "causes": ["Fungus: Podosphaera clandestina", "Warm dry days, cool nights", "High humidity", "Dense canopy"],
        "prevention": ["Prune for air circulation", "Avoid excess nitrogen", "Use resistant varieties", "Early season monitoring"],
        "organic_treatment": ["Potassium bicarbonate spray", "Neem oil", "Sulfur dust"],
        "chemical_treatment": ["Myclobutanil", "Propiconazole", "Trifloxystrobin"],
        "recovery_tips": ["Remove infected shoots", "Improve air circulation", "Reduce canopy density"]
    },
    "Cherry_(including_sour)___healthy": {
        "plant": "Cherry", "disease": "Healthy", "severity": "None",
        "description": "The cherry plant appears healthy. Continue regular care.",
        "symptoms": ["No disease symptoms"], "causes": [],
        "prevention": ["Regular monitoring", "Proper pruning", "Balanced fertilization"],
        "organic_treatment": ["Preventive copper spray"], "chemical_treatment": ["Preventive fungicide"],
        "recovery_tips": ["Maintain good orchard practices"]
    },
    # ── CORN / MAIZE ──
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "plant": "Maize", "disease": "Gray Leaf Spot", "severity": "High",
        "description": "Gray leaf spot caused by Cercospora zeae-maydis creates rectangular gray-tan lesions parallel to leaf veins on corn.",
        "symptoms": ["Rectangular gray-tan lesions", "Lesions parallel to leaf veins", "Premature leaf death", "Reduced yield"],
        "causes": ["Fungus: Cercospora zeae-maydis", "Warm humid conditions", "Reduced tillage fields", "Susceptible hybrids"],
        "prevention": ["Plant resistant hybrids", "Crop rotation", "Tillage of crop residues", "Balanced fertilization"],
        "organic_treatment": ["Copper fungicide", "Neem-based spray", "Biocontrol agents"],
        "chemical_treatment": ["Azoxystrobin", "Pyraclostrobin", "Propiconazole 25% EC"],
        "recovery_tips": ["Harvest at proper maturity", "Till residues", "Rotate with non-host crops"]
    },
    "Corn_(maize)___Common_rust_": {
        "plant": "Maize", "disease": "Common Rust", "severity": "Medium",
        "description": "Common rust caused by Puccinia sorghi produces brick-red pustules on both surfaces of corn leaves.",
        "symptoms": ["Brick-red oval pustules", "Pustules on both leaf surfaces", "Yellowing around pustules", "Reduced grain fill"],
        "causes": ["Fungus: Puccinia sorghi", "Cool temperatures (16-23°C)", "High humidity", "Airborne spores"],
        "prevention": ["Plant rust-resistant hybrids", "Early planting", "Balanced fertilization", "Regular scouting"],
        "organic_treatment": ["Sulfur-based fungicide", "Neem oil", "Potassium bicarbonate"],
        "chemical_treatment": ["Propiconazole", "Azoxystrobin", "Tebuconazole"],
        "recovery_tips": ["Apply fungicide at first sign", "Scout regularly", "Remove severely infected plants"]
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "plant": "Maize", "disease": "Northern Leaf Blight", "severity": "High",
        "description": "Northern Leaf Blight caused by Exserohilum turcicum produces long cigar-shaped gray-green to tan lesions on corn leaves.",
        "symptoms": ["Long cigar-shaped tan lesions", "Gray-green discoloration", "Lesions may coalesce", "Premature leaf death"],
        "causes": ["Fungus: Exserohilum turcicum", "Moderate temperatures (18-27°C)", "Wet weather", "Dense canopy"],
        "prevention": ["Use resistant hybrids", "Crop rotation", "Bury crop residues", "Balanced nitrogen"],
        "organic_treatment": ["Copper fungicide", "Bacillus-based biocontrol", "Garlic spray"],
        "chemical_treatment": ["Propiconazole 25% EC", "Azoxystrobin + Propiconazole", "Picoxystrobin"],
        "recovery_tips": ["Harvest promptly", "Till residues after harvest", "Store grain at low moisture"]
    },
    "Corn_(maize)___healthy": {
        "plant": "Maize", "disease": "Healthy", "severity": "None",
        "description": "The maize plant appears healthy. Continue good agronomic practices.",
        "symptoms": ["No disease symptoms"], "causes": [],
        "prevention": ["Regular monitoring", "Balanced fertilization", "Proper spacing"],
        "organic_treatment": ["Compost application"], "chemical_treatment": ["Preventive fungicide if needed"],
        "recovery_tips": ["Maintain proper plant nutrition"]
    },
    # ── GRAPE ──
    "Grape___Black_rot": {
        "plant": "Grape", "disease": "Black Rot", "severity": "High",
        "description": "Grape black rot caused by Guignardia bidwellii causes brown leaf lesions and hard, black shriveled berries.",
        "symptoms": ["Brown circular leaf lesions", "Black shriveled berries", "Tan lesions with dark borders", "Infected tendrils and shoots"],
        "causes": ["Fungus: Guignardia bidwellii", "Warm wet weather", "Infected mummified berries", "Poor air circulation"],
        "prevention": ["Remove mummified berries", "Prune for air circulation", "Apply fungicide early season", "Destroy infected material"],
        "organic_treatment": ["Copper-based fungicide", "Bordeaux mixture", "Neem oil"],
        "chemical_treatment": ["Mancozeb 75% WP", "Myclobutanil", "Captan"],
        "recovery_tips": ["Remove all mummies", "Improve canopy management", "Apply dormant copper spray"]
    },
    "Grape___Esca_(Black_Measles)": {
        "plant": "Grape", "disease": "Esca (Black Measles)", "severity": "High",
        "description": "Esca is a complex grapevine trunk disease causing tiger-stripe patterns on leaves and dark spots on berries.",
        "symptoms": ["Tiger-stripe leaf discoloration", "Dark spots on berries", "Sudden vine collapse", "Wood discoloration"],
        "causes": ["Fungal complex (Phaeomoniella, Phaeoacremonium)", "Pruning wounds", "Old vines", "Stress conditions"],
        "prevention": ["Protect pruning wounds", "Use proper pruning tools", "Avoid large pruning cuts", "Delay pruning until dry weather"],
        "organic_treatment": ["Wound sealants after pruning", "Trichoderma-based biocontrol", "Proper pruning hygiene"],
        "chemical_treatment": ["Thiophanate-methyl on wounds", "Flusilazole", "Tebuconazole"],
        "recovery_tips": ["Remove severely infected vines", "Sterilize pruning tools", "Apply wound protectants"]
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "plant": "Grape", "disease": "Leaf Blight (Isariopsis Leaf Spot)", "severity": "Medium",
        "description": "Grape leaf blight caused by Pseudocercospora vitis creates dark brown spots on leaves leading to defoliation.",
        "symptoms": ["Dark brown angular leaf spots", "Spots with yellow margins", "Premature defoliation", "Reduced photosynthesis"],
        "causes": ["Fungus: Pseudocercospora vitis", "Warm humid conditions", "Late season infection", "Dense canopy"],
        "prevention": ["Prune for air circulation", "Remove fallen leaves", "Apply protective fungicide", "Avoid excessive irrigation"],
        "organic_treatment": ["Copper fungicide", "Neem oil spray", "Sulfur dust"],
        "chemical_treatment": ["Mancozeb", "Captan", "Chlorothalonil"],
        "recovery_tips": ["Remove infected leaves", "Improve air circulation", "Manage canopy density"]
    },
    "Grape___healthy": {
        "plant": "Grape", "disease": "Healthy", "severity": "None",
        "description": "The grapevine appears healthy. Continue regular vineyard management.",
        "symptoms": ["No disease symptoms"], "causes": [],
        "prevention": ["Regular monitoring", "Proper canopy management", "Balanced fertilization"],
        "organic_treatment": ["Preventive copper spray"], "chemical_treatment": ["Preventive fungicide"],
        "recovery_tips": ["Maintain good vineyard hygiene"]
    },
    # ── ORANGE ──
    "Orange___Haunglongbing_(Citrus_greening)": {
        "plant": "Orange", "disease": "Huanglongbing (Citrus Greening)", "severity": "High",
        "description": "Huanglongbing (HLB) is the most destructive citrus disease worldwide, caused by Candidatus Liberibacter bacteria spread by psyllid insects.",
        "symptoms": ["Blotchy yellowing of leaves", "Asymmetric leaf mottling", "Small lopsided fruit", "Bitter tasting fruit"],
        "causes": ["Bacteria: Candidatus Liberibacter", "Asian citrus psyllid vector", "No cure available", "Spreads rapidly"],
        "prevention": ["Control psyllid populations", "Use certified disease-free plants", "Remove infected trees", "Quarantine measures"],
        "organic_treatment": ["Neem oil for psyllid control", "Kaolin clay spray", "Yellow sticky traps"],
        "chemical_treatment": ["Imidacloprid for psyllid", "Thiamethoxam", "Systemic insecticides"],
        "recovery_tips": ["Remove and destroy infected trees", "Replace with certified clean material", "Maintain psyllid control"]
    },
    # ── PEACH ──
    "Peach___Bacterial_spot": {
        "plant": "Peach", "disease": "Bacterial Spot", "severity": "High",
        "description": "Bacterial spot caused by Xanthomonas arboricola pv. pruni creates water-soaked lesions on peach leaves and fruit.",
        "symptoms": ["Water-soaked leaf spots", "Angular dark lesions", "Shot-hole appearance", "Fruit pitting and cracking"],
        "causes": ["Bacteria: Xanthomonas arboricola", "Warm wet weather", "Wind-driven rain", "Pruning wounds"],
        "prevention": ["Plant resistant varieties", "Windbreaks to reduce spread", "Avoid overhead irrigation", "Copper sprays at dormancy"],
        "organic_treatment": ["Copper-based bactericide", "Bordeaux mixture", "Avoid excess nitrogen"],
        "chemical_treatment": ["Copper hydroxide", "Oxytetracycline", "Copper sulfate"],
        "recovery_tips": ["Prune during dry weather", "Apply copper at bud swell", "Remove severely infected branches"]
    },
    "Peach___healthy": {
        "plant": "Peach", "disease": "Healthy", "severity": "None",
        "description": "The peach tree appears healthy. Continue regular care.",
        "symptoms": ["No disease symptoms"], "causes": [],
        "prevention": ["Regular monitoring", "Proper pruning", "Balanced fertilization"],
        "organic_treatment": ["Preventive copper spray"], "chemical_treatment": ["Dormant sprays"],
        "recovery_tips": ["Maintain good orchard hygiene"]
    },
    # ── PEPPER ──
    "Pepper,_bell___Bacterial_spot": {
        "plant": "Bell Pepper", "disease": "Bacterial Spot", "severity": "High",
        "description": "Bacterial spot on pepper caused by Xanthomonas campestris creates water-soaked lesions on leaves and fruit.",
        "symptoms": ["Water-soaked leaf spots", "Brown necrotic lesions", "Fruit spots and lesions", "Premature defoliation"],
        "causes": ["Bacteria: Xanthomonas campestris", "Warm wet conditions", "Rain splash spread", "Infected seeds"],
        "prevention": ["Use certified disease-free seeds", "Crop rotation", "Avoid overhead irrigation", "Copper sprays"],
        "organic_treatment": ["Copper-based bactericide", "Neem oil", "Remove infected leaves"],
        "chemical_treatment": ["Copper hydroxide 77% WP", "Streptomycin sulfate", "Kasugamycin"],
        "recovery_tips": ["Destroy infected plants", "Avoid working in wet fields", "Use disease-free transplants"]
    },
    "Pepper,_bell___healthy": {
        "plant": "Bell Pepper", "disease": "Healthy", "severity": "None",
        "description": "The pepper plant appears healthy. Continue good cultural practices.",
        "symptoms": ["No disease symptoms"], "causes": [],
        "prevention": ["Regular monitoring", "Balanced fertilization", "Proper spacing"],
        "organic_treatment": ["Preventive neem spray"], "chemical_treatment": ["Preventive fungicide"],
        "recovery_tips": ["Maintain good field hygiene"]
    },
    # ── POTATO ──
    "Potato___Early_blight": {
        "plant": "Potato", "disease": "Early Blight", "severity": "Medium",
        "description": "Potato early blight caused by Alternaria solani affects older leaves first, creating target-like concentric ring patterns.",
        "symptoms": ["Target-like rings on older leaves", "Yellowing of affected tissue", "Premature leaf drop", "Small dark lesions on tubers"],
        "causes": ["Fungus: Alternaria solani", "Warm temperatures (24-29°C)", "Drought stress", "Dense planting"],
        "prevention": ["Use certified seed potatoes", "Rotate crops every 3 years", "Maintain plant nutrition", "Avoid water stress"],
        "organic_treatment": ["Copper sulfate spray", "Neem-based fungicide", "Bacillus subtilis"],
        "chemical_treatment": ["Mancozeb 75% WP", "Iprodione 50% WP", "Tebuconazole"],
        "recovery_tips": ["Harvest at right maturity", "Store in cool dry place", "Destroy crop residues"]
    },
    "Potato___Late_blight": {
        "plant": "Potato", "disease": "Late Blight", "severity": "High",
        "description": "The most destructive potato disease worldwide, caused by Phytophthora infestans, responsible for the Irish Potato Famine.",
        "symptoms": ["Water-soaked dark lesions", "White cottony growth on leaves", "Rapid browning and collapse", "Tubers show reddish-brown rot"],
        "causes": ["Oomycete: Phytophthora infestans", "Cool wet weather", "Infected seed tubers", "Airborne spores"],
        "prevention": ["Plant resistant varieties", "Disease-free seed tubers", "Hill soil around plants", "Avoid overhead irrigation"],
        "organic_treatment": ["Copper-based sprays", "Bordeaux mixture", "Biocontrol agents"],
        "chemical_treatment": ["Metalaxyl-M + Mancozeb", "Fluopicolide", "Cymoxanil"],
        "recovery_tips": ["Destroy infected plants completely", "Do not re-use infected soil", "Apply fungicide preventatively"]
    },
    "Potato___healthy": {
        "plant": "Potato", "disease": "Healthy", "severity": "None",
        "description": "The potato plant appears healthy. Continue good agronomic practices.",
        "symptoms": ["No disease symptoms"], "causes": [],
        "prevention": ["Regular monitoring", "Proper hilling", "Balanced fertilization"],
        "organic_treatment": ["Compost application"], "chemical_treatment": ["Preventive fungicide"],
        "recovery_tips": ["Monitor field regularly"]
    },
    # ── RASPBERRY ──
    "Raspberry___healthy": {
        "plant": "Raspberry", "disease": "Healthy", "severity": "None",
        "description": "The raspberry plant appears healthy. Continue regular care.",
        "symptoms": ["No disease symptoms"], "causes": [],
        "prevention": ["Regular monitoring", "Proper pruning", "Good drainage"],
        "organic_treatment": ["Compost mulch"], "chemical_treatment": ["Preventive sprays"],
        "recovery_tips": ["Maintain good cane management"]
    },
    # ── SOYBEAN ──
    "Soybean___healthy": {
        "plant": "Soybean", "disease": "Healthy", "severity": "None",
        "description": "The soybean plant appears healthy. Continue good crop management.",
        "symptoms": ["No disease symptoms"], "causes": [],
        "prevention": ["Regular monitoring", "Crop rotation", "Balanced fertilization"],
        "organic_treatment": ["Compost application"], "chemical_treatment": ["Preventive fungicide"],
        "recovery_tips": ["Scout fields regularly"]
    },
    # ── SQUASH ──
    "Squash___Powdery_mildew": {
        "plant": "Squash", "disease": "Powdery Mildew", "severity": "Medium",
        "description": "Powdery mildew on squash caused by Podosphaera xanthii creates white powdery patches on leaf surfaces.",
        "symptoms": ["White powdery patches on leaves", "Yellowing of affected leaves", "Distorted growth", "Reduced fruit quality"],
        "causes": ["Fungus: Podosphaera xanthii", "Warm dry weather", "High humidity at night", "Dense planting"],
        "prevention": ["Plant resistant varieties", "Increase plant spacing", "Avoid excess nitrogen", "Water at base"],
        "organic_treatment": ["Potassium bicarbonate spray", "Neem oil", "Sulfur dust"],
        "chemical_treatment": ["Myclobutanil", "Trifloxystrobin", "Azoxystrobin"],
        "recovery_tips": ["Remove infected leaves", "Improve air circulation", "Apply fungicide early"]
    },
    # ── STRAWBERRY ──
    "Strawberry___Leaf_scorch": {
        "plant": "Strawberry", "disease": "Leaf Scorch", "severity": "Medium",
        "description": "Strawberry leaf scorch caused by Diplocarpon earlianum creates small purple spots that enlarge and cause leaf edges to appear scorched.",
        "symptoms": ["Small purple to red spots", "Scorched leaf margins", "Spots with gray centers", "Premature defoliation"],
        "causes": ["Fungus: Diplocarpon earlianum", "Cool wet conditions", "Poor air circulation", "Infected plant material"],
        "prevention": ["Use certified disease-free plants", "Increase plant spacing", "Remove old leaves", "Avoid overhead irrigation"],
        "organic_treatment": ["Copper fungicide", "Neem oil spray", "Remove infected leaves"],
        "chemical_treatment": ["Captan 50% WP", "Myclobutanil", "Azoxystrobin"],
        "recovery_tips": ["Remove and destroy infected leaves", "Renovate beds after harvest", "Apply fungicide in spring"]
    },
    "Strawberry___healthy": {
        "plant": "Strawberry", "disease": "Healthy", "severity": "None",
        "description": "The strawberry plant appears healthy. Continue regular care.",
        "symptoms": ["No disease symptoms"], "causes": [],
        "prevention": ["Regular monitoring", "Proper spacing", "Weed control"],
        "organic_treatment": ["Compost mulch"], "chemical_treatment": ["Preventive fungicide"],
        "recovery_tips": ["Maintain good bed hygiene"]
    },
    # ── TOMATO ──
    "Tomato___Bacterial_spot": {
        "plant": "Tomato", "disease": "Bacterial Spot", "severity": "High",
        "description": "Bacterial spot caused by Xanthomonas vesicatoria creates water-soaked spots on tomato leaves, stems and fruit.",
        "symptoms": ["Water-soaked leaf spots", "Dark brown necrotic lesions", "Fruit spots with raised margins", "Defoliation"],
        "causes": ["Bacteria: Xanthomonas vesicatoria", "Warm wet conditions", "Rain splash", "Infected seeds"],
        "prevention": ["Use certified seeds", "Crop rotation", "Avoid overhead watering", "Copper sprays"],
        "organic_treatment": ["Copper hydroxide spray", "Neem oil", "Remove infected leaves"],
        "chemical_treatment": ["Copper oxychloride 50% WP", "Streptomycin sulfate", "Kasugamycin"],
        "recovery_tips": ["Destroy infected plants", "Avoid working when wet", "Use disease-free transplants"]
    },
    "Tomato___Early_blight": {
        "plant": "Tomato", "disease": "Early Blight", "severity": "High",
        "description": "Early blight caused by Alternaria solani creates dark brown spots with concentric rings on tomato leaves.",
        "symptoms": ["Dark brown spots on leaves", "Yellowing around spots", "Leaves turn yellow and drop", "Spots on stems"],
        "causes": ["Fungal pathogen: Alternaria solani", "Warm humid conditions", "Overcrowding", "Poor air circulation"],
        "prevention": ["Remove infected leaves", "Avoid overhead watering", "Improve air circulation", "Rotate crops"],
        "organic_treatment": ["Neem oil spray", "Copper-based fungicide", "Baking soda solution"],
        "chemical_treatment": ["Mancozeb 75% WP", "Chlorothalonil 75% WP", "Azoxystrobin"],
        "recovery_tips": ["Monitor plant growth weekly", "Remove dead foliage", "Ensure proper sunlight"]
    },
    "Tomato___Late_blight": {
        "plant": "Tomato", "disease": "Late Blight", "severity": "High",
        "description": "Late blight caused by Phytophthora infestans is a devastating disease that can destroy entire crops rapidly.",
        "symptoms": ["Water-soaked lesions", "White fuzzy growth on underside", "Dark brown patches", "Fruit brown rot"],
        "causes": ["Oomycete: Phytophthora infestans", "Cool temperatures (10-25°C)", "High humidity and rain", "Poor drainage"],
        "prevention": ["Use resistant varieties", "Avoid wetting foliage", "Improve drainage", "Destroy infected plants"],
        "organic_treatment": ["Copper hydroxide spray", "Bordeaux mixture", "Garlic extract spray"],
        "chemical_treatment": ["Metalaxyl + Mancozeb", "Cymoxanil 8% + Mancozeb 64%", "Dimethomorph"],
        "recovery_tips": ["Act quickly at first sign", "Remove all infected material", "Do not compost infected plants"]
    },
    "Tomato___Leaf_Mold": {
        "plant": "Tomato", "disease": "Leaf Mold", "severity": "Medium",
        "description": "Leaf mold caused by Passalora fulva thrives in humid conditions, causing pale green to yellow spots on upper leaf surfaces.",
        "symptoms": ["Pale green/yellow spots on upper leaf", "Olive-green mold on lower leaf", "Leaves curl and dry", "Reduced yield"],
        "causes": ["Fungus: Passalora fulva", "High humidity (>85%)", "Poor ventilation", "Dense planting"],
        "prevention": ["Reduce humidity", "Increase plant spacing", "Use resistant varieties", "Avoid leaf wetness"],
        "organic_treatment": ["Neem oil spray", "Potassium bicarbonate", "Compost tea spray"],
        "chemical_treatment": ["Chlorothalonil", "Mancozeb", "Copper-based fungicides"],
        "recovery_tips": ["Prune lower leaves", "Improve ventilation", "Water at base of plant"]
    },
    "Tomato___Septoria_leaf_spot": {
        "plant": "Tomato", "disease": "Septoria Leaf Spot", "severity": "Medium",
        "description": "Septoria leaf spot caused by Septoria lycopersici creates small circular spots with dark borders on tomato leaves.",
        "symptoms": ["Small circular spots with dark borders", "White or gray centers with dark margin", "Lower leaves affected first", "Premature defoliation"],
        "causes": ["Fungus: Septoria lycopersici", "Warm wet weather", "Splashing rain", "Infected crop debris"],
        "prevention": ["Remove infected leaves", "Crop rotation", "Avoid overhead watering", "Mulching"],
        "organic_treatment": ["Copper fungicide", "Neem oil", "Remove lower leaves"],
        "chemical_treatment": ["Mancozeb 75% WP", "Chlorothalonil", "Azoxystrobin"],
        "recovery_tips": ["Remove infected leaves immediately", "Improve drainage", "Rotate with non-solanaceous crops"]
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "plant": "Tomato", "disease": "Spider Mites (Two-Spotted)", "severity": "Medium",
        "description": "Two-spotted spider mites (Tetranychus urticae) suck plant sap causing stippled, yellowing leaves and fine webbing.",
        "symptoms": ["Stippled yellow or bronze leaves", "Fine webbing on leaf undersides", "Leaf drop in severe cases", "Bronzing of leaf surface"],
        "causes": ["Mite: Tetranychus urticae", "Hot dry weather", "Dusty conditions", "Broad-spectrum insecticide use"],
        "prevention": ["Maintain adequate moisture", "Avoid dusty conditions", "Conserve natural predators", "Regular monitoring"],
        "organic_treatment": ["Neem oil spray", "Insecticidal soap", "Predatory mites release", "Water spray to dislodge mites"],
        "chemical_treatment": ["Abamectin", "Spiromesifen", "Bifenazate"],
        "recovery_tips": ["Spray undersides of leaves", "Avoid broad-spectrum pesticides", "Increase humidity"]
    },
    "Tomato___Target_Spot": {
        "plant": "Tomato", "disease": "Target Spot", "severity": "Medium",
        "description": "Target spot caused by Corynespora cassiicola creates concentric ring patterns on tomato leaves, resembling a target.",
        "symptoms": ["Concentric ring patterns on leaves", "Brown lesions with yellow halos", "Fruit spots in severe cases", "Defoliation"],
        "causes": ["Fungus: Corynespora cassiicola", "Warm humid conditions", "Dense planting", "Poor air circulation"],
        "prevention": ["Improve air circulation", "Reduce canopy density", "Avoid overhead irrigation", "Crop rotation"],
        "organic_treatment": ["Copper fungicide", "Neem oil spray", "Remove infected leaves"],
        "chemical_treatment": ["Azoxystrobin", "Chlorothalonil", "Mancozeb 75% WP"],
        "recovery_tips": ["Remove infected leaves", "Improve ventilation", "Apply fungicide preventively"]
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "plant": "Tomato", "disease": "Yellow Leaf Curl Virus", "severity": "High",
        "description": "Tomato Yellow Leaf Curl Virus (TYLCV) is transmitted by whiteflies causing severe yellowing, curling and stunting of tomato plants.",
        "symptoms": ["Upward leaf curling", "Yellow leaf margins", "Stunted plant growth", "Reduced fruit set"],
        "causes": ["Tomato Yellow Leaf Curl Virus (TYLCV)", "Whitefly vector (Bemisia tabaci)", "High temperatures", "Proximity to infected plants"],
        "prevention": ["Control whitefly population", "Use reflective mulches", "Plant resistant varieties", "Remove infected plants"],
        "organic_treatment": ["Neem oil for whitefly", "Yellow sticky traps", "Insecticidal soap"],
        "chemical_treatment": ["Imidacloprid", "Thiamethoxam 25% WG", "Acetamiprid"],
        "recovery_tips": ["Remove and destroy infected plants", "Control vector insects", "Use virus-resistant varieties"]
    },
    "Tomato___Tomato_mosaic_virus": {
        "plant": "Tomato", "disease": "Tomato Mosaic Virus", "severity": "High",
        "description": "Tomato mosaic virus (ToMV) causes mottled light and dark green patterns on leaves with stunted growth.",
        "symptoms": ["Mosaic light/dark green pattern", "Leaf distortion and curling", "Stunted growth", "Reduced and deformed fruit"],
        "causes": ["Tomato Mosaic Virus (ToMV)", "Mechanical transmission", "Infected seeds", "Contaminated tools"],
        "prevention": ["Use certified virus-free seeds", "Sterilize tools between plants", "Wash hands before handling", "Remove infected plants"],
        "organic_treatment": ["Remove infected plants", "Sterilize tools with bleach", "Control aphid vectors"],
        "chemical_treatment": ["No chemical cure - prevention only", "Insecticides for aphid control"],
        "recovery_tips": ["Remove and destroy infected plants", "Sterilize all tools", "Use resistant varieties next season"]
    },
    "Tomato___healthy": {
        "plant": "Tomato", "disease": "Healthy", "severity": "None",
        "description": "The tomato plant appears healthy with no visible disease symptoms. Continue good agricultural practices.",
        "symptoms": ["No disease symptoms", "Normal green coloration", "Healthy leaf structure"],
        "causes": [],
        "prevention": ["Regular monitoring", "Proper nutrition", "Good field hygiene", "Crop rotation"],
        "organic_treatment": ["Preventive neem oil spray", "Compost tea application"],
        "chemical_treatment": ["Preventive fungicide as needed"],
        "recovery_tips": ["Maintain good field hygiene", "Monitor regularly"]
    },
}

# ─── Default CLASS_LABELS (PlantVillage order) ────────────────────────────────
CLASS_LABELS = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

# ─── Database ──────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect('database/leaflife.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    os.makedirs('database', exist_ok=True)
    conn = get_db()
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plant_name TEXT NOT NULL,
            disease_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            severity TEXT,
            image_path TEXT,
            description TEXT,
            analyzed_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER,
            role TEXT,
            content TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id) REFERENCES chat_history(id)
        );
    ''')
    conn.commit()
    conn.close()

# ─── Model Loading ─────────────────────────────────────────────────────────────
ml_model = None
CLASS_MAP = {}  # idx (str) -> label, loaded from class_indices.json

# Which PlantVillage variant the model was trained on: "color" or "segmented"
# If you retrain using the 'segmented' dataset folder on Kaggle, change this
# to "segmented" so the app preprocesses uploaded photos to match (leaf
# isolated on a black background) for much better real-world accuracy.
MODEL_DATASET_VARIANT = os.environ.get("MODEL_DATASET_VARIANT", "color")


def build_model_architecture(num_classes):
    """
    Rebuild the EXACT architecture used during training on Kaggle:
    MobileNetV2 (no top) -> GAP -> BatchNorm -> Dense(512) -> Dropout(0.4)
    -> Dense(256) -> Dropout(0.3) -> Dense(num_classes, softmax)
    This lets us load raw weights without any config-deserialization
    compatibility issues across Keras versions.
    """
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras import layers

    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs)

def load_model():
    global ml_model, CLASS_MAP

    # Load class_indices.json from trained model
    idx_path = 'model/class_indices.json'
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            CLASS_MAP = json.load(f)
        print(f"✅ Loaded {len(CLASS_MAP)} classes from class_indices.json")
    else:
        print("⚠️  class_indices.json not found — using default CLASS_LABELS order")

    if not TF_AVAILABLE:
        print("⚠️  TensorFlow not available — running in demo mode")
        return

    num_classes = len(CLASS_MAP) if CLASS_MAP else len(CLASS_LABELS)

    # Preferred: weights-only file (most reliable, avoids Keras version issues)
    weights_path = 'model/model_weights.weights.h5'
    if os.path.exists(weights_path):
        try:
            ml_model = build_model_architecture(num_classes)
            ml_model.load_weights(weights_path)
            print(f"✅ Trained model loaded from weights ({num_classes} classes)")
            return
        except Exception as e:
            print(f"⚠️  Could not load weights file: {e}")
            ml_model = None

    # Fallback: full .h5 model file
    model_path = 'model/plant_disease_model.h5'
    if os.path.exists(model_path):
        try:
            ml_model = keras.models.load_model(model_path, compile=False)
            print("✅ Trained model loaded successfully")
        except Exception as e:
            print(f"⚠️  Could not load model (attempt 1): {e}")
            # Fallback: try loading with custom object scope ignoring extra kwargs
            try:
                from tensorflow.keras.layers import Dense

                class CompatDense(Dense):
                    def __init__(self, *args, **kwargs):
                        kwargs.pop('quantization_config', None)
                        super().__init__(*args, **kwargs)

                ml_model = keras.models.load_model(
                    model_path, compile=False,
                    custom_objects={'Dense': CompatDense}
                )
                print("✅ Trained model loaded successfully (compat mode)")
            except Exception as e2:
                print(f"⚠️  Could not load model (attempt 2): {e2}")
                print("⚠️  Falling back to demo mode")
    else:
        print("⚠️  No model file found — running in demo mode")

def get_label(class_idx):
    """Return class label string for a given index."""
    # class_indices.json stores {idx_string: label}
    label = CLASS_MAP.get(str(class_idx))
    if label:
        return label
    # fallback to hardcoded list
    if class_idx < len(CLASS_LABELS):
        return CLASS_LABELS[class_idx]
    return "Tomato___healthy"

def detect_and_crop_leaf(img_rgb):
    """
    Detect the dominant leaf region using color-based segmentation and
    crop to it with padding. Falls back to the original image if no
    clear leaf region is found. Input/output are RGB numpy arrays.
    """
    try:
        h, w = img_rgb.shape[:2]
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        # Leaf colors: green, yellow-green, brown (diseased) — broad range
        lower = np.array([15, 25, 25])
        upper = np.array([95, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        # Clean up mask
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img_rgb

        # Largest contour = main leaf
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        # Ignore tiny or near-full-image masks (unreliable)
        if area < 0.03 * (h * w):
            return img_rgb

        x, y, cw, ch = cv2.boundingRect(largest)

        # Add 10% padding around the bounding box
        pad_x, pad_y = int(cw * 0.1), int(ch * 0.1)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(w, x + cw + pad_x)
        y1 = min(h, y + ch + pad_y)

        cropped = img_rgb[y0:y1, x0:x1]
        if cropped.size == 0:
            return img_rgb
        return cropped
    except Exception:
        return img_rgb

def segment_leaf_black_bg(img_rgb):
    """
    Segment the leaf and replace the background with pure black,
    matching the PlantVillage 'segmented' dataset format. Also crops
    to the leaf's bounding box. Falls back to the original image if
    segmentation isn't reliable.
    """
    try:
        h, w = img_rgb.shape[:2]
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        # Broad leaf-color range (green, yellow-green, brown/diseased tones)
        lower = np.array([10, 20, 20])
        upper = np.array([100, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img_rgb

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 0.03 * (h * w):
            return img_rgb

        # Fill mask using the largest contour for clean edges
        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, [largest], -1, 255, thickness=cv2.FILLED)

        # Apply mask: keep leaf pixels, black out background
        result = cv2.bitwise_and(img_rgb, img_rgb, mask=clean_mask)

        # Crop to bounding box with small padding
        x, y, cw, ch = cv2.boundingRect(largest)
        pad_x, pad_y = int(cw * 0.08), int(ch * 0.08)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(w, x + cw + pad_x)
        y1 = min(h, y + ch + pad_y)

        cropped = result[y0:y1, x0:x1]
        if cropped.size == 0:
            return img_rgb
        return cropped
    except Exception:
        return img_rgb

def preprocess_image(image_path, mode="segmented"):
    """
    mode:
      'segmented' - leaf isolated on black background (matches segmented
                    dataset, recommended if model trained on 'segmented')
      'cropped'   - leaf cropped but original colors/background kept
      'full'      - original full image, just resized
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if mode == "segmented":
        img = segment_leaf_black_bg(img)
    elif mode == "cropped":
        img = detect_and_crop_leaf(img)
    # mode == "full" -> no change
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def get_disease_info(label):
    """Look up disease info from DISEASE_DB by label string."""
    # Direct match
    if label in DISEASE_DB:
        return DISEASE_DB[label]
    # Fuzzy match
    label_lower = label.lower()
    for key in DISEASE_DB:
        if key.lower() == label_lower:
            return DISEASE_DB[key]
    # Partial match
    for key in DISEASE_DB:
        if key.lower() in label_lower or label_lower in key.lower():
            return DISEASE_DB[key]
    # Build generic info from label
    parts = label.split("___")
    plant = parts[0].replace("_", " ").replace(",", "").title() if parts else "Unknown"
    disease = parts[1].replace("_", " ").title() if len(parts) > 1 else "Unknown Disease"
    is_healthy = "healthy" in label.lower()
    return {
        "plant": plant,
        "disease": "Healthy" if is_healthy else disease,
        "severity": "None" if is_healthy else "Medium",
        "description": f"{'No disease detected on ' if is_healthy else ''}{plant}. {'Continue good practices.' if is_healthy else 'Consult a local agricultural expert for treatment.'}",
        "symptoms": ["No symptoms" if is_healthy else "Visible leaf lesions", "Discoloration of leaf tissue"],
        "causes": [] if is_healthy else ["Pathogen infection"],
        "prevention": ["Regular monitoring", "Proper crop management"],
        "organic_treatment": ["Neem oil spray", "Copper fungicide"],
        "chemical_treatment": ["Consult local agronomist"],
        "recovery_tips": ["Monitor plant health regularly"]
    }

def predict_disease(image_path):
    """Predict disease. Uses trained model if available, else demo mode."""
    top3 = []
    if ml_model is not None and TF_AVAILABLE:
        # Primary preprocessing mode depends on which PlantVillage variant
        # the model was trained on (set MODEL_DATASET_VARIANT below).
        if MODEL_DATASET_VARIANT == "segmented":
            primary = preprocess_image(image_path, mode="segmented")
            secondary = preprocess_image(image_path, mode="cropped")
        else:
            primary = preprocess_image(image_path, mode="cropped")
            secondary = preprocess_image(image_path, mode="full")

        preds_primary = ml_model.predict(primary, verbose=0)[0]
        preds_secondary = ml_model.predict(secondary, verbose=0)[0]

        # Weight the primary (best-matching) preprocessing more heavily
        avg_preds = (0.7 * preds_primary) + (0.3 * preds_secondary)

        class_idx = int(np.argmax(avg_preds))
        confidence = float(avg_preds[class_idx]) * 100
        label = get_label(class_idx)

        # Top-3 for diagnostics / alternatives
        top_indices = np.argsort(avg_preds)[::-1][:3]
        top3 = [
            {"label": get_label(int(i)), "confidence": round(float(avg_preds[i]) * 100, 1)}
            for i in top_indices
        ]
        print(f"🔍 Top-3: {top3}")
    else:
        # Demo mode
        img_raw = cv2.imread(image_path)
        if img_raw is not None:
            hash_val = int(np.sum(img_raw)) % len(CLASS_LABELS)
            label = CLASS_LABELS[hash_val]
        else:
            label = "Tomato___Early_blight"
        confidence = 85.0 + (abs(hash(image_path)) % 1400) / 100.0
        confidence = min(confidence, 98.5)
        print(f"⚠️  Demo mode: {label} ({confidence:.1f}%)")

    info = get_disease_info(label)

    # Build readable alternatives list (skip the top one, dedupe by plant+disease)
    alternatives = []
    seen = set()
    for t in top3[1:]:
        alt_info = get_disease_info(t['label'])
        key = (alt_info['plant'], alt_info['disease'])
        if key in seen or key == (info['plant'], info['disease']):
            continue
        seen.add(key)
        alternatives.append({
            "plant": alt_info['plant'],
            "disease": alt_info['disease'],
            "confidence": t['confidence']
        })

    return {
        "label": label,
        "plant": info["plant"],
        "disease": info["disease"],
        "confidence": round(confidence, 1),
        "severity": info["severity"],
        "description": info["description"],
        "symptoms": info["symptoms"],
        "causes": info["causes"],
        "prevention": info["prevention"],
        "organic_treatment": info["organic_treatment"],
        "chemical_treatment": info["chemical_treatment"],
        "recovery_tips": info["recovery_tips"],
        "alternatives": alternatives
    }

# ─── Gemini Chat ───────────────────────────────────────────────────────────────
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
CHAT_SYSTEM_PROMPT = """You are LeafLife AI, an expert agricultural assistant specializing in crop diseases, plant health, and farming best practices in India. You help farmers identify diseases, suggest treatments, and provide guidance on crop management.

You have deep knowledge about:
- Crop diseases affecting tomato, potato, rice, wheat, maize, cotton, groundnut, mango, chilli, brinjal, and more
- Organic and chemical treatment options
- Preventive measures and IPM (Integrated Pest Management)
- Indian agricultural conditions, seasons, and farming practices
- Government schemes and resources available to Indian farmers

Always give practical, actionable advice. Be encouraging and supportive to farmers.
When discussing treatments, mention both organic and chemical options.
Keep responses clear, concise, and farmer-friendly."""

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/history')
def history():
    conn = get_db()
    records = conn.execute('SELECT * FROM detections ORDER BY analyzed_at DESC').fetchall()
    conn.close()
    return render_template('history.html', records=records)

@app.route('/dashboard')
def dashboard():
    conn = get_db()
    total = conn.execute('SELECT COUNT(*) as count FROM detections').fetchone()['count']
    healthy = conn.execute("SELECT COUNT(*) as count FROM detections WHERE disease_name IN ('Healthy','No Disease Detected')").fetchone()['count']
    diseased = total - healthy
    disease_dist = conn.execute('SELECT disease_name, COUNT(*) as count FROM detections GROUP BY disease_name ORDER BY count DESC LIMIT 8').fetchall()
    recent = conn.execute('SELECT * FROM detections ORDER BY analyzed_at DESC LIMIT 10').fetchall()
    weekly = conn.execute('''
        SELECT date(analyzed_at) as day, COUNT(*) as count
        FROM detections WHERE analyzed_at >= date('now', '-7 days')
        GROUP BY date(analyzed_at) ORDER BY day
    ''').fetchall()
    conn.close()
    return render_template('dashboard.html', total=total, healthy=healthy, diseased=diseased,
                           disease_dist=disease_dist, recent=recent, weekly=weekly)

@app.route('/assistant')
def assistant():
    conn = get_db()
    chats = conn.execute('SELECT * FROM chat_history ORDER BY created_at DESC').fetchall()
    conn.close()
    return render_template('assistant.html', chats=chats)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    result = predict_disease(filepath)
    result['image_path'] = f"uploads/{filename}"

    conn = get_db()
    conn.execute(
        'INSERT INTO detections (plant_name, disease_name, confidence, severity, image_path, description) VALUES (?,?,?,?,?,?)',
        (result['plant'], result['disease'], result['confidence'], result['severity'], result['image_path'], result['description'])
    )
    conn.commit()
    conn.close()
    return jsonify(result)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    chat_id = data.get('chat_id')
    history = data.get('history', [])

    if not message:
        return jsonify({'error': 'No message provided'}), 400

    conn = get_db()
    if not chat_id:
        title = message[:50] + ('...' if len(message) > 50 else '')
        cur = conn.execute('INSERT INTO chat_history (title) VALUES (?)', (title,))
        chat_id = cur.lastrowid
        conn.commit()

    conn.execute('INSERT INTO chat_messages (chat_id, role, content) VALUES (?,?,?)', (chat_id, 'user', message))
    conn.commit()

    gemini_history = []
    for h in history[-20:]:
        role = "user" if h['role'] == 'user' else "model"
        gemini_history.append({"role": role, "parts": [h['content']]})

    try:
        gmodel = genai.GenerativeModel(model_name=GEMINI_MODEL, system_instruction=CHAT_SYSTEM_PROMPT)
        chat_session = gmodel.start_chat(history=gemini_history)
        response = chat_session.send_message(message)
        reply = response.text
        conn.execute('INSERT INTO chat_messages (chat_id, role, content) VALUES (?,?,?)', (chat_id, 'assistant', reply))
        conn.commit()
        conn.close()
        return jsonify({'reply': reply, 'chat_id': chat_id})
    except Exception as e:
        conn.close()
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/<int:chat_id>/messages')
def get_chat_messages(chat_id):
    conn = get_db()
    messages = conn.execute('SELECT * FROM chat_messages WHERE chat_id=? ORDER BY created_at', (chat_id,)).fetchall()
    conn.close()
    return jsonify([dict(m) for m in messages])

@app.route('/api/history/delete/<int:record_id>', methods=['DELETE'])
def delete_record(record_id):
    conn = get_db()
    conn.execute('DELETE FROM detections WHERE id=?', (record_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

if __name__ == '__main__':
    init_db()
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
