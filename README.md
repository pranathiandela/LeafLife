# 🌿 LeafLife — Smart Crop Disease Identification System

A full-stack AI-powered web application for detecting crop leaf diseases using deep learning (TensorFlow/Keras + OpenCV), built with Flask, SQLite, and an AI chat assistant powered by Claude.

---

## 🚀 Quick Start

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your Anthropic API key

```bash
# Linux/macOS
export ANTHROPIC_API_KEY=your_api_key_here

# Windows CMD
set ANTHROPIC_API_KEY=your_api_key_here

# Windows PowerShell
$env:ANTHROPIC_API_KEY="your_api_key_here"
```

Get your API key from: https://console.anthropic.com

### 3. Run the application

```bash
python app.py
```

Then open: http://localhost:5000

---

## 🤖 AI Model Setup (Optional for Training)

The app runs in **demo mode** without a trained model, but for real predictions:

### Download PlantVillage Dataset

1. Go to: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
2. Download and extract to: `model/dataset/PlantVillage/`

The directory should look like:
```
model/dataset/PlantVillage/
├── Tomato___Early_blight/
├── Tomato___Late_blight/
├── Potato___Early_blight/
├── ... (38 class folders)
```

### Train the Model

```bash
python model/train_model.py
```

Training uses MobileNetV2 transfer learning with two phases:
- **Phase 1** (15 epochs): Train classification head only
- **Phase 2** (30 epochs): Fine-tune top layers

Expected accuracy: **95%+** on validation set.

### Test a Single Prediction

```bash
python model/train_model.py --predict path/to/leaf_image.jpg
```

---

## 📁 Project Structure

```
leaflife/
├── app.py                    # Flask application & API routes
├── requirements.txt
├── README.md
├── database/
│   └── leaflife.db           # SQLite database (auto-created)
├── model/
│   ├── train_model.py        # Training script
│   ├── plant_disease_model.h5  # Trained model (after training)
│   └── class_indices.json    # Class mapping (auto-generated)
├── static/
│   ├── css/style.css         # Main stylesheet
│   ├── js/
│   │   ├── main.js           # Navbar, language selector
│   │   ├── detection.js      # Upload & analysis UI
│   │   ├── history.js        # History filters & pagination
│   │   └── assistant.js      # AI chat interface
│   └── uploads/              # Uploaded leaf images (auto-created)
└── templates/
    ├── base.html             # Base template with navbar
    ├── home.html             # Landing page
    ├── detection.html        # Disease detection page
    ├── history.html          # Detection history
    ├── dashboard.html        # Analytics dashboard
    └── assistant.html        # AI chat assistant
```

---

## 🌾 Supported Crops & Diseases

| Crop       | Diseases Covered                                      |
|------------|-------------------------------------------------------|
| Tomato     | Early Blight, Late Blight, Leaf Mold, Septoria, Bacterial Spot, Yellow Curl Virus, Mosaic Virus, Spider Mites, Target Spot, Bacterial Spot |
| Potato     | Early Blight, Late Blight                             |
| Maize      | Common Rust, Northern Leaf Blight, Gray Leaf Spot     |
| Rice       | Brown Spot, Leaf Scald                                |
| Wheat      | Yellow Rust (Stripe Rust)                             |
| Mango      | Anthracnose                                           |
| Chilli     | Leaf Curl Virus                                       |
| Brinjal    | Phomopsis Blight                                      |
| Groundnut  | Early Leaf Spot                                       |
| Cotton     | Bacterial Blight                                      |
| Grape      | Black Rot, Esca, Leaf Blight                         |
| Apple      | Apple Scab, Black Rot, Cedar Apple Rust               |
| Peach      | Bacterial Spot                                        |
| Strawberry | Leaf Scorch                                           |
| Pepper     | Bacterial Spot                                        |

---

## 🖥️ Pages

| Page         | URL          | Description                                   |
|--------------|--------------|-----------------------------------------------|
| Home         | `/`          | Landing page with features overview           |
| Detection    | `/detection` | Upload & analyze leaf images                  |
| Dashboard    | `/dashboard` | Analytics charts and recent detections        |
| History      | `/history`   | All past detections with search & filter      |
| AI Assistant | `/assistant` | Chat with Claude AI about crop diseases       |

---

## 🌐 Language Support

The navbar includes an Indian language selector supporting:
- English, हिंदी (Hindi), తెలుగు (Telugu), தமிழ் (Tamil)
- ಕನ್ನಡ (Kannada), मराठी (Marathi), ગુજરાતી (Gujarati)
- ਪੰਜਾਬੀ (Punjabi), বাংলা (Bengali), മലയാളം (Malayalam), ଓଡ଼ିଆ (Odia)

---

## 🔧 API Endpoints

| Method | Endpoint                      | Description                  |
|--------|-------------------------------|------------------------------|
| POST   | `/api/analyze`                | Analyze uploaded leaf image  |
| POST   | `/api/chat`                   | Send message to AI assistant |
| GET    | `/api/chat/<id>/messages`     | Get chat history             |
| DELETE | `/api/history/delete/<id>`    | Delete a detection record    |

---

## ⚙️ Technology Stack

- **Backend**: Python 3.10+, Flask 3.0
- **ML/AI**: TensorFlow 2.16, Keras, MobileNetV2, OpenCV
- **Database**: SQLite (via sqlite3)
- **AI Chat**: Anthropic Claude API
- **Frontend**: HTML5, CSS3 (custom design system), Vanilla JS
- **Charts**: Chart.js 4
- **Icons**: Font Awesome 6
- **Fonts**: Plus Jakarta Sans, Syne

---

## 📝 Notes

- The app runs in **demo mode** if no trained model is found (`model/plant_disease_model.h5`).
- In demo mode, predictions are simulated for demonstration purposes.
- Place your trained `.h5` model file in the `model/` directory for real predictions.
- The AI Assistant requires a valid `ANTHROPIC_API_KEY` environment variable.
