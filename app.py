from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

templates = Jinja2Templates(directory="templates")
MODEL_PATH = os.path.join("model", "skin_model_final.h5")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model loading failed:", e)
    model = None

class_names = [
    "Acne and Rosacea",
    "Eczema",
    "Melanoma",
    "Psoriasis"
]

def read_image(file_bytes):
    img = Image.open(BytesIO(file_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

common_treatments = [
    "Daily Cleanser: Use CeraVe Hydrating Cleanser or Cetaphil Gentle Skin Cleanser",
    "Morning Moisturizer: Apply CeraVe AM Facial Moisturizing Lotion with SPF 30",
    "Night Cream: Use Neutrogena Hydro Boost Water Gel or Olay Regenerist Night Cream",
    "Sunscreen (must): Apply Neutrogena Ultra Sheer SPF 50+ every morning",
    "Body Lotion: Use Vaseline Intensive Care or Lubriderm Daily Moisture Lotion",
    "Avoid scratching or irritating the affected area at all times",
    "Drink plenty of water and maintain a healthy diet for skin health",
    "Consult a dermatologist if symptoms worsen or do not improve in 2 weeks"
]

specific_treatments = {
    "Acne and Rosacea": [
        "Creams: Benzac AC (benzoyl peroxide 5%), Epiduo Gel, Clindac A Gel (clindamycin)",
        "OTC options: Clean & Clear Persa-Gel, Neutrogena On-The-Spot Acne cream",
        "Wash face twice daily with CeraVe Acne Foaming Cream Wash (salicylic acid)",
        "Avoid oily and comedogenic skincare products",
        "For severe acne consult a dermatologist for tretinoin (Retin-A) prescription",
        "Use oil-free moisturizer like Cetaphil Oil Control Moisturizer"
    ],
    "Eczema": [
        "Creams: Hydrocortisone 1% cream, Elidel (pimecrolimus), Protopic (tacrolimus)",
        "Moisturizers: CeraVe Moisturizing Cream, Eucerin Eczema Relief, Aveeno Eczema Therapy",
        "For flare-ups: Betamethasone valerate cream or Triamcinolone cream (prescription)",
        "Avoid hot showers and harsh soaps — use Dove Sensitive Skin bar instead",
        "Apply moisturizer immediately after bathing while skin is still damp",
        "Use fragrance-free laundry detergent like Tide Free and Gentle"
    ],
    "Melanoma": [
        "URGENT: See a dermatologist or oncologist immediately — do not delay",
        "Do not apply any home remedy creams on the affected area",
        "Protect area completely from sun — use SPF 50+ sunscreen at all times",
        "Treatment is medical: surgery, immunotherapy, or targeted therapy by doctor",
        "Sunscreens for protection: Neutrogena Ultra Sheer SPF 100, La Roche-Posay Anthelios",
        "Early detection is critical — the sooner treated, the better the outcome"
    ],
    "Psoriasis": [
        "Creams: Dovobet (calcipotriol + betamethasone), Daivobet Gel, Clobetasol propionate",
        "Moisturizers: CeraVe Psoriasis Cream, Eucerin Roughness Relief, Vaseline Intensive Care",
        "Coal tar creams: MG217 Psoriasis Cream, Psoriasin Gel — reduce scaling effectively",
        "Vitamin D creams: Calcipotriene (Dovonex) — slows skin cell overgrowth",
        "Phototherapy (UVB light therapy) recommended for widespread psoriasis",
        "Avoid stress and skin injuries as they can trigger flare-ups"
    ]
}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "home.html")

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    if not file or file.filename == "":
        return templates.TemplateResponse(request, "error.html", {
            "error": "Please upload an image"
        })
    if model is None:
        return templates.TemplateResponse(request, "error.html", {
            "error": "Model not loaded properly"
        })
    try:
        img_bytes = await file.read()
        img_array = read_image(img_bytes)
        preds = model.predict(img_array)
        class_idx = int(np.argmax(preds))
        confidence = float(np.max(preds))
        probability = round(confidence * 100, 2)

        # ✅ Confidence threshold check
        if confidence < 0.70:
            return templates.TemplateResponse(request, "error.html", {
                "error": f"⚠️ This does not appear to be a skin disease image. Model confidence is only {probability}%. Please upload a clear close-up photo of a skin condition."
            })

        prediction = class_names[class_idx]

        return templates.TemplateResponse(request, "result.html", {
            "prediction": prediction,
            "probability": probability,
            "common_treatments": common_treatments,
            "specific_treatments": specific_treatments[prediction]
        })
    except Exception as e:
        return templates.TemplateResponse(request, "error.html", {
            "error": str(e)
        })