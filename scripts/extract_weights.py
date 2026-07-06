import os
from tensorflow import keras
from tensorflow.keras.layers import Dense

MODEL_PATH = 'model/plant_disease_model.h5'
WEIGHTS_OUT = 'model/model_weights.weights.h5'

if not os.path.exists(MODEL_PATH):
    print(f"❌ {MODEL_PATH} not found. Place your trained model there first.")
    raise SystemExit(1)

print(f"Loading {MODEL_PATH} (compat mode)...")

class CompatDense(Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

model = keras.models.load_model(
    MODEL_PATH, compile=False,
    custom_objects={'Dense': CompatDense}
)
print("✅ Model loaded")
print(f"   Input shape:  {model.input_shape}")
print(f"   Output shape: {model.output_shape}")
print(f"   Total params: {model.count_params():,}")

model.save_weights(WEIGHTS_OUT)
print(f"✅ Weights saved to {WEIGHTS_OUT}")
print("\nNow run: python app.py")
