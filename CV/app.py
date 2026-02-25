import os
import cv2
from ultralytics import YOLO

MODEL_PATH = "CV/best.pt"

INPUT_DIR = "CV"

OUTPUT_DIR = "CV"

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Model not found at {MODEL_PATH}")
else:
    print(f"✅ Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # Get list of images
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if not image_files:
        print(f"⚠️ No images found in {INPUT_DIR}")
    else:
        print(f"🔍 Found {len(image_files)} images. Processing...")

        for img_name in image_files:
            img_path = os.path.join(INPUT_DIR, img_name)

            # --- PREDICT ---
            # conf=0.25: Only show boxes with >25% confidence
            # save=False: We will handle saving manually to control the path
            results = model.predict(img_path, conf=0.45)

            print(results[0])

            # --- DRAW BOXES ---
            # .plot() creates the numpy array of the image with boxes drawn
            annotated_img = results[0].plot()

            # --- SAVE RESULT ---
            save_path = os.path.join(OUTPUT_DIR, f"pred_{img_name}")
            cv2.imwrite(save_path, annotated_img)

            # --- DISPLAY (Optional) ---
            print(f"\nSaved prediction to: {save_path}")
print("\n🎉 All done! Check the '/content/output_predictions' folder.")