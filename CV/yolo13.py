from ultralytics import YOLO
import cv2
import json
from pathlib import Path

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "yolov8n.pt"
model = YOLO(MODEL_PATH)
CONF_THRESHOLD = 0.4

OUTPUT_DIR = Path("output")
CROPS_DIR = OUTPUT_DIR / "regions"
OUTPUT_DIR.mkdir(exist_ok=True)
CROPS_DIR.mkdir(exist_ok=True)

CLASS_NAMES = [
    "paragraph_text",
    "key_value",
    "title",
    "subtitle",
    "marginalia",
    "table",
    "bom_table",
    "subtable",
    "empty_table",
    "title_box",
    "diagram",
    "engineering_drawing",
    "flowchart",
    "image"
]

# -------------------------------
# LOAD MODEL
# -------------------------------
model = YOLO(MODEL_PATH)


def refine_layout_class(coarse_class, crop_img):
    """
    Convert YOLO coarse layout → engineering-aware layout
    """
    h, w = crop_img.shape[:2]
    aspect_ratio = w / max(h, 1)

    if coarse_class == "title":
        return "title"

    if coarse_class == "text":
        # Heuristic example
        if aspect_ratio > 5:
            return "key_value"
        return "paragraph_text"

    if coarse_class == "table":
        return "table"  # later refined into BOM / subtable

    if coarse_class == "figure":
        return "engineering_drawing"

    return "image"


# -------------------------------
# MAIN FUNCTION
# -------------------------------
def generate_layout_with_regions(image_path):
    image = cv2.imread(image_path)
    original = image.copy()

    results = model(image, conf=CONF_THRESHOLD)[0]
    layout_map = []

    region_id = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        coarse_class = model.names[cls_id]
        class_name = refine_layout_class(coarse_class, crop)
        

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # -------------------------------
        # SAVE CROPPED REGION
        # -------------------------------
        crop = original[y1:y2, x1:x2]
        crop_filename = f"{region_id:03d}_{class_name}.png"
        cv2.imwrite(str(CROPS_DIR / crop_filename), crop)

        # -------------------------------
        # DRAW ON OVERVIEW IMAGE
        # -------------------------------
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{region_id}: {class_name}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

        # -------------------------------
        # STORE METADATA
        # -------------------------------
        layout_map.append({
            "region_id": region_id,
            "class": class_name,
            "confidence": round(confidence, 3),
            "bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            },
            "crop_path": f"regions/{crop_filename}"
        })

        region_id += 1

    return layout_map, image

# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    layout, annotated_img = generate_layout_with_regions("CV/floor_2.png")

    # Save layout JSON
    with open(OUTPUT_DIR / "layout_map.json", "w") as f:
        json.dump(layout, f, indent=2)

    # Save annotated image
    cv2.imwrite(str(OUTPUT_DIR / "layout_overview.png"), annotated_img)
