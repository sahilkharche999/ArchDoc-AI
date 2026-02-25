import cv2
from ultralytics import YOLO

# -------------------------------
# Load model
# -------------------------------
model = YOLO("CV/yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt")

CLASS_NAMES = model.names  # id → label mapping

# -------------------------------
def detect_and_draw_boxes(img_path: str, output_path: str):
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Failed to load image: {img_path}")

    # YOLO inference
    results = model.predict(
        source=img_path,
        imgsz=1024,
        conf=0.25,
        device="cpu"
    )

    result = results[0]
    boxes = result.boxes

    # -------------------------------
    # Draw rectangles only
    # -------------------------------
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = CLASS_NAMES[cls_id]

        # draw rectangle
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # label
        cv2.putText(
            image,
            f"{label} {conf:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

        # YOLO-like console output
        print(
            f"{label}: "
            f"({x1}, {y1}, {x2}, {y2}), "
            f"conf={conf:.2f}"
        )

    cv2.imwrite(output_path, image)

# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    detect_and_draw_boxes(
        "CV/floor_1.png",
        "yolov8_doclaynet_result.jpg"
    )
