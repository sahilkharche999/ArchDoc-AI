import cv2
from pathlib import Path
from doclayout_yolo import YOLOv10

# Load the pre-trained model
model = YOLOv10("CV/doclayout_yolo_docstructbench_imgsz1024.pt")
# model = YOLOv10.from_pretrained("juliozhao/DocLayout-YOLO-DocStructBench")
class_names = model.names

def  get_coordinates_of_the_segmentation(img_path: str,output_path:str):
    # -------------------------------
    # Prepare output directory
    # -------------------------------
    crops_dir = Path(output_path)
    crops_dir.mkdir(exist_ok=True)

    # -------------------------------
    # Load original image
    # -------------------------------
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Failed to load image: {img_path}")

    original = image.copy()

    # -------------------------------
    # Perform prediction
    # -------------------------------
    det_res = model.predict(
        img_path,
        imgsz=1024,
        conf=0.02,
        device="cpu"
    )

    result = det_res[0]
    boxes = result.boxes

    # -------------------------------
    # Iterate over detections 
    # -------------------------------
    for i, box in enumerate(boxes):
        # Get bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = class_names[cls_id]

        print(
            f"Box {i}: "
            f"({x1}, {y1}, {x2}, {y2}), "
            f"class={class_name}, conf={conf:.2f}"
        )
        
        # -------------------------------
        # Crop region
        # -------------------------------
        crop = original[y1:y2, x1:x2]

        crop_path = crops_dir / f"crop_{i}.png"
        cv2.imwrite(str(crop_path), crop)

    # -------------------------------
    # Save annotated overview image
    # -------------------------------
    annotated_frame = result.plot(pil=False, line_width=3)
    cv2.imwrite("result2.jpg", annotated_frame)

# -------------------------------
# RUN
# ------------------------------- 
if __name__ == "__main__":
    get_coordinates_of_the_segmentation("FLOORPLAN302.png","output_temp/crops")
