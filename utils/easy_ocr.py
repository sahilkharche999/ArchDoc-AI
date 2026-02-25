import easyocr
import re
reader = easyocr.Reader(['en'])
import os
"""
Converts EasyOCR polygon bbox to (x1, y1, x2, y2)
"""
import cv2

from dataclasses import dataclass

@dataclass
class OCRBox:
    text: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


def preprocess_image_inplace(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    processed = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    cv2.imwrite(image_path, processed)
    return True


def image_ocr_with_cords(imgPath: str):
    results = []
    ocr_output = reader.readtext(imgPath)

    for bbox, text, prob in ocr_output:
        xs = [int(p[0]) for p in bbox]
        ys = [int(p[1]) for p in bbox]

        results.append(
            OCRBox(
                text=text,
                confidence=prob,
                x1=min(xs),
                y1=min(ys),
                x2=max(xs),
                y2=max(ys),
            )
        )

    return results




if __name__=='__main__':
    input_page='output_temp/crops/crop_1.png'
    ocr_results=image_ocr_with_cords(input_page)
    print(ocr_results)
    # Load image
    image_path =input_page  # your image path
    img = cv2.imread(image_path)

    # Example input list (like yours)
    data = ocr_results

    for item in ocr_results:
        cv2.rectangle(
            img,
            (item.x1, item.y1),
            (item.x2, item.y2),
            (0, 255, 0),
            2
        )
    # Save output
    cv2.imwrite("output.jpg", img)

    # Show image (optional)
    cv2.imshow("Bounding Boxes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




