import cv2
import numpy as np
import os

# Load image
img = cv2.imread("Floor.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Adaptive threshold: more sensitive to drawing lines
thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY_INV, 15, 10
)

# Dilate to connect small text/lines into blocks
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
dilated = cv2.dilate(thresh, kernel, iterations=2)

# Find contours of connected blocks
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

os.makedirs("segments", exist_ok=True)

count = 0
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    # Keep all reasonable content, filter only very tiny noise
    if w > 50 and h > 50:
        crop = img[y:y+h, x:x+w]

        # Optional: zoom 2x
        zoomed = cv2.resize(crop, (w*2, h*2), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(f"segments/segment_{count}.png", zoomed)
        count += 1

print(f"{count} segments saved.")
