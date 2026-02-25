import layoutparser as lp
import cv2
import matplotlib.pyplot as plt

# Load image
image = cv2.imread("CV/floor_2.png")
if image is None:
    raise FileNotFoundError("Image not found at CV/floor_2.png")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load pretrained layout detection model
model = lp.Detectron2LayoutModel(
    config_path="detectron2://configs/PubLayNet/faster_rcnn_R_50_FPN_3x.yaml",
    label_map={
        0: "Text",
        1: "Title",
        2: "List",
        3: "Table",
        4: "Figure",
    },
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
    device="cpu"
)

# Detect layout
layout = model.detect(image_rgb)

# Print detected blocks
for block in layout:
    print(
        f"Type: {block.type}, "
        f"Score: {block.score:.2f}, "
        f"Box: {block.coordinates}"
    )

# Draw boxes
viz_image = lp.draw_box(
    image_rgb,
    layout,
    box_width=3,
    show_element_type=True
)

# Save result
output_path = "result.jpg"
cv2.imwrite(output_path, cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR))
print(f"Saved layout result to: {output_path}")

# Show image
plt.figure(figsize=(10, 10))
plt.imshow(viz_image)
plt.axis("off")
plt.show()
