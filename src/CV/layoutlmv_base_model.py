from transformers import LayoutLMv3Model, LayoutLMv3Processor
from PIL import Image, ImageDraw
import torch

# Load a FINE-TUNED model (trained to detect layout elements)
model_id = "microsoft/layoutlmv3-base-finetuned-publaynet"
processor = LayoutLMv3Processor.from_pretrained(model_id, apply_ocr=True)
model = LayoutLMv3ForObjectDetection.from_pretrained(model_id)

# Load Image
image = Image.open("CV/floor_1.png").convert("RGB")

# Process
inputs = processor(images=image, return_tensors="pt")

# Predict
with torch.no_grad():
    outputs = model(**inputs)

# Convert outputs (bounding boxes and classes) to image coordinates
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

# Draw the results
draw = ImageDraw.Draw(image)
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(f"Detected {model.config.id2label[label.item()]} with confidence {round(score.item(), 3)} at location {box}")
    
    # Draw rectangle
    draw.rectangle(box, outline="red", width=3)
    draw.text((box[0], box[1]), model.config.id2label[label.item()], fill="red")

# Show image
image.show()