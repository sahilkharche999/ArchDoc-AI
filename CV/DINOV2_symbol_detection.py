from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image, ImageDraw
import torch
import os
from groq  import Groq
from PIL import Image
from dotenv import load_dotenv
import base64
from io import BytesIO
import json
from pydantic import BaseModel
from typing import Optional
load_dotenv()
# 1. Load Model
model_id = "IDEA-Research/grounding-dino-base"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))



class SymbolOutput(BaseModel):
    shape: str
    count: int
    reference: Optional[str] = None


def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def clean_llm_json(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```", "")

    return text.strip()

def extracted_the_symbol_meaning(image_paths: list):
    result = []

    for image_path in image_paths:
        base64_image = image_to_base64(image_path)

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """
You are analyzing a civil engineering floor plan symbol.

Return ONLY valid JSON in this format:

{
  "shape": "<string>",
  "count": <integer>,
  "reference": "<string or null>"
}
"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0
        )

        raw_output = chat_completion.choices[0].message.content

        try:

            cleaned = clean_llm_json(raw_output)
            parsed = SymbolOutput.model_validate_json(cleaned)
            result.append(parsed)
        except Exception as e:
            print("Invalid LLM output:", raw_output)
            print("Error:", e)

    return result


def crop_and_save(image_path: str, box, index, output_dir="crops"):
    os.makedirs(output_dir, exist_ok=True)

    image = Image.open(image_path).convert("RGB")

    x1, y1, x2, y2 = map(int, box.tolist())

    padding = 10
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.width, x2 + padding)
    y2 = min(image.height, y2 + padding)

    cropped = image.crop((x1, y1, x2, y2))

    crop_path = os.path.join(output_dir, f"crop_{index}.png")
    cropped.save(crop_path)

    return crop_path

def dinoV2_symbol_detection(image_path: str,output_dir="symbol_crops"):

    file_name = os.path.basename(image_path)
    image = Image.open(image_path).convert("RGB")

    text = "hexagon. circle. "

    inputs = processor(images=image, text=text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.19,
        text_threshold=0.10,
        target_sizes=[image.size[::-1]]
    )[0]

    draw = ImageDraw.Draw(image)

    saved_paths = []


    for idx, (score, label, box) in enumerate(
        zip(results["scores"], results["labels"], results["boxes"])
    ):

        print(f"Detected: {label} | Confidence: {score.item():.4f}")

        if score.item() > 0.25:

            # Crop and save
            crop_path = crop_and_save(
                image_path=image_path,
                box=box,
                index=idx,
                output_dir=output_dir
            )
            saved_paths.append(crop_path)


            # Draw rectangle for debug
            x1, y1, x2, y2 = map(int, box.tolist())
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1), f"{label} {score.item():.2f}", fill="red")
    identifed_things=extracted_the_symbol_meaning(saved_paths)
    image.save(f"{file_name}_detection.jpg")
    print(identifed_things)
    return saved_paths

if __name__=="__main__":
    image_path='output_temp/floor_4/floor_4/vlm/images/cb7ea89114e1c238311cf9bf3f1babcc1ef68eec3373691da3efe37289b125fe.jpg'
    output_dir='symbol_crops'
    dinoV2_symbol_detection(image_path,output_dir)
