# utils/symbol_detection.py
import base64
import os
from typing import List, Dict

import torch
from PIL import Image
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

load_dotenv()

# --- CONFIG ---
# Load models once to save time
DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class SymbolData(BaseModel):
    shape: str
    text_content: str
    bbox: List[int]  # [x1, y1, x2, y2]


def image_to_base64(pil_image):
    from io import BytesIO
    buff = BytesIO()
    pil_image.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")


def detect_and_read_symbols(image_path: str, output_dir: str) -> List[Dict]:
    """
    1. Uses Grounding DINO to find symbols (Hexagons, Circles).
    2. Crops them.
    3. Uses Groq (Llama Vision) to read the text inside.
    """
    print(f"  > Running Symbol Detection on {os.path.basename(image_path)}...")

    image = Image.open(image_path).convert("RGB")

    # 1. DINO Detection
    text_prompt = "hexagon. circle. triangle."  # Add shapes relevant to your plans
    inputs = processor(images=image, text=text_prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.19,  # Adjust based on sensitivity needs
        text_threshold=0.10,
        target_sizes=[image.size[::-1]]
    )[0]

    detected_symbols = []
    os.makedirs(output_dir, exist_ok=True)

    # 2. Process Detections
    for i, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
        if score.item() < 0.20: continue

        # Get Coords
        x1, y1, x2, y2 = map(int, box.tolist())

        # Add padding for better OCR
        pad = 10
        crop_box = (
            max(0, x1 - pad), max(0, y1 - pad),
            min(image.width, x2 + pad), min(image.height, y2 + pad)
        )

        # Crop
        crop = image.crop(crop_box)

        # 3. Groq / Llama Vision for Reading
        try:
            b64 = image_to_base64(crop)
            chat_completion = groq_client.chat.completions.create(
                messages=[
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": """
                You are reading a structural drawing symbol.
                
                There are only two valid outputs:
                
                1) If this is a HEXAGON containing a number N:
                   return exactly: hex-N
                
                2) If this is a DETAIL CALLOUT (circle over triangle)
                   containing:
                   - Top: a number (e.g., 3)
                   - Bottom: a sheet reference (e.g., S-3.2)
                
                   return exactly: NUMBER/SHEET
                
                Examples:
                hex-1
                3/S-3.2
                4/S-4.0
                
                Rules:
                - NO spaces
                - NO newline
                - NO explanation
                - NO markdown
                - Output only the final formatted value
                - If unreadable return: Unknown
                """
                                            },
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:image/png;base64,{b64}"
                                                }
                                            }
                                        ],
                                    }
                                ],
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=0
            )
            content_text = chat_completion.choices[0].message.content.strip()

            # Store Result
            symbol_data = {
                "type": label,  # e.g., 'hexagon'
                "content": content_text,  # e.g., '1' or '7/S-3.2'
                "bbox": [x1, y1, x2, y2],
                "confidence": score.item()
            }
            detected_symbols.append(symbol_data)

            # Optional: Save crop for debug
            # crop.save(f"{output_dir}/symbol_{i}_{content_text.replace('/','-')}.png")

        except Exception as e:
            print(f"    ! Groq Error on symbol {i}: {e}")

    print(f"  > Found {len(detected_symbols)} symbols.")
    print(detected_symbols)
    return detected_symbols


if __name__ == "__main__":
    image_path = 'output_temp/floor_3/floor_3/vlm/images/c2071a8eb39ff6495f84a2cb170897bc62a795ef8b60ce9e337bd32f615e99dc.jpg'
    output_dir = 'symbol_crops'
    detect_and_read_symbols(image_path, output_dir)
