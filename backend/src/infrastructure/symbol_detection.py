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

from src.logger import setup_logger

logger = setup_logger(__name__)
load_dotenv()

DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
SYMBOL_OCR_PROMPT = """
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
- Valid outputs ONLY:
    hex-N
    NUMBER/SHEET
    Unknown

    Examples:
    hex-1
    3/S-3.2
    4/S-4.0
    Unknown
"""


class SymbolData(BaseModel):
    shape: str
    text_content: str
    bbox: List[int]


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
    logger.info(f"  > Running Symbol Detection on {os.path.basename(image_path)}...")
    with Image.open(image_path) as img:
        image = img.convert("RGB")

    # 1. DINO Detection
    text_prompt = """
    hexagon. circle. triangle.
    detail reference bubble.
    section reference bubble.
    drawing callout bubble.

    """

    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.19,
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
                                "text": SYMBOL_OCR_PROMPT
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

            symbol_data = SymbolData(
                shape=str(label),
                text_content=content_text,
                bbox=[x1, y1, x2, y2]
            )

            detected_symbols.append(symbol_data.model_dump())

        except Exception as e:
            logger.error(f"    ! Groq Error on symbol {i}: {e}")

    logger.info(f"  > Found {len(detected_symbols)} symbols.")

    return detected_symbols
