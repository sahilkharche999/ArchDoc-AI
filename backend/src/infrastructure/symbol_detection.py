# utils/symbol_detection.py
import base64
import io
import os
import re
from typing import List, Dict

import torch
from PIL import Image
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from src.logger import setup_logger
from src.workflow.workflows.estimation.prompt import SYMBOL_OCR_PROMPT

logger = setup_logger(__name__)
load_dotenv()
llm_flash = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_processor = None
_model = None


def _load_model():
    global _processor, _model
    if _processor is None:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        logger.info("Loading Grounding DINO model...")
        _processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
        _model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID).to(DEVICE)
        logger.debug("Grounding DINO model loaded.")


def clean_output(text: str) -> str:
    text = text.strip()

    # Valid patterns
    if re.match(r"^hex-\d+$", text):
        return text

    if re.match(r"^\d+/S-\d+(\.\d+)?$", text):
        return text

    return "Unknown"


class SymbolData(BaseModel):
    shape: str
    text_content: str
    bbox: List[int]


def detect_and_read_symbols(image_path: str, output_dir: str) -> List[Dict]:
    """
    1. Uses Grounding DINO to find symbols (Hexagons, Circles).
    2. Crops them.
    3. Uses Gemini to read the text inside.
    """
    _load_model()
    logger.debug(
        f"Symbol detection started | image={os.path.basename(image_path)} | output_dir={output_dir}"
    )
    try:
        with Image.open(image_path) as img:
            image = img.convert("RGB")
    except Exception as e:
        logger.error(f"Image load failed | image={image_path} | error={str(e)}")
        raise

    # 1. DINO Detection
    text_prompt = """
    hexagon. circle. triangle.
    detail reference bubble.
    section reference bubble.
    drawing callout bubble.

    """

    inputs = _processor(images=image, text=text_prompt, return_tensors="pt").to(DEVICE)

    try:
        logger.debug("Running DINO inference")
        with torch.no_grad():
            outputs = _model(**inputs)
        logger.debug("DINO inference completed")
    except Exception as e:
        logger.error(f"DINO inference failed | image={image_path} | error={str(e)}")
        raise

    results = _processor.post_process_grounded_object_detection(
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
        logger.debug(
            f"Detection accepted | index={i} | label={label} | score={score.item():.2f}"
        )

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
        buffer = io.BytesIO()
        crop.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode()

        # 3. Gemini Vision for Reading
        try:
            logger.debug(
                f"OCR request | index={i} | label={label} | bbox={crop_box}"
            )

            msg = HumanMessage(content=[
                {"type": "text", "text": SYMBOL_OCR_PROMPT()},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }

            ])

            response = llm_flash.invoke([msg])

            if isinstance(response.content, str):
                raw_text = response.content.strip()
            else:
                raw_text = response.content[0]["text"].strip()
            content_text = clean_output(raw_text)

            logger.debug(
                f"OCR success | index={i} | raw={raw_text} | cleaned={content_text}"
            )

            symbol_data = SymbolData(
                shape=str(label),
                text_content=content_text,
                bbox=[x1, y1, x2, y2]
            )

            detected_symbols.append(symbol_data.model_dump())

        except Exception as e:
            logger.error(
                f"OCR failed | image={os.path.basename(image_path)} | index={i} | label={label} | bbox={crop_box} | error={str(e)}"
            )
            raise

    logger.info(
        f"Symbol detection completed | image={os.path.basename(image_path)} | count={len(detected_symbols)}"
    )

    return detected_symbols
