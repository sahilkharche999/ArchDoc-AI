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

def _boxes_overlap(box1: List[int], box2: List[int], threshold: float = 0.5) -> bool:
    """Check if two bboxes overlap significantly (IoU > threshold)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return False

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return (intersection / union) > threshold

def clean_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r'(?<=/)[5s](?=\d)', 'S', text, flags=re.IGNORECASE)
    # Valid patterns
    
    if re.match(r"^cir-\d+$", text, re.IGNORECASE):
        return text
    
    if re.match(r"^hex-\d+$", text, re.IGNORECASE):
        return text
    
    if re.match(r"^[^/\s]+/[^/\s]+$", text):
        return text.upper() 

    return "Unknown"


class SymbolData(BaseModel):
    shape: str
    text_content: str
    bbox: List[int]


def detect_and_read_symbols(image_path: str, output_dir: str,llm_flash) -> List[Dict]:
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

    text_prompt = "circle. hexagon. triangle."

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
        threshold=0.15,
        text_threshold=0.8,
        target_sizes=[image.size[::-1]]
    )[0]

    scored=sorted(
        zip(results["scores"],results["labels"], results["boxes"]),
        key=lambda x:x[0].item(),
        reverse=True
    )
    kept_boxes=[]
    for score,label,box in scored:
        coords=list(map(int,box.tolist()))
        if any(_boxes_overlap(coords,kept[2],threshold=0.4)for kept in kept_boxes):
            logger.debug(f"Pre-OCR dedup: dropped score={score:.2f} bbox={coords}")
            continue
        kept_boxes.append((score.item(), label, coords))
    logger.info(f"Pre-OCR dedup | raw={len(scored)} → kept={len(kept_boxes)}")


    detected_symbols = []
    os.makedirs(output_dir, exist_ok=True)

    # 2. Process Detections
    for i, (score, label, coords) in enumerate(kept_boxes):
        if score < 0.15: continue
        logger.debug(f"Detection accepted | index={i} | label={label} | score={score:.2f}")

        # Get Coords
        x1, y1, x2, y2 = coords
        img_area = image.width * image.height
        bbox_area = (x2 - x1) * (y2 - y1)
        if bbox_area > 0.5 * img_area:
            logger.debug(f"Skipping oversized detection | index={i} | bbox_area_ratio={bbox_area/img_area:.2f}")
            continue

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

            lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
            if len(lines) <= 1:
                # Single symbol — original behavior
                content_text = clean_output(raw_text)
                logger.debug(f"OCR success | index={i} | raw={raw_text} | cleaned={content_text}")
                detected_symbols.append(SymbolData(
                    shape=str(label),
                    text_content=content_text,
                    bbox=[x1, y1, x2, y2]
                ).model_dump())
            else:
                # Multiple stacked symbols — iterate one by one
                logger.info(f"OCR returned {len(lines)} stacked symbols | index={i} | lines={lines}")
                height = y2 - y1
                slice_h = height / len(lines)
                for idx, line in enumerate(lines):
                    cleaned = clean_output(line)
                    if cleaned == "Unknown":
                        logger.debug(f"  → skipped line {idx+1}/{len(lines)}: {line!r} (invalid)")
                        continue
                    sy1 = int(y1 + idx * slice_h)
                    sy2 = int(y1 + (idx + 1) * slice_h)
                    detected_symbols.append(SymbolData(
                        shape=str(label),
                        text_content=cleaned,
                        bbox=[x1, sy1, x2, sy2]
                    ).model_dump())
                    logger.debug(f"  → split symbol {idx+1}/{len(lines)}: {cleaned} | bbox=({x1},{sy1},{x2},{sy2})")

        except Exception as e:
            logger.error(
                f"OCR failed | image={os.path.basename(image_path)} | index={i} | label={label} | bbox={crop_box} | error={str(e)}"
            )
            raise

    logger.info(
        f"Symbol detection completed | image={os.path.basename(image_path)} | count={len(detected_symbols)}"
    )
    unique_symbols = []
    for sym in detected_symbols:
        if sym["text_content"] == "Unknown":
            continue
        is_duplicate = False
        for kept in unique_symbols:
            if (sym["text_content"] == kept["text_content"]  and _boxes_overlap(sym["bbox"], kept["bbox"], threshold=0.5)):
                is_duplicate = True
                logger.debug(
                    f"Dedup: skipping {sym['text_content']} at {sym['bbox']} "
                    f"— overlaps with {kept['text_content']} at {kept['bbox']}"
                )
                break
        if not is_duplicate:
            unique_symbols.append(sym)

    logger.info(
        f"After dedup | unique={len(unique_symbols)} "
        f"(removed {len(detected_symbols) - len(unique_symbols)} spatial duplicates)"
    )

    return unique_symbols


