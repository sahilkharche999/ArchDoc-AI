import base64
import cv2
import os
import re 
from typing import  List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import pdfplumber
from PIL import Image
from io import BytesIO

load_dotenv()

# --- 1. SETUP MODELS ---
llm_pro = ChatGoogleGenerativeAI(model="gemini-3-pro-preview") 
llm_flash = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite") 

def load_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
def preprocess_image_inplace(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return False

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold
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


def normalize_text(text):
    """Removes punctuation and extra spaces for better matching."""
    if not text: return ""
    return re.sub(r'[^a-zA-Z0-9]', '', text).upper()

def find_title(image_path: str):
    image_base64 = load_image_base64(image_path)
    prompt = """
    Look at this construction sheet.
    Extract ONLY the bold, uppercase TITLES found below the detail drawings.
    Examples: "TYP. WALL SECTION", "LADDER DETAIL", "SECTION A-A".
    Do NOT extract notes, dimensions, or material labels.
    Return just the titles, one per line.
    """
    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
    ])
    response = llm_flash.invoke([message])
    titles = [line.strip() for line in response.content.split("\n") if line.strip()]
    return titles

def find_all_title_coordinates(words, titles):
    results = {} 
    
    # Pre-normalize PDF text for fuzzy matching
    for title in titles:
        norm_title = normalize_text(title)
        if not norm_title: continue
        
        title_words = title.split()
        n = len(title_words)
        
        candidates = []
        
        # Sliding window search
        for i in range(len(words) - n + 1):
            segment_text = "".join([w["text"] for w in words[i:i+n]])
            norm_segment = normalize_text(segment_text)
            
            if norm_title in norm_segment: 
                boxes = words[i:i+n]
                candidates.append({
                    "x1": min(w["x0"] for w in boxes),
                    "y1": min(w["top"] for w in boxes),
                    "x2": max(w["x1"] for w in boxes),
                    "y2": max(w["bottom"] for w in boxes)
                })
        
        if candidates:
            results[title] = candidates
            
    return results

def disambiguate_repeated_titles(image_path, title_coords_candidates):
    final_coords = {}
    for title, candidates in title_coords_candidates.items():
        if len(candidates) == 1:
            final_coords[title] = candidates[0]
        else:
            final_coords[title] = candidates[0] 
    return final_coords

def find_title_coordinates_from_image_and_pdf(pdf_path):
    results = {}
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        temp_img = "temp_title_scan.png"
        page.to_image(resolution=300).save(temp_img)
        
        titles = find_title(temp_img)
        print(f"   > AI Found Titles: {titles}") 
        
        words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
        candidates = find_all_title_coordinates(words, titles)
        final_coords = disambiguate_repeated_titles(temp_img, candidates)
        
        results['page_1'] = final_coords
        if os.path.exists(temp_img): os.remove(temp_img)
        
    return results

def extract_text_from_response(response):
    if isinstance(response.content, list):
        return "".join([part["text"] for part in response.content if "text" in part]).strip()
    return str(response.content).strip()

def get_sheet_number(image_path: str) -> str:
    image_b64 = load_image_base64(image_path)
    prompt = "Look at the BOTTOM RIGHT CORNER. Extract the SHEET NUMBER (e.g., S-3.2)."
    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
    ])
    response = llm_pro.invoke([msg])
    return extract_text_from_response(response)


# Helper function for sementic_segementation

def normalize_bbox(bbox: List[float]) -> List[float]:
    """Clamps coordinates between 0.0 and 1.0"""
    return [max(0.0, min(1.0, val)) for val in bbox]

def image_to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded_string}"

def pil_to_data_url(pil_image: Image.Image) -> str:
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded_string}"

def crop_image_dynamic(original_path: str, bbox: List[float]) -> Image.Image:
    img = Image.open(original_path).convert("RGB")
    w, h = img.size
    x1, y1, x2, y2 = bbox
    
    # Pixel conversion
    left, top = int(x1 * w), int(y1 * h)
    right, bottom = int(x2 * w), int(y2 * h)

    # Safety: Ensure box has area
    if right - left < 10 or bottom - top < 10:
        return img.crop((left, top, left+50, top+50))
        
    return img.crop((left, top, right, bottom))