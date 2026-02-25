# from langgraph.graph import StateGraph, START, END
# import base64
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.messages import HumanMessage
# from typing import TypedDict, List
# from dotenv import load_dotenv
# import os
# import pdfplumber
# from pdfplumber import open as pdf_open
# from PIL import Image, ImageDraw, ImageFont
# # from utils.crop_in_quandrant import crop_image_into_quad
# # from utils.pdf_page_to_png import convert_specific_page_to_png
# from pydantic import BaseModel
# from typing import Literal
# import cv2 
# from PIL import Image
# import pdfplumber
# import cv2

# load_dotenv()

# llm = ChatGoogleGenerativeAI(
#     model="gemini-3-pro-preview"
# )
# llm_for_decision=ChatGoogleGenerativeAI(
#         model='gemini-2.0-flash-lite'
# )
# class TitleState(TypedDict):
#     architecturePdf: str
#     listOfTitle: List[str]

# class DrawingTypeResponse(BaseModel):
#     drawing_type: Literal["text", "floor", "section"]




# # # def deside_whether_notes_floor_section(imgURL:str):  
# # #     prompt = """
# # # You are analyzing an architectural / structural drawing sheet.

# # # Your task is to classify the image into ONE of the following categories:

# # # 1. "text"
# # #    - The sheet is mostly notes, specifications, guidelines, schedules, or tables
# # #    - Around 80–90% text
# # #    - No clear full plan or section drawing

# # # 2. "floor"
# # #    - A foundation plan, roof framing plan, or floor plan
# # #    - Clear plan view of a building or level
# # #    - Typically 50–60% or more of the image shows a single floor layout

# # # 3. "section"
# # #    - Section cuts or detail drawings
# # #    - Vertical cuts, stepped footings, wall sections, column details, joints, or construction details
# # #    - Focused on how elements are built, not the overall floor layout

# # # Rules:
# # # - Choose ONLY ONE category
# # # - Do NOT explain your answer
# # # - Do NOT guess if the image is unclear
# # # - Respond ONLY with valid JSON in this format:

# # # { "drawing_type": "text" | "floor" | "section" }

# # # """
# # #     image_base64 = load_image_base64(imgURL)

# # #     message = HumanMessage(
# # #     content=[
# # #         {"type": "text", "text": prompt},
# # #         {
# # #             "type": "image_url",
# # #             "image_url": {
# # #                 "url": f"data:image/png;base64,{image_base64}"
# # #             },
# # #         },
# # #     ]
# # # )
# # #     structured_llm = llm_for_decision.with_structured_output(DrawingTypeResponse)
# # #     response: DrawingTypeResponse = structured_llm.invoke([message])
# # #     return response

# # # print(deside_whether_notes_floor_section('langgraph_temp_workflow/section.png'))



# # # def handel_floor_plan(pdf_path: str, page_num: int):
# # #     out_path = "floor"

# # #     if not os.path.exists(out_path):
# # #         os.mkdir(out_path)

# # #     dpi = 300
# # #     floor_img = f"{out_path}/floor.png"

# # #     convert_specific_page_to_png(pdf_path, page_num, floor_img, dpi)
# # #     if not os.path.exists(floor_img):
# # #         raise ValueError("image not created")

# # #     # creates 8 cropped images in out_path
# # #     crop_image_into_quad(floor_img, out_path)

# # #     img_list = []
# # #     for img_name in os.listdir(out_path):

# # #             img_name.lower().endswith((".png", ".jpg", ".jpeg"))
# # #             img_path = os.path.join(out_path, img_name)
# # #             image_base64 = load_image_base64(img_path)
# # #             img_list.append(image_base64)

# # #     prompt = """
# # # Act as a Structural Data Extraction Agent and Civil Engineer. Your goal is to perform a comprehensive Material Takeoff (MTO) by extracting structural notations and geometric data from the provided "Global" plan and 8 high-resolution zoomed images.

# # # ### YOUR MISSION:
# # # Identify and list every structural member, section callout, and annotation explicitly labeled on this drawing. You have expert knowledge in reading floor plans and structural notations.

# # # ### EXTRACTION RULES:
# # # 1. NO CALCULATIONS: Do not calculate weights, lengths, or totals. Extract text exactly as written.
# # # 2. NO GUESSING: If a grid segment has no written notation, do not report data for that area.
# # # 3. VISUAL ANCHORING: Map every extracted item to its specific Grid Coordinates (e.g., A-G, 1-3).
# # # 4. COMPREHENSIVE SIGHT: 
# # #    - Extract all member sizes (e.g., W-shapes, L-shapes, or other notations).
# # #    - Identify all section callouts (e.g., notations like 3/S-4.1, 9/S-4.1).
# # #    - Capture all textual notes found outside the main drawing area (Schedules, General Notes, Legend).

# # # ### OUTPUT FORMAT (JSON):
# # # {
# # #   "material_takeoff": [
# # #     {
# # #       "type": "Member / Section Callout / Note",
# # #       "label": "Exactly as written (e.g., W24x62 or 3/S-4.1)",
# # #       "grid_location": "The specific grid line or span where it is located",
# # #       "geometric_context": "Direction (North-South/East-West) or associated dimension (e.g., 27'-0\")",
# # #       "image_source": "top_left / top_second / top_third / top_last / bottom_left / bottom_second / bottom_third / bottom_last"
# # #     }
# # #   ],
# # #   "general_notes_and_schedules": {
# # #     "extracted_schedules": ["List names of visible schedules, e.g., 'Wall Opening Schedule'"],
# # #     "important_notes": ["List any specific text notes found on the sheet"]
# # #   }
# # # }

# # # ### CHAIN-OF-THOUGHT (FOLLOW THESE STEPS):
# # # 1. SCAN: Look through each of the 8 high-resolution zoomed images (top_left, top_second, top_third, top_last, bottom_left, bottom_second, bottom_third, bottom_last) for structural notations.
# # # 2. ANCHOR: For every item found, look at the closest Grid Lines to define its geometric position.
# # # 3. SYMBOL RECOGNITION: Recognize specific symbols such as Beam Splices (S-mark) and Section Cuts (triangular bubbles with page numbers like S-4.1).
# # # 4. SCHEDULE EXTRACTION: Read the 'Wall Opening Schedule', 'Roof Truss Loads', and 'Snow Drift Loading Schedule' to capture non-drawing data.
# # # 5. VERIFY: Use the 'Global' plan to ensure the data from the 8 images is correctly placed in the building's overall 120'-4" x 68'-0" footprint.
# # # """
# # #     content=[
# # #         {"type": "text", "text": prompt}
# # #     ]
# # #     for img_base64 in img_list:
# # #         content.append(
# # #             {
# # #                 "type":"image_url",
# # #                 "image_url":{
# # #             "url": f"data:image/png;base64,{img_base64}"
# # #              }
# # #             }
# # #         )
# # #     message = HumanMessage(
# # #     content=content
# # # )
# # #     response = llm.invoke([message])
# # #     return response.content

# # # print(handel_floor_plan('langgraph_temp_workflow/floor.pdf',0))
  

# # # Initial state
# # initial_state = {
# #     "architecturePdf": "langgraph_temp_workflow/temp_ext.pdf"
# # }

# # graph = StateGraph(TitleState)
# # graph.add_node("find_crop_image", find_title)
# # graph.add_edge(START, "find_crop_image")
# # graph.add_edge("find_crop_image", END)

# # workflow = graph.compile()
# # ans = workflow.invoke(initial_state)

# # print(ans["listOfTitle"])





import os
import json
import base64
import shutil
from typing import TypedDict, List, Dict, Any, Literal
from dotenv import load_dotenv

# LangChain / LangGraph Imports
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

# PDF & Image Processing Imports
import pdfplumber
from PIL import Image, ImageDraw
from pypdf import PdfReader, PdfWriter
from utils.pdf_page_to_png import convert_specific_page_to_png

load_dotenv()

# --- 1. SETUP MODELS ---
llm_pro = ChatGoogleGenerativeAI(model="gemini-2.5-pro") 
llm_flash = ChatGoogleGenerativeAI(model="gemini-2.0-flash") 

# --- 2. DEFINE STATE ---
class ProjectState(TypedDict):
    pdf_path: str
    output_dir: str
    page_map: Dict[int, str] 
    detail_library: Dict[str, Any] 
    general_rules: str 
    final_estimates: List[Dict]

# --- 3. HELPER FUNCTIONS (Integrated) ---

def load_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# --- Title Finding Logic ---
class TitleChoice(BaseModel):
    choice: int  

def find_title(image_path: str):
    image_base64 = load_image_base64(image_path)
    prompt = """
    Extract ONLY the section titles present in this drawing Sheet.
    Titles are bold, uppercase, and placed below details.
    Return each title on a new line. Do not include extra text.
    """
    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
    ])
    response = llm_flash.invoke([message])
    return [line.strip() for line in response.content.split("\n") if line.strip()]

def find_all_title_coordinates(words, titles):
    results = {} 
    word_texts = [w["text"].upper() for w in words]

    for title in titles:
        title_words = title.upper().split()
        n = len(title_words)
        candidates = []
        for i in range(len(word_texts) - n + 1):
            if word_texts[i:i+n] == title_words:
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
            # For speed, we default to the first candidate. 
            # You can enable the LLM check here if needed, but it consumes tokens.
            final_coords[title] = candidates[0] 
    return final_coords

def find_title_coordinates_from_image_and_pdf(pdf_path):
    results = {}
    # We assume single page PDF here based on usage in the node
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        temp_img = "temp_title_scan.png"
        page.to_image(resolution=300).save(temp_img)
        
        titles = find_title(temp_img)
        words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
        candidates = find_all_title_coordinates(words, titles)
        final_coords = disambiguate_repeated_titles(temp_img, candidates)
        
        results['page_1'] = final_coords # Using a fixed key for the single page
        if os.path.exists(temp_img): os.remove(temp_img)
        
    return results

# --- Robust Cropping Logic ---
def crop_sections_from_page(coords_dict, page_image_path, pdf_path, output_dir="cropped_sections"):
    os.makedirs(output_dir, exist_ok=True)
    
    page_image = Image.open(page_image_path)
    img_width, img_height = page_image.size

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        pdf_width = page.width
        pdf_height = page.height

    # Internal Scaling
    scale_x = img_width / pdf_width
    scale_y = img_height / pdf_height

    scaled = {}
    for title, c in coords_dict.items():
        scaled[title] = {
            "x1": int(c["x1"] * scale_x), "y1": int(c["y1"] * scale_y),
            "x2": int(c["x2"] * scale_x), "y2": int(c["y2"] * scale_y),
        }

    # Row Grouping
    sorted_titles = sorted(scaled.keys(), key=lambda t: scaled[t]["y2"])
    rows = []
    current_row = []
    current_row_y_max = -1
    ROW_THRESHOLD = 50 

    for title in sorted_titles:
        y2 = scaled[title]["y2"]
        if not current_row:
            current_row.append(title)
            current_row_y_max = y2
        else:
            if abs(y2 - current_row_y_max) < ROW_THRESHOLD:
                current_row.append(title)
                current_row_y_max = max(current_row_y_max, y2)
            else:
                rows.append(current_row)
                current_row = [title]
                current_row_y_max = y2
    if current_row: rows.append(current_row)

    cropped_sections = []
    prev_y = 0 

    for row_idx, titles_in_row in enumerate(rows, start=1):
        row_y_bottom = max(scaled[t]["y2"] for t in titles_in_row)
        
        if row_y_bottom <= prev_y: continue 

        cropped_row = page_image.crop((0, prev_y, img_width, row_y_bottom))
        titles_in_row_sorted = sorted(titles_in_row, key=lambda t: scaled[t]["x1"])

        left_margin, right_margin = 400, 200

        for col_idx, t in enumerate(titles_in_row_sorted):
            curr_x1 = scaled[t]["x1"]
            x_start = max(0, curr_x1 - left_margin)

            if col_idx < len(titles_in_row_sorted) - 1:
                next_x1 = scaled[titles_in_row_sorted[col_idx + 1]]["x1"]
                x_end = max(0, next_x1 - right_margin)
            else:
                x_end = img_width

            if x_end <= x_start: continue

            cropped_section = cropped_row.crop((x_start, 0, x_end, cropped_row.height))
            
            safe_title = "".join(c for c in t if c.isalnum() or c in (' ', '_')).strip()
            save_path = os.path.join(output_dir, f"{safe_title}.png")
            cropped_section.save(save_path)

            cropped_sections.append({"title": t, "image_path": save_path})
        
        prev_y = row_y_bottom + 10 

    return cropped_sections

# --- 4. AGENT 1: CLASSIFIER NODE ---
class DrawingTypeResponse(BaseModel):
    drawing_type: Literal["text", "floor", "section"]

def node_classify_pages(state: ProjectState):
    print("--- NODE: Classifying Pages ---")
    pdf_path = state["pdf_path"]
    page_map = {}
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        
    for page_num in range(total_pages):
        temp_img_path = f"{state['output_dir']}/temp_page_{page_num}.png"
        convert_specific_page_to_png(pdf_path, page_num, temp_img_path, dpi=150)
        
        prompt = """
        Classify this construction sheet into ONE category:
        1. "text" (Notes, Schedules, Tables)
        2. "floor" (Plan View, Foundation, Roof Framing)
        3. "section" (Details, Wall Sections, Cuts)
        """
        image_b64 = load_image_base64(temp_img_path)
        msg = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
        ])
        
        result = llm_flash.with_structured_output(DrawingTypeResponse).invoke([msg])
        page_map[page_num] = result.drawing_type
        print(f"Page Index {page_num} (PDF Page {page_num+1}): {result.drawing_type}")

    return {"page_map": page_map}

# --- 5. AGENT 1: TEXT PROCESSOR ---
def node_process_text_rules(state: ProjectState):
    print("--- NODE: Processing Text Rules ---")
    text_pages = [p for p, t in state["page_map"].items() if t == "text"]
    accumulated_rules = ""
    
    for page_num in text_pages:
        with pdfplumber.open(state["pdf_path"]) as pdf:
            text = pdf.pages[page_num].extract_text() or ""
            
        prompt = f"Extract structural rules (Lintel schedules, Bolt spacing) from:\n{text}"
        msg = HumanMessage(content=prompt)
        response = llm_flash.invoke([msg])
        accumulated_rules += f"\nPage {page_num}: {response.content}\n"
        
    return {"general_rules": accumulated_rules}

# --- 6. AGENT 3: DETAIL PROCESSOR ---
def node_process_details(state: ProjectState):
    print("--- NODE: Processing Details ---")
    detail_library = state.get("detail_library", {})
    section_pages = [p for p, t in state["page_map"].items() if t == "section"]
    
    for page_num in section_pages:
        print(f"Processing Details on Page Index {page_num} (PDF Page {page_num+1})...")
        
        page_img_path = f"{state['output_dir']}/section_page_{page_num}.png"
        page_pdf_path = f"{state['output_dir']}/section_page_{page_num}.pdf"
        
        # 1. Convert to Image
        convert_specific_page_to_png(state["pdf_path"], page_num, page_img_path, dpi=300)
        
        # 2. Extract Single Page PDF
        try:
            reader = PdfReader(state["pdf_path"])
            writer = PdfWriter()
            writer.add_page(reader.pages[page_num])
            with open(page_pdf_path, "wb") as f: writer.write(f)
        except Exception as e:
            print(f"PDF extraction failed: {e}")
            continue

        # 3. Get Coords
        try:
            all_coords = find_title_coordinates_from_image_and_pdf(page_pdf_path)
            # We use 'page_1' because the temp PDF only has 1 page
            coords_dict = all_coords.get('page_1', {})
        except Exception as e:
            print(f"Coord extraction failed: {e}")
            coords_dict = {}

        if not coords_dict:
            print(f"No titles found on page {page_num}. Skipping.")
            continue

        # 4. Crop (NO DOUBLE SCALING HERE)
        # We pass the raw PDF coords. The function handles scaling internally.
        cropped_sections = crop_sections_from_page(
            coords_dict, 
            page_img_path, 
            page_pdf_path, 
            output_dir=f"{state['output_dir']}/crops_{page_num}"
        )
        
        # 5. Analyze
        for crop in cropped_sections:
            title = crop['title']
            print(f"Analyzing detail: {title}")
            
            prompt = f"""
            Analyze this structural detail titled "{title}".
            Identify the specific BOM (Bill of Materials) components.
            Look for:
            - Steel Profiles (e.g., MC6x15.1, C-Channels)
            - Angles (e.g., L4x4x1/4)
            - Plates (e.g., PL 1/4" x 4")
            - Rods/Bolts (e.g., 3/4" Anchor Bolt)
            
            Return ONLY valid JSON. No markdown formatting.
            Format:
            {{
                "symbol_ref": "{title}",
                "components": [
                    {{"item": "Material Name", "qty_rule": "count/length", "notes": "context"}}
                ]
            }}
            """
            
            img_b64 = load_image_base64(crop["image_path"])
            msg = HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            ])
            
            try:
                resp = llm_pro.invoke([msg])
                # Clean the response string to ensure valid JSON
                json_str = resp.content.strip()
                if json_str.startswith("```json"):
                    json_str = json_str.replace("```json", "").replace("```", "")
                
                detail_library[title] = json.loads(json_str)
            except Exception as e:
                print(f"Failed to parse JSON for {title}: {e}")
                # Optional: Add a fallback or raw text if JSON fails
                detail_library[title] = {"raw_text": resp.content}

    return {"detail_library": detail_library}

# --- 7. AGENT 2: PLAN ESTIMATOR ---
# Helper for Quadrant Cropping (Assuming you have this logic)
def crop_image_into_quad(image_path, output_folder):
    # Simple placeholder logic if you don't have the external file
    img = Image.open(image_path)
    w, h = img.size
    mid_w, mid_h = w // 2, h // 2
    quads = [
        (0, 0, mid_w, mid_h), (mid_w, 0, w, mid_h),
        (0, mid_h, mid_w, h), (mid_w, mid_h, w, h)
    ]
    os.makedirs(output_folder, exist_ok=True)
    for i, box in enumerate(quads):
        img.crop(box).save(f"{output_folder}/quad_{i}.png")

def node_process_plans(state: ProjectState):
    print("--- NODE: Processing Plans ---")
    final_estimates = []
    floor_pages = [p for p, t in state["page_map"].items() if t == "floor"]
    
    context = json.dumps(state["detail_library"], indent=2)
    rules = state["general_rules"]
    
    for page_num in floor_pages:
        print(f"Estimating Page Index {page_num}...")
        page_dir = f"{state['output_dir']}/floor_{page_num}"
        convert_specific_page_to_png(state["pdf_path"], page_num, f"{page_dir}.png", dpi=300)
        crop_image_into_quad(f"{page_dir}.png", page_dir)
        
        quads = [os.path.join(page_dir, f) for f in os.listdir(page_dir) if f.endswith(".png")]
        
        prompt = f"""
        Act as a Senior Structural Estimator.
        CONTEXT: {context}
        RULES: {rules}
        
        Task: Extract Material Takeoff (HSS, WF, C, L, FB, ROD).
        - Count HSS Columns (Use 18.29' height).
        - Measure WF Beams via Grids.
        - Link Symbols (e.g. 7/S-3.2) to Context Library.
        
        Return JSON list of items.
        """
        
        content = [{"type": "text", "text": prompt}]
        for q in quads:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{load_image_base64(q)}"}} )
            
        msg = HumanMessage(content=content)
        resp = llm_pro.invoke([msg])
        try:
            json_str = resp.content.strip().replace("```json", "").replace("```", "")
            final_estimates.append(json.loads(json_str))
        except: pass

    return {"final_estimates": final_estimates}

# --- 8. BUILD GRAPH ---
workflow = StateGraph(ProjectState)
workflow.add_node("classify", node_classify_pages)
workflow.add_node("process_text", node_process_text_rules)
workflow.add_node("process_details", node_process_details)
workflow.add_node("process_plans", node_process_plans)

workflow.add_edge(START, "classify")
workflow.add_edge("classify", "process_text")
workflow.add_edge("classify", "process_details")
workflow.add_edge("process_text", "process_plans")
workflow.add_edge("process_details", "process_plans")
workflow.add_edge("process_plans", END)

app = workflow.compile()

if __name__ == "__main__":
    os.makedirs("output_temp", exist_ok=True)
    
    # Replace with your actual PDF path
    state = {
        "pdf_path": "langgraph_temp_workflow/input.pdf",
        "output_dir": "output_temp",
        "page_map": {}, "detail_library": {}, "general_rules": "", "final_estimates": []
    }
    
    print("Starting Estimation Workflow...")
    result = app.invoke(state)
    
    print("\n\n=== FINAL ESTIMATION REPORT ===")
    print(json.dumps(result["final_estimates"], indent=2))

