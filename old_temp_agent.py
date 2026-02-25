# import os
# import json
# import base64
# import shutil
# from typing import TypedDict, List, Dict, Any, Literal
# from dotenv import load_dotenv

# # LangChain / LangGraph Imports
# from langgraph.graph import StateGraph, START, END
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.messages import HumanMessage
# from pydantic import BaseModel, Field

# # PDF & Image Processing Imports
# import pdfplumber
# from PIL import Image, ImageDraw
# from pypdf import PdfReader, PdfWriter
# from utils.pdf_page_to_png import convert_specific_page_to_png
# from utils.crop_in_quandrant import crop_image_into_quad
# from utils.croped_sections import crop_sections_from_page

# load_dotenv()

# # --- 1. SETUP MODELS ---
# llm_pro = ChatGoogleGenerativeAI(model="gemini-2.5-pro") 
# llm_flash = ChatGoogleGenerativeAI(model="gemini-2.0-flash") 

# # --- 2. DEFINE STATE ---
# class ProjectState(TypedDict):
#     pdf_path: str
#     output_dir: str
#     page_map: Dict[int, str] 
#     detail_library: Dict[str, Any] 
#     general_rules: str 
#     final_estimates: List[Dict]

# # --- 3. HELPER FUNCTIONS (Integrated) ---

# def load_image_base64(image_path: str) -> str:
#     with open(image_path, "rb") as f:
#         return base64.b64encode(f.read()).decode("utf-8")

# # --- Title Finding Logic ---
# class TitleChoice(BaseModel):
#     choice: int  

# def find_title(image_path: str):
#     image_base64 = load_image_base64(image_path)
#     prompt = """
#     Extract ONLY the section titles present in this drawing Sheet.
#     Titles are bold, uppercase, and placed below details.
#     Return each title on a new line. Do not include extra text.
#     """
#     message = HumanMessage(content=[
#         {"type": "text", "text": prompt},
#         {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
#     ])
#     response = llm_flash.invoke([message])
#     return [line.strip() for line in response.content.split("\n") if line.strip()]

# def find_all_title_coordinates(words, titles):
#     results = {} 
#     word_texts = [w["text"].upper() for w in words]

#     for title in titles:
#         title_words = title.upper().split()
#         n = len(title_words)
#         candidates = []
#         for i in range(len(word_texts) - n + 1):
#             if word_texts[i:i+n] == title_words:
#                 boxes = words[i:i+n]
#                 candidates.append({
#                     "x1": min(w["x0"] for w in boxes),
#                     "y1": min(w["top"] for w in boxes),
#                     "x2": max(w["x1"] for w in boxes),
#                     "y2": max(w["bottom"] for w in boxes)
#                 })
#         if candidates:
#             results[title] = candidates
#     return results

# def disambiguate_repeated_titles(image_path, title_coords_candidates):
#     final_coords = {}
#     for title, candidates in title_coords_candidates.items():
#         if len(candidates) == 1:
#             final_coords[title] = candidates[0]
#         else:
#             # For speed, we default to the first candidate. 
#             # You can enable the LLM check here if needed, but it consumes tokens.
#             final_coords[title] = candidates[0] 
#     return final_coords

# def find_title_coordinates_from_image_and_pdf(pdf_path):
#     results = {}
#     # We assume single page PDF here based on usage in the node
#     with pdfplumber.open(pdf_path) as pdf:
#         page = pdf.pages[0]
#         temp_img = "temp_title_scan.png"
#         page.to_image(resolution=300).save(temp_img)
        
#         titles = find_title(temp_img)
#         words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
#         candidates = find_all_title_coordinates(words, titles)
#         final_coords = disambiguate_repeated_titles(temp_img, candidates)
        
#         results['page_1'] = final_coords # Using a fixed key for the single page
#         if os.path.exists(temp_img): os.remove(temp_img)
        
#     return results

# # --- Robust Cropping Logic ---
# def crop_sections_from_page(coords_dict, page_image_path, pdf_path, output_dir="cropped_sections"):
#     os.makedirs(output_dir, exist_ok=True)
    
#     page_image = Image.open(page_image_path)
#     img_width, img_height = page_image.size

#     with pdfplumber.open(pdf_path) as pdf:
#         page = pdf.pages[0]
#         pdf_width = page.width
#         pdf_height = page.height

#     # Internal Scaling
#     scale_x = img_width / pdf_width
#     scale_y = img_height / pdf_height

#     scaled = {}
#     for title, c in coords_dict.items():
#         scaled[title] = {
#             "x1": int(c["x1"] * scale_x), "y1": int(c["y1"] * scale_y),
#             "x2": int(c["x2"] * scale_x), "y2": int(c["y2"] * scale_y),
#         }

#     # Row Grouping
#     sorted_titles = sorted(scaled.keys(), key=lambda t: scaled[t]["y2"])
#     rows = []
#     current_row = []
#     current_row_y_max = -1
#     ROW_THRESHOLD = 50 

#     for title in sorted_titles:
#         y2 = scaled[title]["y2"]
#         if not current_row:
#             current_row.append(title)
#             current_row_y_max = y2
#         else:
#             if abs(y2 - current_row_y_max) < ROW_THRESHOLD:
#                 current_row.append(title)
#                 current_row_y_max = max(current_row_y_max, y2)
#             else:
#                 rows.append(current_row)
#                 current_row = [title]
#                 current_row_y_max = y2
#     if current_row: rows.append(current_row)

#     cropped_sections = []
#     prev_y = 0 

#     for row_idx, titles_in_row in enumerate(rows, start=1):
#         row_y_bottom = max(scaled[t]["y2"] for t in titles_in_row)
        
#         if row_y_bottom <= prev_y: continue 

#         cropped_row = page_image.crop((0, prev_y, img_width, row_y_bottom))
#         titles_in_row_sorted = sorted(titles_in_row, key=lambda t: scaled[t]["x1"])

#         left_margin, right_margin = 400, 200

#         for col_idx, t in enumerate(titles_in_row_sorted):
#             curr_x1 = scaled[t]["x1"]
#             x_start = max(0, curr_x1 - left_margin)

#             if col_idx < len(titles_in_row_sorted) - 1:
#                 next_x1 = scaled[titles_in_row_sorted[col_idx + 1]]["x1"]
#                 x_end = max(0, next_x1 - right_margin)
#             else:
#                 x_end = img_width

#             if x_end <= x_start: continue

#             cropped_section = cropped_row.crop((x_start, 0, x_end, cropped_row.height))
            
#             safe_title = "".join(c for c in t if c.isalnum() or c in (' ', '_')).strip()
#             save_path = os.path.join(output_dir, f"{safe_title}.png")
#             cropped_section.save(save_path)

#             cropped_sections.append({"title": t, "image_path": save_path})
        
#         prev_y = row_y_bottom + 10 

#     return cropped_sections

# # --- 4. AGENT 1: CLASSIFIER NODE ---
# class DrawingTypeResponse(BaseModel):
#     drawing_type: Literal["text", "floor", "section"]

# def node_classify_pages(state: ProjectState):
#     print("--- NODE: Classifying Pages ---")
#     pdf_path = state["pdf_path"]
#     page_map = {}
    
#     with pdfplumber.open(pdf_path) as pdf:
#         total_pages = len(pdf.pages)
        
#     for page_num in range(total_pages):
#         temp_img_path = f"{state['output_dir']}/temp_page_{page_num}.png"
#         convert_specific_page_to_png(pdf_path, page_num, temp_img_path, dpi=150)
        
#         prompt = """
#         Classify this construction sheet into ONE category:
#         1. "text" (Notes, Schedules, Tables)
#         2. "floor" (Plan View, Foundation, Roof Framing)
#         3. "section" (Details, Wall Sections, Cuts)
#         """
#         image_b64 = load_image_base64(temp_img_path)
#         msg = HumanMessage(content=[
#             {"type": "text", "text": prompt},
#             {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
#         ])
        
#         result = llm_flash.with_structured_output(DrawingTypeResponse).invoke([msg])
#         page_map[page_num] = result.drawing_type
#         print(f"Page Index {page_num} (PDF Page {page_num+1}): {result.drawing_type}")

#     return {"page_map": page_map}

# # --- 5. AGENT 1: TEXT PROCESSOR ---
# def node_process_text_rules(state: ProjectState):
#     print("--- NODE: Processing Text Rules ---")
#     text_pages = [p for p, t in state["page_map"].items() if t == "text"]
#     accumulated_rules = ""
    
#     for page_num in text_pages:
#         with pdfplumber.open(state["pdf_path"]) as pdf:
#             text = pdf.pages[page_num].extract_text() or ""
            
#         prompt = f"Extract structural rules (Lintel schedules, Bolt spacing) from:\n{text}"
#         msg = HumanMessage(content=prompt)
#         response = llm_flash.invoke([msg])
#         accumulated_rules += f"\nPage {page_num}: {response.content}\n"
        
#     return {"general_rules": accumulated_rules}

# # --- 6. AGENT 3: DETAIL PROCESSOR ---
# def node_process_details(state: ProjectState):
#     print("--- NODE: Processing Details ---")
#     detail_library = state.get("detail_library", {})
#     section_pages = [p for p, t in state["page_map"].items() if t == "section"]
    
#     for page_num in section_pages:
#         print(f"Processing Details on Page Index {page_num} (PDF Page {page_num+1})...")
        
#         page_img_path = f"{state['output_dir']}/section_page_{page_num}.png"
#         page_pdf_path = f"{state['output_dir']}/section_page_{page_num}.pdf"
        
#         # 1. Convert to Image
#         convert_specific_page_to_png(state["pdf_path"], page_num, page_img_path, dpi=300)
        
#         # 2. Extract Single Page PDF
#         try:
#             reader = PdfReader(state["pdf_path"])
#             writer = PdfWriter()
#             writer.add_page(reader.pages[page_num])
#             with open(page_pdf_path, "wb") as f: writer.write(f)
#         except Exception as e:
#             print(f"PDF extraction failed: {e}")
#             continue

#         # 3. Get Coords
#         try:
#             all_coords = find_title_coordinates_from_image_and_pdf(page_pdf_path)
#             # We use 'page_1' because the temp PDF only has 1 page
#             coords_dict = all_coords.get('page_1', {})
#         except Exception as e:
#             print(f"Coord extraction failed: {e}")
#             coords_dict = {}

#         if not coords_dict:
#             print(f"No titles found on page {page_num}. Skipping.")
#             continue

#         # 4. Crop (NO DOUBLE SCALING HERE)
#         # We pass the raw PDF coords. The function handles scaling internally.
#         cropped_sections = crop_sections_from_page(
#             coords_dict, 
#             page_img_path, 
#             page_pdf_path, 
#             output_dir=f"{state['output_dir']}/crops_{page_num}"
#         )
        
#         # 5. Analyze
#         for crop in cropped_sections:
#             title = crop['title']
#             print(f"Analyzing detail: {title}")
            
#             prompt = f"""
#             Analyze this structural detail titled "{title}".
#             Identify the specific BOM (Bill of Materials) components.
#             Look for:
#             - Steel Profiles (e.g., MC6x15.1, C-Channels)
#             - Angles (e.g., L4x4x1/4)
#             - Plates (e.g., PL 1/4" x 4")
#             - Rods/Bolts (e.g., 3/4" Anchor Bolt)
            
#             Return ONLY valid JSON. No markdown formatting.
#             Format:
#             {{
#                 "symbol_ref": "{title}",
#                 "components": [
#                     {{"item": "Material Name", "qty_rule": "count/length", "notes": "context"}}
#                 ]
#             }}
#             """
            
#             img_b64 = load_image_base64(crop["image_path"])
#             msg = HumanMessage(content=[
#                 {"type": "text", "text": prompt},
#                 {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
#             ])
            
#             try:
#                 resp = llm_pro.invoke([msg])
#                 # Clean the response string to ensure valid JSON
#                 json_str = resp.content.strip()
#                 if json_str.startswith("```json"):
#                     json_str = json_str.replace("```json", "").replace("```", "")
                
#                 detail_library[title] = json.loads(json_str)
#             except Exception as e:
#                 print(f"Failed to parse JSON for {title}: {e}")
#                 # Optional: Add a fallback or raw text if JSON fails
#                 detail_library[title] = {"raw_text": resp.content}

#     return {"detail_library": detail_library}

# # --- 7. AGENT 2: PLAN ESTIMATOR ---
# # Helper for Quadrant Cropping (Assuming you have this logic)
# def crop_image_into_quad(image_path, output_folder):
#     # Simple placeholder logic if you don't have the external file
#     img = Image.open(image_path)
#     w, h = img.size
#     mid_w, mid_h = w // 2, h // 2
#     quads = [
#         (0, 0, mid_w, mid_h), (mid_w, 0, w, mid_h),
#         (0, mid_h, mid_w, h), (mid_w, mid_h, w, h)
#     ]
#     os.makedirs(output_folder, exist_ok=True)
#     for i, box in enumerate(quads):
#         img.crop(box).save(f"{output_folder}/quad_{i}.png")

# def node_process_plans(state: ProjectState):
#     print("--- NODE: Processing Plans ---")
#     final_estimates = []
#     floor_pages = [p for p, t in state["page_map"].items() if t == "floor"]
    
#     context = json.dumps(state["detail_library"], indent=2)
#     rules = state["general_rules"]
    
#     for page_num in floor_pages:
#         print(f"Estimating Page Index {page_num}...")
#         page_dir = f"{state['output_dir']}/floor_{page_num}"
#         convert_specific_page_to_png(state["pdf_path"], page_num, f"{page_dir}.png", dpi=300)
#         crop_image_into_quad(f"{page_dir}.png", page_dir)
        
#         quads = [os.path.join(page_dir, f) for f in os.listdir(page_dir) if f.endswith(".png")]
        
#         prompt = f"""
#         Act as a Senior Structural Estimator.
#         CONTEXT: {context}
#         RULES: {rules}
        
#         Task: Extract Material Takeoff (HSS, WF, C, L, FB, ROD).
#         - Count HSS Columns (Use 18.29' height).
#         - Measure WF Beams via Grids.
#         - Link Symbols (e.g. 7/S-3.2) to Context Library.
        
#         Return JSON list of items.
#         """
        
#         content = [{"type": "text", "text": prompt}]
#         for q in quads:
#             content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{load_image_base64(q)}"}} )
            
#         msg = HumanMessage(content=content)
#         resp = llm_pro.invoke([msg])
#         try:
#             json_str = resp.content.strip().replace("```json", "").replace("```", "")
#             final_estimates.append(json.loads(json_str))
#         except: pass

#     return {"final_estimates": final_estimates}

# # --- 8. BUILD GRAPH ---
# workflow = StateGraph(ProjectState)
# workflow.add_node("classify", node_classify_pages)
# workflow.add_node("process_text", node_process_text_rules)
# workflow.add_node("process_details", node_process_details)
# workflow.add_node("process_plans", node_process_plans)

# workflow.add_edge(START, "classify")
# workflow.add_edge("classify", "process_text")
# workflow.add_edge("classify", "process_details")
# workflow.add_edge("process_text", "process_plans")
# workflow.add_edge("process_details", "process_plans")
# workflow.add_edge("process_plans", END)

# app = workflow.compile()

# if __name__ == "__main__":
#     os.makedirs("output_temp", exist_ok=True)
    
#     # Replace with your actual PDF path
#     state = {
#         "pdf_path": "langgraph_temp_workflow/input.pdf",
#         "output_dir": "output_temp",
#         "page_map": {}, "detail_library": {}, "general_rules": "", "final_estimates": []
#     }
    
#     print("Starting Estimation Workflow...")
#     result = app.invoke(state)
    
#     print("\n\n=== FINAL ESTIMATION REPORT ===")
#     print(json.dumps(result["final_estimates"], indent=2))















# import os
# import json
# import base64
# import re
# from typing import TypedDict, List, Dict, Any, Literal
# from dotenv import load_dotenv

# # LangChain / LangGraph Imports
# from langgraph.graph import StateGraph, START, END
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.messages import HumanMessage
# from pydantic import BaseModel

# # PDF & Image Processing Imports
# import pdfplumber
# from PIL import Image
# from pypdf import PdfReader, PdfWriter
# from utils.pdf_page_to_png import convert_specific_page_to_png

# load_dotenv()

# # --- 1. SETUP MODELS ---
# llm_pro = ChatGoogleGenerativeAI(model="gemini-3-flash-preview") 
# llm_flash = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite") 

# # --- 2. DEFINE STATE ---
# class ProjectState(TypedDict):
#     pdf_path: str
#     output_dir: str
#     page_map: Dict[int, str] 
#     detail_library: Dict[str, Any] 
#     general_rules: str 
#     final_estimates: List[Dict]

# # --- 3. HELPER FUNCTIONS ---

# def load_image_base64(image_path: str) -> str:
#     with open(image_path, "rb") as f:
#         return base64.b64encode(f.read()).decode("utf-8")

# # --- NEW: 2x4 GRID CROPPER (8 Parts) ---
# def crop_image_into_2x4_grid(image_path, output_folder):
#     """Crops image into 2 Rows and 4 Columns (8 total images)."""
#     img = Image.open(image_path)
#     w, h = img.size
    
#     # Calculate step sizes
#     col_step = w // 4
#     row_step = h // 2
    
#     os.makedirs(output_folder, exist_ok=True)
    
#     quad_paths = []
#     count = 0
    
#     # Overlap buffer (to ensure text on the cut line isn't lost)
#     buffer = 50 

#     for r in range(2):
#         for c in range(4):
#             # Calculate coordinates with buffer
#             x1 = max(0, (c * col_step) - buffer)
#             y1 = max(0, (r * row_step) - buffer)
#             x2 = min(w, ((c + 1) * col_step) + buffer)
#             y2 = min(h, ((r + 1) * row_step) + buffer)
            
#             crop_name = f"grid_r{r}_c{c}.png"
#             save_path = os.path.join(output_folder, crop_name)
            
#             img.crop((x1, y1, x2, y2)).save(save_path)
#             quad_paths.append(save_path)
#             count += 1
            
#     return quad_paths

# # --- TITLE FINDING LOGIC (IMPROVED) ---

# def normalize_text(text):
#     """Removes punctuation and extra spaces for better matching."""
#     return re.sub(r'[^a-zA-Z0-9]', '', text).upper()

# def find_title(image_path: str):
#     image_base64 = load_image_base64(image_path)
#     prompt = """
#     Look at this construction sheet.
#     Extract ONLY the bold, uppercase TITLES found below the detail drawings.
#     Examples: "TYP. WALL SECTION", "LADDER DETAIL", "SECTION A-A".
#     Do NOT extract notes, dimensions, or material labels.
#     Return just the titles, one per line.
#     """
#     message = HumanMessage(content=[
#         {"type": "text", "text": prompt},
#         {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
#     ])
#     response = llm_flash.invoke([message])
#     titles = [line.strip() for line in response.content.split("\n") if line.strip()]
#     return titles

# def find_all_title_coordinates(words, titles):
#     results = {} 
#     # Create a normalized map of the PDF text
#     pdf_text_sequence = []
#     for w in words:
#         pdf_text_sequence.append({
#             "text": normalize_text(w["text"]),
#             "obj": w
#         })
    
#     for title in titles:
#         norm_title = normalize_text(title)
#         if not norm_title: continue
        
#         # Sliding window search on normalized text
#         candidates = []
#         # We don't know exactly how many words the title splits into in the PDF
#         # So we look for the sequence of characters
        
#         # Simple heuristic: Match word sequences
#         title_words = title.split()
#         n = len(title_words)
        
#         # Search in the original words list (fuzzy match)
#         for i in range(len(words) - n + 1):
#             # Construct string from n words
#             segment = "".join([normalize_text(words[j]["text"]) for j in range(i, i+n)])
            
#             if norm_title in segment: # Partial match allowed
#                 boxes = words[i:i+n]
#                 candidates.append({
#                     "x1": min(w["x0"] for w in boxes),
#                     "y1": min(w["top"] for w in boxes),
#                     "x2": max(w["x1"] for w in boxes),
#                     "y2": max(w["bottom"] for w in boxes)
#                 })
        
#         if candidates:
#             results[title] = candidates
            
#     return results

# def find_title_coordinates_from_image_and_pdf(pdf_path):
#     results = {}
#     with pdfplumber.open(pdf_path) as pdf:
#         page = pdf.pages[0]
#         temp_img = "temp_title_scan.png"
#         page.to_image(resolution=300).save(temp_img)
        
#         titles = find_title(temp_img)
#         print(f"   > AI Found Titles: {titles}") # Debug print
        
#         words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
#         candidates = find_all_title_coordinates(words, titles)
        
#         # Simple disambiguation: take first found
#         final_coords = {t: c[0] for t, c in candidates.items() if c}
        
#         results['page_1'] = final_coords
#         if os.path.exists(temp_img): os.remove(temp_img)
        
#     return results

# # --- ROBUST CROPPER ---
# def crop_sections_from_page(coords_dict, page_image_path, pdf_path, output_dir="cropped_sections"):
#     os.makedirs(output_dir, exist_ok=True)
    
#     page_image = Image.open(page_image_path)
#     img_width, img_height = page_image.size

#     with pdfplumber.open(pdf_path) as pdf:
#         page = pdf.pages[0]
#         pdf_width = page.width
#         pdf_height = page.height

#     scale_x = img_width / pdf_width
#     scale_y = img_height / pdf_height

#     scaled = {}
#     for title, c in coords_dict.items():
#         scaled[title] = {
#             "x1": int(c["x1"] * scale_x), "y1": int(c["y1"] * scale_y),
#             "x2": int(c["x2"] * scale_x), "y2": int(c["y2"] * scale_y),
#         }

#     # Row Grouping
#     sorted_titles = sorted(scaled.keys(), key=lambda t: scaled[t]["y2"])
#     rows = []
#     current_row = []
#     current_row_y_max = -1
#     ROW_THRESHOLD = 50 

#     for title in sorted_titles:
#         y2 = scaled[title]["y2"]
#         if not current_row:
#             current_row.append(title)
#             current_row_y_max = y2
#         else:
#             if abs(y2 - current_row_y_max) < ROW_THRESHOLD:
#                 current_row.append(title)
#                 current_row_y_max = max(current_row_y_max, y2)
#             else:
#                 rows.append(current_row)
#                 current_row = [title]
#                 current_row_y_max = y2
#     if current_row: rows.append(current_row)

#     cropped_sections = []
#     prev_y = 0 

#     for row_idx, titles_in_row in enumerate(rows, start=1):
#         row_y_bottom = max(scaled[t]["y2"] for t in titles_in_row)
        
#         if row_y_bottom <= prev_y: continue 

#         cropped_row = page_image.crop((0, prev_y, img_width, row_y_bottom))
#         titles_in_row_sorted = sorted(titles_in_row, key=lambda t: scaled[t]["x1"])

#         left_margin, right_margin = 400, 200

#         for col_idx, t in enumerate(titles_in_row_sorted):
#             curr_x1 = scaled[t]["x1"]
#             x_start = max(0, curr_x1 - left_margin)

#             if col_idx < len(titles_in_row_sorted) - 1:
#                 next_x1 = scaled[titles_in_row_sorted[col_idx + 1]]["x1"]
#                 x_end = max(0, next_x1 - right_margin)
#             else:
#                 x_end = img_width

#             if x_end <= x_start: continue

#             cropped_section = cropped_row.crop((x_start, 0, x_end, cropped_row.height))
            
#             safe_title = "".join(c for c in t if c.isalnum() or c in (' ', '_')).strip()
#             save_path = os.path.join(output_dir, f"{safe_title}.png")
#             cropped_section.save(save_path)

#             cropped_sections.append({"title": t, "image_path": save_path})
        
#         prev_y = row_y_bottom + 10 

#     return cropped_sections

# # --- 4. AGENT 1: CLASSIFIER NODE ---
# class DrawingTypeResponse(BaseModel):
#     drawing_type: Literal["text", "floor", "section"]

# def node_classify_pages(state: ProjectState):
#     print("--- NODE: Classifying Pages ---")
#     pdf_path = state["pdf_path"]
#     page_map = {}
    
#     with pdfplumber.open(pdf_path) as pdf:
#         total_pages = len(pdf.pages)
        
#     for page_num in range(total_pages):
#         temp_img_path = f"{state['output_dir']}/temp_page_{page_num}.png"
#         convert_specific_page_to_png(pdf_path, page_num, temp_img_path, dpi=150)
        
#         prompt = """
#         Classify this construction sheet into ONE category:
#         1. "text" (Notes, Schedules, Tables)
#         2. "floor" (Plan View, Foundation, Roof Framing)
#         3. "section" (Details, Wall Sections, Cuts)
#         """
#         image_b64 = load_image_base64(temp_img_path)
#         msg = HumanMessage(content=[
#             {"type": "text", "text": prompt},
#             {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
#         ])
        
#         result = llm_flash.with_structured_output(DrawingTypeResponse).invoke([msg])
#         page_map[page_num] = result.drawing_type
#         print(f"Page Index {page_num}: {result.drawing_type}")

#     return {"page_map": page_map}

# # --- 5. AGENT 1: TEXT PROCESSOR ---
# def node_process_text_rules(state: ProjectState):
#     print("--- NODE: Processing Text Rules ---")
#     text_pages = [p for p, t in state["page_map"].items() if t == "text"]
#     accumulated_rules = ""
    
#     for page_num in text_pages:
#         with pdfplumber.open(state["pdf_path"]) as pdf:
#             text = pdf.pages[page_num].extract_text() or ""
            
#         prompt = f"Extract structural rules (Lintel schedules, Bolt spacing) from:\n{text}"
#         msg = HumanMessage(content=prompt)
#         response = llm_flash.invoke([msg])
#         accumulated_rules += f"\nPage {page_num}: {response.content}\n"
        
#     return {"general_rules": accumulated_rules}

# # --- 6. AGENT 3: DETAIL PROCESSOR ---
# def node_process_details(state: ProjectState):
#     print("--- NODE: Processing Details ---")
#     detail_library = state.get("detail_library", {})
#     section_pages = [p for p, t in state["page_map"].items() if t == "section"]
    
#     for page_num in section_pages:
#         print(f"Processing Page {page_num}...")
        
#         page_img_path = f"{state['output_dir']}/section_page_{page_num}.png"
#         page_pdf_path = f"{state['output_dir']}/section_page_{page_num}.pdf"
        
#         convert_specific_page_to_png(state["pdf_path"], page_num, page_img_path, dpi=300)
        
#         try:
#             reader = PdfReader(state["pdf_path"])
#             writer = PdfWriter()
#             writer.add_page(reader.pages[page_num])
#             with open(page_pdf_path, "wb") as f: writer.write(f)
#         except Exception as e:
#             print(f"PDF extraction failed: {e}")
#             continue

#         try:
#             all_coords = find_title_coordinates_from_image_and_pdf(page_pdf_path)
#             coords_dict = all_coords.get('page_1', {})
#         except Exception as e:
#             print(f"Coord extraction failed: {e}")
#             coords_dict = {}

#         if not coords_dict:
#             print(f"No titles matched on page {page_num}. Skipping.")
#             continue

#         cropped_sections = crop_sections_from_page(
#             coords_dict, page_img_path, page_pdf_path, 
#             output_dir=f"{state['output_dir']}/crops_{page_num}"
#         )
        
#         for crop in cropped_sections:
#             prompt = f"""
#             Analyze detail "{crop['title']}". Identify BOM: Steel Profiles, Angles, Plates, Bolts.
#             Return JSON: {{ "symbol": "{crop['title']}", "materials": [...] }}
#             """
#             img_b64 = load_image_base64(crop["image_path"])
#             msg = HumanMessage(content=[
#                 {"type": "text", "text": prompt},
#                 {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
#             ])
#             try:
#                 resp = llm_pro.invoke([msg])
#                 json_str = resp.content.replace("```json", "").replace("```", "")
#                 detail_library[crop['title']] = json.loads(json_str)
#             except: pass

#     return {"detail_library": detail_library}

# # --- 7. AGENT 2: PLAN ESTIMATOR (FIXED 2x4 CROP) ---
# def node_process_plans(state: ProjectState):
#     print("--- NODE: Processing Plans ---")
#     final_estimates = []
#     floor_pages = [p for p, t in state["page_map"].items() if t == "floor"]
    
#     context = json.dumps(state["detail_library"], indent=2)
#     rules = state["general_rules"]
    
#     for page_num in floor_pages:
#         print(f"Estimating Page Index {page_num}...")
#         page_dir = f"{state['output_dir']}/floor_{page_num}"
        
#         # 1. Convert Page to High-Res Image
#         convert_specific_page_to_png(state["pdf_path"], page_num, f"{page_dir}.png", dpi=300)
        
#         # 2. Crop into 8 Parts (2x4 Grid)
#         quad_paths = crop_image_into_2x4_grid(f"{page_dir}.png", page_dir)
        
#         # 3. Add the Original Full Image as well
#         quad_paths.insert(0, f"{page_dir}.png")
        
#         prompt = f"""
#         Act as a Senior Structural Estimator.
        
#         ### CONTEXT: DETAIL LIBRARY (From Section Cuts)
#         {context}
        
#         ### CONTEXT: GENERAL RULES (From Notes)
#         {rules}
        
#         ### TASK
#         Analyze these 9 images (1 Global View + 8 Zoomed Segments).
#         Extract Material Takeoff for: HSS, WF, C, L, FB, ROD.
        
#         ### SPECIFIC LOGIC
#         1. **WF Beams:** Look for labels like "W24x62". Trace the grid lines to find the length.
#         2. **HSS Columns:** Count black squares. Use 18.29' height.
#         3. **Hexagon Tags:** Look for tags like <1>. Use the Shear Wall Schedule rule from the text context.
#         4. **Lintels:** Look for "R.O." dimensions. Use the Lintel Schedule rule (Width + 16").
        
#         Return a JSON list of items found.
#         """
        
#         content = [{"type": "text", "text": prompt}]
#         for q in quad_paths:
#             content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{load_image_base64(q)}"}} )
            
#         msg = HumanMessage(content=content)
#         resp = llm_pro.invoke([msg])
#         try:
#             json_str = resp.content.strip().replace("```json", "").replace("```", "")
#             final_estimates.append(json.loads(json_str))
#         except: pass

#     return {"final_estimates": final_estimates}

# # --- 8. BUILD GRAPH ---
# workflow = StateGraph(ProjectState)
# workflow.add_node("classify", node_classify_pages)
# workflow.add_node("process_text", node_process_text_rules)
# workflow.add_node("process_details", node_process_details)
# workflow.add_node("process_plans", node_process_plans)

# workflow.add_edge(START, "classify")
# workflow.add_edge("classify", "process_text")
# workflow.add_edge("classify", "process_details")
# workflow.add_edge("process_text", "process_plans")
# workflow.add_edge("process_details", "process_plans")
# workflow.add_edge("process_plans", END)

# app = workflow.compile()

# if __name__ == "__main__":
#     os.makedirs("output_temp", exist_ok=True)
    
#     state = {
#         "pdf_path": "langgraph_temp_workflow/input.pdf",
#         "output_dir": "output_temp",
#         "page_map": {}, "detail_library": {}, "general_rules": "", "final_estimates": []
#     }
    
#     print("Starting Estimation Workflow...")
#     result = app.invoke(state)
    
#     print("\n\n=== FINAL ESTIMATION REPORT ===")
#     print(json.dumps(result["final_estimates"], indent=2))




# import os
# import json
# import base64
# import re
# import shutil
# from typing import TypedDict, List, Dict, Any, Literal
# from dotenv import load_dotenv

# # LangChain / LangGraph Imports
# from langgraph.graph import StateGraph, START, END
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.messages import HumanMessage
# from pydantic import BaseModel

# # PDF & Image Processing Imports
# import pdfplumber
# from PIL import Image
# from pypdf import PdfReader, PdfWriter
# from utils.pdf_page_to_png import convert_specific_page_to_png

# # --- YOUR CUSTOM UTILS ---
# from utils.crop_in_quandrant import crop_image_into_quad
# from utils.croped_sections import crop_sections_from_page

# load_dotenv()

# # --- 1. SETUP MODELS ---
# llm_pro = ChatGoogleGenerativeAI(model="gemini-3-pro-preview") 
# llm_flash = ChatGoogleGenerativeAI(model="gemini-2.5-flash") 

# # --- 2. DEFINE STATE ---
# class ProjectState(TypedDict):
#     pdf_path: str
#     output_dir: str
#     page_map: Dict[int, str] 
#     detail_library: Dict[str, Any] 
#     general_rules: str 
#     final_estimates: List[Dict]

# # --- 3. HELPER FUNCTIONS ---

# def load_image_base64(image_path: str) -> str:
#     with open(image_path, "rb") as f:
#         return base64.b64encode(f.read()).decode("utf-8")

# # --- TITLE FINDING LOGIC (FIXED WITH NORMALIZATION) ---

# def normalize_text(text):
#     """Removes punctuation and extra spaces for better matching."""
#     if not text: return ""
#     return re.sub(r'[^a-zA-Z0-9]', '', text).upper()

# def find_title(image_path: str):
#     image_base64 = load_image_base64(image_path)
#     prompt = """
#     Look at this construction sheet.
#     Extract ONLY the bold, uppercase TITLES found below the detail drawings.
#     Examples: "TYP. WALL SECTION", "LADDER DETAIL", "SECTION A-A".
#     Do NOT extract notes, dimensions, or material labels.
#     Return just the titles, one per line.
#     """
#     message = HumanMessage(content=[
#         {"type": "text", "text": prompt},
#         {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
#     ])
#     response = llm_flash.invoke([message])
#     titles = [line.strip() for line in response.content.split("\n") if line.strip()]
#     return titles

# def find_all_title_coordinates(words, titles):
#     results = {} 
    
#     # Pre-normalize PDF text for fuzzy matching
#     # This fixes the issue where PDF has "T Y P .  W A L L" but LLM sees "TYP. WALL"
    
#     for title in titles:
#         norm_title = normalize_text(title)
#         if not norm_title: continue
        
#         title_words = title.split()
#         n = len(title_words)
        
#         candidates = []
        
#         # Sliding window search
#         for i in range(len(words) - n + 1):
#             # Construct string from n words and normalize it
#             segment_text = "".join([w["text"] for w in words[i:i+n]])
#             norm_segment = normalize_text(segment_text)
            
#             # Check if the normalized title is inside the segment
#             if norm_title in norm_segment: 
#                 boxes = words[i:i+n]
#                 candidates.append({
#                     "x1": min(w["x0"] for w in boxes),
#                     "y1": min(w["top"] for w in boxes),
#                     "x2": max(w["x1"] for w in boxes),
#                     "y2": max(w["bottom"] for w in boxes)
#                 })
        
#         if candidates:
#             results[title] = candidates
            
#     return results

# def disambiguate_repeated_titles(image_path, title_coords_candidates):
#     final_coords = {}
#     for title, candidates in title_coords_candidates.items():
#         if len(candidates) == 1:
#             final_coords[title] = candidates[0]
#         else:
#             # Default to first candidate to save tokens/time
#             final_coords[title] = candidates[0] 
#     return final_coords

# def find_title_coordinates_from_image_and_pdf(pdf_path):
#     results = {}
#     with pdfplumber.open(pdf_path) as pdf:
#         page = pdf.pages[0]
#         temp_img = "temp_title_scan.png"
#         page.to_image(resolution=300).save(temp_img)
        
#         titles = find_title(temp_img)
#         print(f"   > AI Found Titles: {titles}") 
        
#         words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
#         candidates = find_all_title_coordinates(words, titles)
#         final_coords = disambiguate_repeated_titles(temp_img, candidates)
        
#         results['page_1'] = final_coords
#         if os.path.exists(temp_img): os.remove(temp_img)
        
#     return results

# # --- 4. AGENT 1: CLASSIFIER NODE ---
# class DrawingTypeResponse(BaseModel):
#     drawing_type: Literal["text", "floor", "section"]

# def node_classify_pages(state: ProjectState):
#     print("--- NODE: Classifying Pages ---")
#     pdf_path = state["pdf_path"]
#     page_map = {}
    
#     with pdfplumber.open(pdf_path) as pdf:
#         total_pages = len(pdf.pages)
        
#     for page_num in range(total_pages):
#         temp_img_path = f"{state['output_dir']}/temp_page_{page_num}.png"
#         convert_specific_page_to_png(pdf_path, page_num, temp_img_path, dpi=150)
        
#         prompt = """
#         Classify this construction sheet into ONE category:
#         1. "text" (Notes, Schedules, Tables)
#         2. "floor" (Plan View, Foundation, Roof Framing)
#         3. "section" (Details, Wall Sections, Cuts)
#         """
#         image_b64 = load_image_base64(temp_img_path)
#         msg = HumanMessage(content=[
#             {"type": "text", "text": prompt},
#             {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
#         ])
        
#         result = llm_flash.with_structured_output(DrawingTypeResponse).invoke([msg])
#         page_map[page_num] = result.drawing_type
#         print(f"Page Index {page_num}: {result.drawing_type}")

#     return {"page_map": page_map}

# # --- 5. AGENT 1: TEXT PROCESSOR ---
# def node_process_text_rules(state: ProjectState):
#     print("--- NODE: Processing Text Rules ---")
#     text_pages = [p for p, t in state["page_map"].items() if t == "text"]
#     accumulated_rules = ""
    
#     for page_num in text_pages:
#         with pdfplumber.open(state["pdf_path"]) as pdf:
#             text = pdf.pages[page_num].extract_text() or ""
            
#         prompt = f"Extract structural rules (Lintel schedules, Bolt spacing) from:\n{text}"
#         msg = HumanMessage(content=prompt)
#         response = llm_flash.invoke([msg])
#         accumulated_rules += f"\nPage {page_num}: {response.content}\n"
        
#     return {"general_rules": accumulated_rules}

# # --- 6. AGENT 3: DETAIL PROCESSOR ---
# def node_process_details(state: ProjectState):
#     print("--- NODE: Processing Details ---")
#     detail_library = state.get("detail_library", {})
#     section_pages = [p for p, t in state["page_map"].items() if t == "section"]
    
#     for page_num in section_pages:
#         print(f"Processing Page {page_num}...")
        
#         page_img_path = f"{state['output_dir']}/section_page_{page_num}.png"
#         page_pdf_path = f"{state['output_dir']}/section_page_{page_num}.pdf"
        
#         # 1. Convert to Image
#         convert_specific_page_to_png(state["pdf_path"], page_num, page_img_path, dpi=300)
        
#         # 2. Extract Single Page PDF
#         try:
#             reader = PdfReader(state["pdf_path"])
#             writer = PdfWriter()
#             writer.add_page(reader.pages[page_num])
#             with open(page_pdf_path, "wb") as f: writer.write(f)
#         except Exception as e:
#             print(f"PDF extraction failed: {e}")
#             continue

#         # 3. Get Coords (Using Normalized Logic)
#         try:
#             all_coords = find_title_coordinates_from_image_and_pdf(page_pdf_path)
#             coords_dict = all_coords.get('page_1', {})
#         except Exception as e:
#             print(f"Coord extraction failed: {e}")
#             coords_dict = {}

#         if not coords_dict:
#             print(f"No titles matched on page {page_num}. Skipping.")
#             continue

#         # 4. Crop (Using YOUR Utility)
#         # Note: Your utility handles scaling internally if passed raw PDF coords
#         try:
#             cropped_sections = crop_sections_from_page(
#                 coords_dict, 
#                 page_img_path, 
#                 page_pdf_path, 
#                 f"page_{page_num}",
#                 base_output_dir=state['output_dir']
#             )
#         except Exception as e:
#             print(f"Cropping failed: {e}")
#             continue
        
#         # 5. Analyze
#         for crop in cropped_sections:
#             title = crop['title']
#             prompt = f"""
#             Analyze detail "{title}". Identify BOM: Steel Profiles, Angles, Plates, Bolts.
#             Return JSON: {{ "symbol": "{title}", "materials": [...] }}
#             """
#             img_b64 = load_image_base64(crop["image_path"])
#             msg = HumanMessage(content=[
#                 {"type": "text", "text": prompt},
#                 {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
#             ])
#             try:
#                 resp = llm_pro.invoke([msg])
#                 json_str = resp.content.replace("```json", "").replace("```", "")
#                 detail_library[title] = json.loads(json_str)
#             except: pass

#     return {"detail_library": detail_library}

# # --- 7. AGENT 2: PLAN ESTIMATOR ---
# def node_process_plans(state: ProjectState):
#     print("--- NODE: Processing Plans ---")
#     final_estimates = []
#     floor_pages = [p for p, t in state["page_map"].items() if t == "floor"]
    
#     context = json.dumps(state["detail_library"], indent=2)
#     rules = state["general_rules"]
    
#     for page_num in floor_pages:
#         print(f"Estimating Page Index {page_num}...")
#         page_dir = f"{state['output_dir']}/floor_{page_num}"
        
#         # 1. Convert Page
#         convert_specific_page_to_png(state["pdf_path"], page_num, f"{page_dir}.png", dpi=300)
        
#         # 2. Crop (Using YOUR Utility)
#         crop_image_into_quad(f"{page_dir}.png", page_dir)
        
#         # 3. Load Quadrants
#         quads = []
#         if os.path.exists(page_dir):
#             quads = [os.path.join(page_dir, f) for f in os.listdir(page_dir) if f.endswith(".png")]
        
#         # Add original full image
#         quads.insert(0, f"{page_dir}.png")
        
#         prompt = f"""
#         Act as a Senior Structural Estimator.
        
#         ### CONTEXT: DETAIL LIBRARY (From Section Cuts)
#         {context}
        
#         ### CONTEXT: GENERAL RULES (From Notes)
#         {rules}
        
#         ### TASK
#         Analyze these images (1 Global View + Zoomed Segments).
#         Extract Material Takeoff for: HSS, WF, C, L, FB, ROD.
        
#         ### SPECIFIC LOGIC
#         1. **WF Beams:** Look for labels like "W24x62". Trace the grid lines to find the length.
#         2. **HSS Columns:** Count black squares. Use 18.29' height.
#         3. **Hexagon Tags:** Look for tags like <1>. Use the Shear Wall Schedule rule from the text context.
#         4. **Lintels:** Look for "R.O." dimensions. Use the Lintel Schedule rule (Width + 16").
        
#         Return a JSON list of items found.
#         """
        
#         content = [{"type": "text", "text": prompt}]
#         for q in quads:
#             content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{load_image_base64(q)}"}} )
            
#         msg = HumanMessage(content=content)
#         resp = llm_pro.invoke([msg])
#         try:
#             json_str = resp.content.strip().replace("```json", "").replace("```", "")
#             final_estimates.append(json.loads(json_str))
#         except: pass

#     return {"final_estimates": final_estimates}

# # --- 8. BUILD GRAPH ---
# workflow = StateGraph(ProjectState)
# workflow.add_node("classify", node_classify_pages)
# workflow.add_node("process_text", node_process_text_rules)
# workflow.add_node("process_details", node_process_details)
# workflow.add_node("process_plans", node_process_plans)

# workflow.add_edge(START, "classify")
# workflow.add_edge("classify", "process_text")
# workflow.add_edge("classify", "process_details")
# workflow.add_edge("process_text", "process_plans")
# workflow.add_edge("process_details", "process_plans")
# workflow.add_edge("process_plans", END)

# app = workflow.compile()

# if __name__ == "__main__":
#     os.makedirs("output_temp", exist_ok=True)
    
#     state = {
#         "pdf_path": "langgraph_temp_workflow/input.pdf",
#         "output_dir": "output_temp",
#         "page_map": {}, "detail_library": {}, "general_rules": "", "final_estimates": []
#     }
    
#     print("Starting Estimation Workflow...")
#     result = app.invoke(state)
    
#     print("\n\n=== FINAL ESTIMATION REPORT ===")
#     print(json.dumps(result["final_estimates"], indent=2))
