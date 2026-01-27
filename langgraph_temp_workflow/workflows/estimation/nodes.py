import os
import json
from dotenv import load_dotenv
# LangChain / LangGraph Imports

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


# import state from the common/state file 
from langgraph_temp_workflow.common.state import ProjectState

# import schema from the common/schema file 
from langgraph_temp_workflow.common.schemas import DrawingTypeResponse 
from langgraph_temp_workflow.common.schemas import DetailExtraction 
from langgraph_temp_workflow.common.schemas import PlanExtraction 
from langgraph_temp_workflow.common.schemas import FinalEstimation 

# PDF & Image Processing Imports
import pdfplumber
from pypdf import PdfReader, PdfWriter
from utils.pdf_page_to_png import convert_specific_page_to_png

# --- CUSTOM UTILS ---
from utils.croped_sections import crop_sections_from_page
from utils.sementic_segmentation import semantic_segmentation_app
from langgraph_temp_workflow.common.utils import load_image_base64
from langgraph_temp_workflow.common.utils import extract_text_from_response
from langgraph_temp_workflow.common.utils import get_sheet_number
from langgraph_temp_workflow.common.utils import find_title_coordinates_from_image_and_pdf
from langgraph_temp_workflow.common.utils import preprocess_image_inplace

load_dotenv()


# --- 1. SETUP MODELS ---
llm_pro = ChatGoogleGenerativeAI(model="gemini-3-pro-preview") 
llm_flash = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite") 


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
        print(f"Page Index {page_num}: {result.drawing_type}")

    return {"page_map": page_map}

# --- 5. AGENT 1: TEXT PROCESSOR ---
def node_process_text_rules(state: ProjectState):
    print("--- NODE: Processing Text Rules ---")
    text_pages = [p for p, t in state["page_map"].items() if t == "text"]
    accumulated_rules = ""
    
    for page_num in text_pages:
        with pdfplumber.open(state["pdf_path"]) as pdf:
            text = pdf.pages[page_num].extract_text() or ""
        
        # Text processing doesn't strictly need Pydantic as we just want a summary string
        prompt = f"Extract structural rules (Lintel schedules, Bolt spacing) from:\n{text}"
        msg = HumanMessage(content=prompt)
        response = llm_flash.invoke([msg])
        accumulated_rules += f"\nPage {page_num}: {extract_text_from_response(response)}\n"
        
    return {"general_rules": accumulated_rules}

# --- 6. AGENT 3: DETAIL PROCESSOR ---
def node_process_details(state: ProjectState):
    print("--- NODE: Processing Details ---")
    detail_library = state.get("detail_library", {})
    section_pages = [p for p, t in state["page_map"].items() if t == "section"]
    
    for page_num in section_pages:
        print(f"Processing Page {page_num}...")
        
        page_img_path = f"{state['output_dir']}/section_page_{page_num}.png"
        page_pdf_path = f"{state['output_dir']}/section_page_{page_num}.pdf"
        
        convert_specific_page_to_png(state["pdf_path"], page_num, page_img_path, dpi=300)
        
        # --- STEP 1: EXTRACT SHEET NUMBER (e.g., S-3.2) ---
        sheet_number = get_sheet_number(page_img_path)
        print(f"   > Identified Sheet Number: {sheet_number}")

        # --- STEP 2: EXTRACT SINGLE PAGE PDF ---
        try:
            reader = PdfReader(state["pdf_path"])
            writer = PdfWriter()
            writer.add_page(reader.pages[page_num])
            with open(page_pdf_path, "wb") as f: writer.write(f)
        except Exception as e:
            print(f"PDF extraction failed: {e}")
            continue

        # --- STEP 3: GET COORDS ---
        try:
            all_coords = find_title_coordinates_from_image_and_pdf(page_pdf_path)
            coords_dict = all_coords.get('page_1', {})
        except Exception as e:
            print(f"Coord extraction failed: {e}")
            coords_dict = {}

        if not coords_dict:
            print(f"No titles matched on page {page_num}. Skipping.")
            continue

        # --- STEP 4: CROP SECTIONS ---
        try:
            cropped_sections = crop_sections_from_page(
                coords_dict, 
                page_img_path, 
                page_pdf_path, 
                f"page_{page_num}",
                base_output_dir=state['output_dir']
            )
        except Exception as e:
            print(f"Cropping failed: {e}")
            continue
        
        # --- STEP 5: ANALYZE CROPS & BUILD KEYS ---
        for crop in cropped_sections:
            title = crop['title']
            
            # Updated Prompt to find the Detail Number (Circle/Bubble)
            prompt = """
You are a Senior Structural Detailer. You are analyzing a high-resolution crop of a specific construction detail drawing.

### YOUR GOAL
Extract the **Bill of Materials (BOM)** and the **Detail Identification Number** so that the Estimator can link this detail to the Floor Plan symbols.

### STEP 1: IDENTIFY THE DETAIL (The Link Key)
Look for the "Callout Bubble" associated with this detail title "{title}".
*   **Detail Number:** Extract the number inside the circle/bubble (e.g., "1", "7", "A").
*   **Sheet Reference:** If a sheet number is written below the line (e.g., "S-3.2"), extract it.
*   *Note:* If you cannot find a number, return `null`.

### STEP 2: EXTRACT BILL OF MATERIALS (BOM)
Read every text note and leader line in the image. Identify structural steel components.
*   **Steel Profiles:** Look for HSS, W-Shapes, Channels (C/MC), Tubes.
    *   *Example:* "HSS 5x5x5/16", "MC6x15.1".
*   **Connection Hardware:** Look for Plates (PL), Angles (L), Rods, Bolts.
    *   *Example:* "PL 1/2"x6"x12"", "L4x4x3/8", "3/4" Anchor Bolt".
*   **Welds:** Look for weld symbols (flags/triangles). Note the size (e.g., "3/16") and type (e.g., "Fillet All Around").

### STEP 3: DETERMINE QUANTITY RULES
For each item found, determine if the quantity is "Fixed" (per detail) or "Variable" (per length).
*   **Fixed:** "2 Anchor Bolts", "1 Base Plate". (Count is explicit in the detail).
*   **Variable:** "Continuous Angle", "Rails". (Length depends on the height/width of the object it attaches to).

### OUTPUT FORMAT (Pydantic Schema)
Return a structured object matching this schema:
{{
  "detail_number": "The number found in the bubble (or null)",
  "title": "{title}",
  "materials": [
    {{
      "item_name": "Exact text from drawing (e.g., MC6x15.1)",
      "qty_rule": "FIXED (e.g., 2 pieces) or VARIABLE (e.g., Height of Ladder)",
      "notes": "Context (e.g., Side Rails for Ladder)"
    }},
    {{
      "item_name": "L4x4x1/4",
      "qty_rule": "FIXED (2 clips)",
      "notes": "Base connection to concrete"
    }}
  ]
}}
"""
            
            img_b64 = load_image_base64(crop["image_path"])
            msg = HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            ])
            
            try:
                result = llm_pro.with_structured_output(DetailExtraction).invoke([msg])
                # --- CONSTRUCT THE KEY (e.g., 7/S-3.2) ---
                det_num = result.detail_number
                
                if det_num and str(det_num).lower() != "null":
                    # Clean the number (remove dots/spaces)
                    det_num = str(det_num).strip().replace(".", "")
                    key = f"{det_num}/{sheet_number}"
                else:
                    # Fallback if no number found: Use Title/Sheet
                    key = f"{title}/{sheet_number}"
                
                print(f"   > Stored Detail: {key}")
                detail_library[key] = result.model_dump()
                
            except Exception as e:
                print(f"Failed to parse JSON for {title}: {e}")
        
    print('Here how the all the information about the section parts are stored :- ',detail_library)
    
    return {"detail_library": detail_library}

# --- 7. AGENT 2: PLAN ESTIMATOR (RAW DATA COLLECTOR) ---
def node_process_plans(state: ProjectState):
    print("--- NODE: Agent 2 (Plan Scanner) ---")
    raw_plan_data = []
    floor_pages = [p for p, t in state["page_map"].items() if t == "floor"]
    
    for page_num in floor_pages:
        print(f"Scanning Page Index {page_num}...")
        
        page_dir = f"{state['output_dir']}/floor_{page_num}"
        page_img_path = f"{state['output_dir']}/floor_{page_num}.png"
        semantic_crops_dir = f"{state['output_dir']}/floor_{page_num}"
        convert_specific_page_to_png(state["pdf_path"], page_num, page_img_path, dpi=300)
        
        # convert_specific_page_to_png(state["pdf_path"], page_num, f"{page_dir}.png", dpi=300)
        
        # Crop in quad function call 
        # crop_image_into_quad(f"{page_dir}.png", page_dir)

 
        #calling the langraph agent to get the sementic croping of the same plan
        #it save the croped image in the folder name as file_name
        
        child_initial_state = {
            "image_path": page_img_path, 
            "detected_queue": [], 
            "final_crops": [], 
            "current_retry_count": 0,
            "current_region_label": None, 
            "current_bbox": None,
            "output_dir": state["output_dir"], # Pass the base output dir
            "extracted_data": {}
        }
        # call to the sementic segementation code to get the focused image 
        semantic_segmentation_app.invoke(child_initial_state, config={"recursion_limit": 150})

        # it will stores the sementic segemntation image in the folder named as floor/roof
        sementic_croped_images = [] 
        plan_croped_image:str
        
        if os.path.exists(semantic_crops_dir):
            for f in os.listdir(semantic_crops_dir):
                if f.endswith(".png"):
                    img_path = os.path.join(semantic_crops_dir, f)

                    success = preprocess_image_inplace(img_path)
                    if success:
                        sementic_croped_images.append(img_path)
             
            print(f" > Found {len(sementic_croped_images)-1} semantic crops (processed in-place).")
        else:
            print(f" ! Warning: No semantic crops found at {semantic_crops_dir}")
        
        # if os.path.exists(semantic_crops_dir):
        #     for f in os.listdir(semantic_crops_dir):
        #         if f.endswith(".png"):
        #             #here we will do ml/dl operation over image to convert into grey scale or Adaptive threshold
        #             sementic_croped_images.append(os.path.join(semantic_crops_dir, f))

        #     print(f"  > Found {len(sementic_croped_images)-1} semantic crops.")
        # else:
        #     print(f"  ! Warning: No semantic crops found at {semantic_crops_dir}")
        
        # --- ADVANCED PROMPT FOR AGENT 2 ---
        prompt = """
        You are a Forensic Structural Surveyor. You are provided with 9 images of a single construction plan:
        1. One "Global View" (The whole page).
        2. Eight "Zoomed Quadrants" (High-resolution segments).

        ### YOUR GOAL
        Extract every visible structural annotation, symbol, and dimension. Do NOT calculate totals. Do NOT interpret logic. Just report what exists and WHERE it is (Grid Coordinates).

        ### SECTION 1: STRUCTURAL MEMBERS (Look for Text Labels)
        Scan for these specific patterns and record their Grid Location (e.g., "Intersection of B and 2" or "Between Grid 1 and 2"):
        *   **HSS Columns:** Look for black squares or labels like "HSS5x5x5/16", "F5", "F4".
        *   **WF Beams:** Look for labels like "W24x62", "W14x22". Note the start and end grids.
        *   **Channels (C/MC):** Look for "MC6x15.1" or "C-Channel" notes.
        *   **Angles (L):** Look for "L4x4" or "Bent Plate" notes.

        ### SECTION 2: SYMBOLS & CALLOUTS (The "Pointers")
        Identify every bubble, hexagon, or tag that points to a detail.
        *   **Detail Cuts:** Look for circles/triangles with text like "7/S-3.2", "3/S-4.1".
        *   **Shear Wall Tags:** Look for Hexagons with numbers (e.g., <1>, <5>).
        *   **Window Tags:** Look for Hexagons on the perimeter (e.g., <6>, <4>).
        *   **Holdowns:** Look for small triangles inside walls.

        ### SECTION 3: GEOMETRY & DIMENSIONS (Crucial for Math)
        For every Symbol found in Section 2, find the **Dimension Text** associated with it.
        *   *Example:* If you see Hexagon <1>, look for text like "13'-10\" SW" or "10'-4\"".
        *   *Example:* If you see a Window Tag <6>, look for "R.O." dimensions (e.g., "4'-0\" R.O.").
        *   *Example:* If you see a Beam, look for the span dimension (e.g., "29'-6\"").
        *   *Example:* Look for Elevation Notes (e.g., "T/BEAM = 18'-3 1/2\"").

        ### SECTION 4: SCHEDULES (Data Tables)
        If the sheet contains a table (e.g., "Shear Wall Schedule", "Footing Schedule", "Lintel Schedule"), transcribe the rows exactly.

        ### OUTPUT FORMAT (Strict JSON)
        {
          "sheet_type": "Floor Plan / Roof Plan",
          "members": [
            {"label": "HSS5x5x5/16", "type": "F5", "location": "Grid B-2", "count": 1},
            {"label": "W24x62", "location": "Grid B to C", "length_text": "27'-0\""}
          ],
          "symbols": [
            {"symbol": "7/S-3.2", "location": "Near Grid F-3", "associated_text": "4'-0\""},
            {"symbol": "Hexagon 1", "location": "Grid A-1", "associated_text": "13'-10\" SW"},
            {"symbol": "Hexagon 6", "location": "Grid B Wall", "associated_text": "4'-0\" R.O."}
          ],
          "global_notes": ["T/BEAM = 18'-3 1/2\""],
          "schedules": [
            {"name": "Shear Wall Schedule", "data": "Row 1: 5/8 inch bolt @ 16 oc..."}
          ]
        }
        """
        
        content = [{"type": "text", "text": prompt}]
        for q in sementic_croped_images:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{load_image_base64(q)}"}} )
            
        msg = HumanMessage(content=content)
        
        try:
            result = llm_pro.with_structured_output(PlanExtraction).invoke([msg])
            raw_plan_data.append(result.model_dump())
        except Exception as e:
            print(f"ERROR PARSING JSON for Page {page_num}: {e}")
    print(f"\n\n\n")
    print(f"this how look like when we parse the fllor plana dn roof plan with given section details information in the dictionary format raw_plan_data  {raw_plan_data}")
    print(f"\n\n\n")
    return {"raw_plan_data": raw_plan_data}

# --- 8. AGENT 4: THE LOGIC MERGER ---
def node_agent_4_merger(state: ProjectState):
    print("--- NODE: Agent 4 (The Merger) ---")
    
    # Construct the Prompt
    prompt = f"""
    You are the Senior Structural Estimator. You have received raw data from your field surveyors (Agent 1, 2, and 3).
    Your job is to **INTERLACE** this information to produce the Final Material Takeoff.

    ### INPUT DATA
    1. **RAW PLAN DATA (From Agent 2):** 
    {json.dumps(state['raw_plan_data'], indent=2)}
    
    2. **DETAIL LIBRARY (BOMs from Agent 3):** 
    {json.dumps(state['detail_library'], indent=2)}
    
    3. **TEXT RULES (From Agent 1):** 
    {state['general_rules']}

    ### YOUR EXECUTION PLAN (Step-by-Step Logic)

    **STEP 1: RESOLVE HSS COLUMNS (Cross-Reference Floor & Roof)**
    *   Take HSS counts from Floor Plan Data.
    *   Find the "Top of Steel" elevation note in Roof Plan Data (e.g., "T/BEAM = 18'-3 1/2\"").
    *   *Calculation:* Count * (Roof Elevation - 0).
    *   *Goal:* Total Linear Feet of HSS.

    **STEP 2: RESOLVE WF BEAMS (Trace Grids)**
    *   Take WF labels from Roof Plan Data.
    *   Use the "length_text" found by Agent 2 (e.g., "27'-0\"").
    *   *Goal:* Total Linear Feet of WF.

    **STEP 3: RESOLVE SYMBOLS (The "Lookup" Logic)**
    *   **Detail Callouts:** Look at symbols like "7/S-3.2" found on the Floor Plan.
        *   *Action:* Look up "7/S-3.2" in the **DETAIL LIBRARY**.
        *   *Action:* If Library says "Ladder", and Floor Plan says "Qty: 1", add 1 Ladder to the list.
        *   *Action:* If Library says "MC6x15.1 Rails", add those channels to the C-Channel list.
    *   **Shear Walls:** Look at "Hexagon 1" found on Floor Plan.
        *   *Action:* Look up "Hexagon 1" in the **TEXT RULES** (Shear Wall Schedule).
        *   *Rule Found:* "5/8 inch Bolt @ 16 inch spacing".
        *   *Dimension Found:* Agent 2 saw "13'-10\"" next to the hexagon.
        *   *Math:* (13.83 ft * 12) / 16 inches = 10.3 -> Round up -> 11 Bolts.
        *   *Goal:* Add 11 Bolts to the ROD list.

    **STEP 4: RESOLVE LINTELS (The "R.O." Logic)**
    *   Look at "Window Tag <6>" found on Floor Plan.
    *   *Dimension Found:* Agent 2 saw "4'-0\" R.O.".
    *   *Rule Lookup:* Check Lintel Schedule in **TEXT RULES**. "If width < 5ft, use L4x4x3/8".
    *   *Math:* 4.0 ft + 1.33 ft (bearing) = 5.33 ft.
    *   *Goal:* Add 5.33 ft of L4x4x3/8 to the L-Angle list.

    ### FINAL OUTPUT GOALS
    Produce a JSON summary satisfying these 6 aims:
    1. **Material Takeoff:** List every HSS, WF, C, L, FB, ROD found.
    2. **Total Weight:** Sum of (Length * Lbs/ft).
    3. **Member Count:** Total count of beams, columns, and assemblies.
    4. **Weld Inches:** Sum of welds defined in the Detail Library.
    5. **Hole Count:** Sum of holes defined in connections and base plates.
    6. **Bolt Count:** Sum of Anchor Bolts (from Shear Walls/Columns) + Structural Bolts.

    ### OUTPUT JSON STRUCTURE
    {{
      "final_bill_of_materials": [
        {{
          "description": "HSS 5x5x5/16",
          "total_qty": 5,
          "total_linear_feet": 91.45,
          "logic_trace": "Found 5 on Floor Plan. Applied 18.29' height from Roof Plan."
        }},
        {{
          "description": "Simpson Titen HD 5/8",
          "total_qty": 45,
          "logic_trace": "Calculated from 4 Shear Walls (Hexagon 1) using lengths 13'-10\", 10'-4\", etc."
        }}
      ]
    }}
    """
    
    msg = HumanMessage(content=prompt)
    
    try:
        result = llm_pro.with_structured_output(FinalEstimation).invoke([msg])
        return {"final_bill_of_materials": result.model_dump()}
    except Exception as e:
        return {"final_bill_of_materials": {"error": str(e)}}
