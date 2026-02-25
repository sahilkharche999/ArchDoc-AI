
def prompt_for_detect_regions_node():
    prompt = """
    You are a Senior Architectural Technologist.
    Analyze this sheet to identify **Logical Content Blocks**.

    A "Logical Block" includes:
    1. The graphical content (The drawing, detail, or table).
    2. **The Metadata**: The TITLE text, SCALE, and DETAIL NUMBER usually located BELOW the drawing.
    3. **The Gutter**: The empty whitespace immediately surrounding the content that separates it from neighbors.

    Goal: Return bounding boxes that loosely encapsulate these full blocks. 
    It is better to include slightly too much whitespace than to cut off a title.
    """
    return prompt

def prompt_for_evaluate_crop_node(label:str,retry_count:int):
    prompt = f"""
    Context: We are cropping the region "{label}" (Attempt #{retry_count + 1}).
    
    Image 1: Full Sheet (Global Context).
    Image 2: Current Crop (Local View).

    TASK: Determine if Image 2 is a "Complete Semantic Unit".
    
    A Complete Unit MUST include ALL of the following:
    1. **The Main Content:** The primary Drawing, Table, or Detail itself.
    2. **The Title Block:** Look DOWN. Is the text describing this drawing (e.g., "PLAN VIEW", "SCALE 1:50", "LOOSE LINTEL SCHEDULE") fully visible?
    3. **No Cutoffs:** Are lines leaving the frame abruptly? Does a wall or table row get sliced in half?
    4. **Floating Annotations (CRITICAL):** Look at the white space around the main content.
       - Are there **Grid Bubbles** (Circles with numbers/letters) floating on the left/top?
       - Are there **Section Cuts** (Hexagons/Circles with arrows) floating on the right/bottom?
       - Are there **Dimension Strings** (Lines with numbers) floating outside the walls?
       - **RULE:** If these elements are visually associated with this drawing in Image 1, they MUST be included in Image 2.

    INSTRUCTIONS:
    - If the Title is cut off at the bottom: You MUST request `expand_bottom`.
    - If dimensions, grid bubbles, or floating symbols (like Hexagons) are cut off on the sides: Request `expand_left` or `expand_right`.
    - If the crop is too tight (touching the ink): Request a small expansion (0.05).
    - If it's perfect (includes content + title + all floating elements + small margin): Status = "approved".
    
    BE BOLD: If content is missing, expand by at least 0.05 or 0.10. Do not do tiny adjustments.
    """
    return prompt

def prompt_for_extract_content_node(label:str):
    prompt = f"""
            Analyze this image crop titled: **"{label}"**.
            
            ### TASK:
            Extract the content belonging to "{label}" into a structured format.
            Ignore neighboring text/drawings.
            
            ### RULES:
            1. If it is a **TABLE/SCHEDULE**: 
               - Use the first column (e.g., Mark/Symbol) as the Key.
               - The rest of the row data is the Value.
            2. If it is **NOTES**:
               - Use the Note Number (1, 2, 3) as the Key.
            3. If it is a **DETAIL/LEGEND**:
               - Use the Item Name as Key and Description/Specs as Value.
            """
    return prompt

