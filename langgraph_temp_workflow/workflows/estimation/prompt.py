from langgraph_temp_workflow.common.state import ProjectState
import json

def prompt_for_node_classify_pages():
    prompt = """
        Analyze this construction sheet and classify it into exactly ONE of these categories:

        - "text": If the page contains mostly Notes, Schedules, Tables, or Specifications.
        - "floor": If the page shows a Plan View, Foundation Plan, or Roof Framing Plan.
        - "section": If the page shows Detail Drawings, Wall Sections, or Connection Cuts.

        Do NOT return a number. Return the category name.
        """
    return prompt

def prompt_for_node_process_details(title):
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
    return prompt

def prompt_for_node_process_plans():

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
    
    return prompt


def node_agent_4_merger(state: ProjectState):

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
    return prompt


    

