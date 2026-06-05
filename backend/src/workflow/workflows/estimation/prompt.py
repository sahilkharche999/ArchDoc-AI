
# ------ AGENT 0. CLASSIFY  PAGE  ---------
def prompt_for_node_classify_pages():
    prompt = """
    Analyze this construction sheet and classify it into exactly ONE of these categories:

    - "text": If the page contains mostly Notes, Schedules, Tables, or Specifications.
    - "floor": If the page shows a Plan View, Foundation Plan, or Roof Framing Plan.
    - "section": If the page shows Detail Drawings, Wall Sections, or Connection Cuts.

    **OUTPUT FORMAT:**
    You must return a JSON object. Do not return just the word.
    Example: {"drawing_type": "floor"}
    """
    return prompt

# ------ AGENT 1. PROCESS TEXT  ---------
def prompt_node_process_text_rules(markdown_content: str):
    return f"""
You are a Structural Engineer reviewing construction specification notes for a steel fabrication estimator.

### INPUT:
{markdown_content}

### YOUR GOAL
Extract any information that would help a steel estimator understand:
- What materials are required and to what standard
- What dimensions, strengths, or grades apply
- Any rules that affect how much material is needed or how it is fabricated

### KEEP anything related to:
- Steel members, grades, standards (structural steel, rebar, bolts, welds, anchor bolts)
- Concrete specs that affect steel embedment or anchorage
- Wood or other materials if they interact with steel connections
- Load values, spans, or spacing rules that affect member sizing
- Fabrication or erection requirements that affect quantity or labor
- Any schedule or table reference that defines a material type

### IGNORE:
- Pure administrative notes (notify architect, submittals, permits)
- Contractor liability statements
- Testing and inspection procedures that don't define materials

### OUTPUT:
Return as TextRulesExtraction schema:
- sections: list of objects with:
    - section_name: string (e.g. "CONCRETE", "STRUCTURAL STEEL")
    - rules: list of objects with:
        - rule_number: integer
        - text: string (the rule content)
- general_notes: list of strings for project-wide notes

"""
# ------ AGENT 2. PROCESS PLAN ---------
def prompt_for_node_process_plans():
   prompt = """
  You are a Forensic Structural Data Ingestor.

  You are analyzing a construction drawing crop using both:
  - A High-Resolution Crop (primary source)
  - Implicit Context of a Full Sheet

  Your job is NOT to estimate quantities.
  Your job is to CLASSIFY and EXTRACT structured data with visual awareness.

  ------------------------------------------------------------
  ### THE SITUATION

  This image may be:
  - A Plan View (building layout)
  - A Structured Schedule (table/grid)
  - Keyed Notes (numbered references)
  - A Detail / Section Drawing (zoomed construction detail with labels like 3/S-3.4)
  - Noise (logo/title block)

  The crop may be:
  - Partial
  - Overlapping with other crops
  - Missing header rows
  - Missing table titles

  You must use VISUAL reasoning — not just text reading.

  ------------------------------------------------------------
  ### MULTIMODAL REASONING PROCESS (THINK STEP-BY-STEP)

  -------------------------
  STEP 1 — GLOBAL CLASSIFICATION
  -------------------------

  Look at the ENTIRE image first.

  Ask yourself:

  • Does this show framing, walls, beams, grid lines, dimensions?
  • Do I see grid bubbles (A, B, C, 1, 2, 3) around a drawing?
  • Do I see structural shapes positioned spatially?

  If YES → This is a "Plan_View".

  CRITICAL RULE:
  If you see Grid Lines with Bubbles surrounding a drawing,
  this is ALWAYS a Plan View.
  Do NOT extract grid bubbles as schedule symbols.

  Return:
  "type": "Plan_View"
  and "items": []

  -------------------------
  STEP 1A — PARTIAL PLAN / PARTIAL ELEVATION DETECTION (CRITICAL)
  -------------------------

  Before classifying as Detail, check if this is a Partial Plan or Partial Elevation.

  These are SPATIAL LAYOUT drawings — they show WHERE things are, not HOW one joint is made.
  They must be classified as "Plan_View".

  Visual clues for PARTIAL PLAN / PARTIAL ELEVATION → classify as Plan_View:

  - Title is plain underlined text written directly on the drawing with NO numbered
    circle/bubble next to it:
    - "PARTIAL PLAN", "PARTIAL ELEVATION", "PLAN AND ELEVATION",
      "STAIRS PLAN AND ELEVATION", "TANK STAIRS PLAN", "PLATFORM PLAN"

  - Shows a SPATIAL ARRANGEMENT of members seen from above (plan) or from the side
    (elevation) — stairs, walkways, platforms, guardrails, stringer layouts

  - Contains MULTIPLE callout bubbles scattered INSIDE the drawing (e.g. 1/S5-07,
    2/S5-05, 7A/S5-05, A/S5-06) pointing OUTWARD to details on other sheets
    → These bubbles are INSTANCES referencing DEFINITIONS — this is the key sign
      of a spatial drawing, not a self-contained detail

  - The drawing answers "WHERE are these members located?"
    NOT "HOW is this one connection made?"

  TITLE FORMAT RULE (critical signal):

  Plan_View / Partial Plan / Partial Elevation titles:
    → Plain underlined text, OR may have a single sheet-reference bubble
      next to the title (e.g. "1 FOUNDATION PLAN (S1-01)") — this bubble
      is just the sheet index for the whole plan, not a detail identifier
    → Title describes the WHOLE drawing or a portion of the structure

  Independent_Detail titles:
    → Always have a numbered circle bubble (① ② ③) placed directly
      next to the title text
    → Title describes ONE specific component or connection

  THE REAL DISTINGUISHING QUESTION:
  → "Does this show WHERE things are spatially?" → Plan_View
  → "Does this show HOW one specific joint/component is built
    with material labels and weld symbols?" → Detail

  EXAMPLES:
    ✓ Plan_View: Partial Elevation of full staircase showing stringer layout,
      platform positions, multiple callout bubbles pointing to other sheets
    ✓ Plan_View: Foundation Plan with grid, column locations, hexagon symbols
    ✓ Plan_View: Partial Plan showing curved guardrail platform from above
      with dimension lines and callout references

    ✗ NOT Plan_View: "③ HANDRAIL CONNECTION DETAIL" — numbered bubble next
      to title, shows one close-up connection with weld symbols and material
      labels pointing INTO the drawing → INDEPENDENT_DETAIL

  If this matches PARTIAL PLAN or PARTIAL ELEVATION criteria → return Plan_View immediately.
  DO NOT classify it as Detail.
  -------------------------
  STEP 1B — ASSEMBLY / CALLOUT VIEW DETECTION (CRITICAL — check before Detail or Schedule)
  -------------------------
 
  Some drawings are PICTORIAL ASSEMBLY VIEWS — an isometric, exploded, or 3D
  view of a single fabricated object (a cover, a frame, a weldment) — where
  MANY numbered circles on leader lines point to the individual parts that make
  up that object. Those numbers are INSTANCE CALLOUTS that reference a separate
  PART LIST / BILL OF MATERIALS table elsewhere on the sheet — exactly the way
  grid bubbles reference a plan. This is a SPATIAL INDEX of many parts, so it
  plays the role of a Plan View.
 
  Classify as "Plan_View" ONLY when ALL of these are true together:
 
  • The drawing is a single 3D / isometric / pictorial whole-assembly view
    (NOT a flat orthographic table, NOT a zoomed cross-section of one joint).
  • MULTIPLE numbered circles (typically 4 or more) are spread ACROSS the
    object on leader lines, each pointing to a DIFFERENT part of the assembly.
  • The numbers are BARE INTEGERS (e.g. 1, 2, 14, 15) — NOT a "N/Sheet"
    detail reference like 3/S3-01, and NOT rows of a table.
  • The drawing answers "WHERE is each numbered part on this assembly?"
    — it locates instances that are DEFINED in a part list, it does not
    itself define material specs.
 
  If ALL four hold → return "type": "Plan_View" and "items": [].
 
  DO NOT classify such a view as:
    - "Schedule"  → it has NO spec columns (ITEM/QTY/DESCRIPTION/MATERIAL);
                    the numbered circles are not table rows.
    - "Detail"    → it shows the WHOLE assembly with many part callouts,
                    not ONE joint/connection with weld symbols and material
                    labels pointing into the drawing.
 
  COUNTER-EXAMPLES (these are NOT Step 1A):
    ✗ A flat ITEM / QTY / DESCRIPTION / MATERIAL table → that IS the part list
      itself → classify as "Schedule".
    ✗ A single flat cross-section of one part with 2-3 dimension callouts and
      no 3D body → "Detail".
    ✗ A numbered list of text statements → "Keyed_Notes".
    ✗ Circles in "N/Sheet" format (3/S3-01) pointing outward to other sheets
      → those are detail references, handled elsewhere.
 

  -------------------------
  STEP 1C — DETAIL DETECTION (CRITICAL)
  -------------------------

  If NOT a Plan View, check if this is a Detail / Section Drawing.

  Visual Clues for Detail:

  • A zoomed-in construction drawing (not full building layout)
  • Labeled with identifiers like:
    - "3/S-3.4"
    - "SECTION A-A"
    - "DETAIL 5"
  • Shows connections, joints, reinforcement, beams, columns, footing, etc.
  • May include callouts, arrows, or cut-section indicators
  • Usually NOT surrounded by full grid system like plan views

  If these are present → classify as "Detail"

  Return:
  "type": "Detail"
  "title": "<detected detail label or title if visible>"

  -------------------------
  STEP 2 — STRUCTURE DETECTION
  -------------------------

  If it is NOT a Plan View OR Detail:

  Determine whether it is:

  • A structured TABLE with rows and columns → "Schedule"
  • A numbered list of statements → "Keyed_Notes"
  • A Logo, Title Block, or non-structured element → "Ignore"

  Visual Clues for Schedule:
  - Grid lines
  - Repeated row structure
  - Column alignment
  - Symbol in first column with text to the right

  Visual Clues for Keyed Notes:
  - Sequential numbering (1., 2., 3.)
  - Paragraph text
  - No grid table structure

  If noise (logo, blank area, title block) → "Ignore"


  ------------------------------------------------------------
  STEP 3 — EXTRACTION RULES
  (ONLY IF Schedule OR Keyed_Notes)
  ------------------------------------------------------------

  If you classified as "Plan_View" or "Ignore":
  STOP.
  Return empty items list [].
  DO NOT read drawing symbols.
  DO NOT attempt quantity extraction.

  ------------------------------------------------------------
  STEP 3B — DETAIL EXTRACTION
  ------------------------------------------------------------

  If classified as "Detail":

  1. Extract the detail identifier or title from the image:
    Examples:
    - "3/S-3.4"
    - "DETAIL 5"
    - "SECTION A-A"

  2. If no clear title is visible:
    - Generate a short descriptive title (e.g., "beam_column_connection")

  3. Do NOT extract full materials table unless clearly visible.

  Return minimal structured output.

  ------------------------------------------------------------
  STEP 4 — SYMBOL SHAPE DETECTION (CRITICAL VISUAL TASK)
  ------------------------------------------------------------

  For Schedules or Keyed Notes:

  Focus on the FIRST COLUMN (or note number).

  You MUST NOT simply read the number.
  You MUST visually inspect the SHAPE surrounding it.

  Look carefully:

  • Is the number inside a HEXAGON?
    → Format as: "HEX-<number>"

  • Inside a CIRCLE?
    → Format as: "CIR-<number>"

  • Inside a SQUARE?
    → Format as: "SQR-<number>"

  • Inside a TRIANGLE?
    → Format as: "TRI-<number>"

  • No visible enclosing shape?
    → Use raw text (e.g., "24", F5")

  - No visible enclosing shape?
    → Use raw text (e.g., "F5", "CW8")

  - Bullet points (•, -, *) with no number?
    → Assign sequential numbers: "1", "2", "3"
    → These are unnumbered notes — give them an index.

  CRITICAL:
  The shape is more important than the text alone.
  Do not guess the shape.
  If unclear, default to plain text.

  ------------------------------------------------------------
  STEP 5 — TABLE RECONSTRUCTION (FOR SCHEDULES)
  ------------------------------------------------------------

  If the crop is classified as "Schedule":

  1. Identify the HEADER ROW of the table.
    - The header row usually contains column names such as:
      MARK, WIDTH, LENGTH, SIZE, VERTICAL, HORIZONTAL, SPACING, REMARKS, etc.

  2. Use the detected header row as the column names.

  3. Extract each table row as a structured record where:
    - Keys = column names
    - Values = cell text from that column.

  4. Preserve the exact text written in each cell.
    Do NOT summarize or rewrite values.

  5. If rows are partially cut or the table spans multiple crops,
    reconstruct the row using visible data.

  6. If rows appear duplicated due to overlapping crops,
    return each unique row only once.

  7. The FIRST COLUMN is usually the row identifier
    (examples: CP2121, HEX-1, F5, L4, etc.).
    Preserve it exactly as written.

  IMPORTANT:
  Do NOT compress the row into a sentence.
  Return the actual table structure.

  ------------------------------------------------------------
  STEP 6 — STRICT OUTPUT FORMAT
  ------------------------------------------------------------

  Return STRICT JSON only.

  For Schedule tables:

  {
    "type": "Schedule",
    "title": "Schedule Name",
    "columns": ["COLUMN1","COLUMN2","COLUMN3"],
    "rows": [
      {
        "COLUMN1": "value",
        "COLUMN2": "value",
        "COLUMN3": "value"
      }
    ]
  }

  For Keyed Notes:

  {
    "type": "Keyed_Notes",
    "title": "Notes Title",
    "columns": ["key_id", "text"],
    "rows": [
      {
        "key_id": "HEX-1",
        "text": "note description"
      }
    ]
  }
  
  If notes use bullet points (•) instead of numbers,
  assign sequential key_ids: "1", "2", "3", etc.

  For Plan Views:

  {
    "type": "Plan_View",
    "title": null,
    "columns": [],
    "rows": []
  }

  For Detail:

  {
    "type": "Detail",
    "title": "Detail Identifier or Description",
    "columns": [],
    "rows": []
  }

  For Ignore:

  {
    "type": "Ignore",
    "title": null,
    "columns": [],
    "rows": []
  }

  Rules:
  - No explanation text
  - No markdown
  - No extra commentary
  - If Plan_View or Ignore → items must be []
  - Maintain exact casing for key format (HEX-, CIR-, SQR-, TRI-)

  Think visually. Extract precisely.
  """
   return prompt

# ------ AGENT 3. SECTION DETAILS --------
def prompt_for_map_page_layout():
    map_page_layout_prompt = f"""
    You are a Forensic Layout Analysis Engine.

    You are analyzing a structural drawing sheet using:
    1. Full Page Layout Image
    2. Parsed JSON items (images + text + bbox)

    Your job is NOT to extract materials.
    Your job is to reconstruct the visual grouping logic of the page.

    ------------------------------------------------------------
    ### THE SITUATION

    This page contains multiple STRUCTURAL DETAILS.

    Each Detail is a self-contained “Definition Unit” that consists of:

    • A TITLE (usually at the bottom)
    • One or more DRAWINGS above the title
    • Possibly a TABLE above or beside the drawing
    • Possibly NOTES within the same boundary

    However:
    - MinerU has separated everything.
    - Titles, images, and notes are currently independent items.
    - Some details may contain MULTIPLE images (Plan + Section + Table).
    - Some titles may appear isolated.
    - Some drawings may be misaligned.

    You must reconstruct the page as a human would visually understand it.

    ------------------------------------------------------------
    ### MULTIMODAL CHAIN-OF-THOUGHT PROCESS

    -------------------------
    STEP 1 — IDENTIFY TITLE ANCHORS
    -------------------------

    From JSON text items:

    A Title usually:
    • Contains: DETAIL, SECTION, ELEVATION, TYP
    • May include bubble ref: "7/S-3.2"
    • Usually BELOW drawings

    Extract:
    - detail_id (e.g. "7/S-3.2")
    - title text

    Mark these as anchors

    -------------------------
    STEP 2 — SPATIAL GROUPING
    -------------------------

    For each Title Anchor:

    Look ABOVE it on the layout image.

    Visually determine:
    • Which Images are directly above this title?
    • Which Text Blocks are inside the same boundary?
    • Are there multiple drawings stacked?
    • Is there a table next to the drawing?

    CRITICAL RULE:
    One Detail Unit may include MULTIPLE images.
    Group ALL related images under the same title.

    Use spatial proximity logic:
    - Vertical alignment
    - Horizontal alignment
    - Bounding proximity

    If a drawing appears above a title and is not closer to another title, it belongs to that title.
    ------------------------------------------------------------
    ### STEP 3 — BBOX-TO-IMAGE ALIGNMENT (CRITICAL)

    You MUST NOT search entire image randomly.

    For each image:
    → Use its bbox
    → Search for title ONLY near that region

    Search order:
    1. Below bbox
    2. Above bbox
    3. Nearby horizontal

    ------------------------------------------------------------
    ### STEP 4 — FALLBACK TITLE DETECTION

    If NO title in JSON:

    1. Use bbox to locate region in full image
    2. Visually find nearest title text

    If found:
    → assign detail_id

    If NOT found:
    → assign fallback:
      "UNKNOWN_<json_id>"

    CRITICAL:
    Do NOT ignore such details.
   ------------------------------------------------------------
    ### STEP 4B — HITL CROP HANDLING (CRITICAL)

    Some images in the JSON are marked with "hitl": true.
    These are crops drawn manually by the user on the exact region of interest.

    For HITL crops:
    1. Their bbox tells you EXACTLY where on the page they are.
    2. Look BELOW that bbox region in the full layout image for the title text.
       Titles are usually within 100-200px below the drawing bottom edge.
    3. If you can see title text below the crop region → assign it as detail_id and title.
    4. If the crop itself contains a title bubble (e.g. "3/S3-02") visible in the image → use that.
    5. If no title is visible anywhere → use 'HITL_{{i}}/{{sheet}}' as fallback detail_id.

    IMPORTANT: HITL crops are HIGH PRIORITY.
    Always include them in the output, even if you cannot find a title.
    Never discard a HITL image.

    -------------------------
    STEP 5 — HANDLE FRAGMENTATION
    -------------------------

    MinerU may have:
    • Split one table into multiple image pieces
    • Split notes into multiple blocks
    • Slight coordinate offsets

    You must logically merge them if they visually belong together.

    Do NOT create separate DetailGroups for fragments of the same unit.

    -------------------------
    STEP 6 — FINAL GROUP CONSTRUCTION
    -------------------------

    For each identified Title:

    Create a DetailGroup object:

    - detail_id: Extracted bubble reference
    - title: The full visible title text
    - image_files: List of all associated image filenames (from JSON img_path)
    - text_blocks: List of all associated text content

    If a Title has no associated images, still return it.

    If an image has no clear Title:
    - If it is a HITL crop (hitl: true) → use fallback ID, NEVER discard
    - If it is a MinerU crop → ignore it (it may be a duplicate or fragment)

    ------------------------------------------------------------
    STEP 7 — FULL PAGE IMAGE FALLBACK

    If the JSON contains only 1 image that spans nearly the entire page bbox
    (width > 90% of page, height > 90% of page), this means MinerU treated
    the whole page as a single image and failed to segment it.

    In this case:
    1. Ignore that single full-page image entirely.
    2. Rely ONLY on HITL crops (marked "hitl": true) for grouping.
    3. For each HITL crop, look at its bbox coordinates in the full layout image
       to find the title below it.
    4. Create one DetailGroup per HITL crop.
    ------------------------------------------------------------
    ### ZERO-HALLUCINATION RULE

    DO NOT invent titles.
    If not clearly visible → use UNKNOWN.

    ------------------------------------------------------------
    ### SHEET REFERENCE FORMAT RULE (CRITICAL)

    All detail_id values contain a sheet reference after the slash.
    Sheet references ALWAYS start with the letter S (uppercase).

    Common formats you will see:
      1/S3-01    3/S3-02    4/3-01    7A/S5-05

    CRITICAL OCR CORRECTION:
    The digit 5 and the letter S look visually similar.
    If you read a sheet reference that starts with a digit like:
      1/55-02  →  WRONG. Correct it to  1/S5-02
      3/53-01  →  WRONG. Correct it to  3/S3-01
      2/55-01  →  WRONG. Correct it to  2/S5-01

    Rule: After the slash (/), the sheet prefix is ALWAYS a single letter S
    followed by digits. Never two digits. Never "55", "53", "51".
    If you see two digits after the slash before a hyphen → the first digit
    is always the letter S misread. Replace it with S.

    Apply this correction BEFORE writing any detail_id to output.

    ------------------------------------------------------------
    ### SHEET REFERENCE RULE FOR DETAIL IDs

    When reading a title bubble that contains ONLY a letter or number (e.g. just "B", "A", "1", "2"):
    - The detail_id format is: "LABEL/SHEET"
    - The SHEET part must come from the actual sheet number of this drawing page
    - DO NOT invent or guess a sheet reference
    - DO NOT use sheet references visible in other details on the same page
    - If no sheet reference is visible in the bubble → use the placeholder: "LABEL/UNKNOWN"
    - NEVER hallucinate a sheet number like S3-01, S5-06, etc. that is not visible in this specific bubble

    ------------------------------------------------------------
    ### DETAIL ID EXTRACTION RULE (CRITICAL)

    The detail_id must contain ONLY the reference label, not the type word.

    Common title formats you will see:
      "SECTION A"        → detail_id = "A"      (not "SECTION A")
      "SECTION B-B"      → detail_id = "B-B"    (not "SECTION B-B")  
      "DETAIL 1"         → detail_id = "1"      (not "DETAIL 1")
      "DETAIL 2/S3-01"   → detail_id = "2/S3-01"
      "SECTION A/S3-01"  → detail_id = "A/S3-01" (not "SECTION A/S3-01")

    Strip these words from the START of detail_id before writing output:
      SECTION, DETAIL, ELEVATION, PLAN, VIEW, TYPICAL, TYP

    The label is ONLY the alphanumeric reference: A, B, B-B, 1, 2, 1A, 7A, etc.
  ------------------------------------------------------------

    ### OUTPUT FORMAT

    Return a JSON list:

    [
    {{
        "detail_id": "7/S-3.2",
        "title": "LADDER DETAIL",
        "image_files": ["crop_005.jpg", "crop_006.jpg"],
        "text_blocks": [
            "TYP. SEE PLAN",
            "ALL WELDS 3/16\" FILLET"
        ]
    }}
    ]

    Rules:
    - No explanation
    - No markdown
    - Return only JSON list
    - Preserve exact filenames
    - Preserve full text blocks
    - Do not invent missing IDs
    """
    return  map_page_layout_prompt

def prompt_for_classify_image_as_plan_detail():
    return """You are classifying a cropped image from a structural engineering drawing.

Classify the image as EXACTLY ONE of these four categories:

PLAN_VIEW — A spatial layout of the entire structure:
- Has grid lines with letters (A, B, C, D, E, F, G) AND numbers (1, 2, 3) forming a grid
- Shows walls, columns, beams arranged across a large floor area
- Has dimension lines spanning large distances (e.g. 68'-0", 29'-6", 120'-4")
- Contains multiple callout bubbles scattered across the plan (e.g. 3/S-3.2, 4/S-3.2, hex symbols)
- Is a Foundation Plan, Floor Plan, Framing Plan, or Roof Plan
- EVEN IF it has callout bubbles — if it shows a full spatial layout → PLAN_VIEW

DEPENDENT_DETAIL — A detail that references other details for its materials:
- SIGNAL 1 — GRAPHICAL: Contains 1 or more callout bubbles (circle-over-triangle)
  INSIDE the drawing pointing outward to other details (e.g. G13/S-101, A1/S-101)
- SIGNAL 2 — TEXT-BASED REFERENCE: Leader line text contains PER X/Y or SEE X/Y where Y is a sheet code (alphanumeric with hyphens like ST10, S522, S3-01). The word after SEE or PER must be a NUMBER/SHEETCODE pattern.
  These do NOT qualify as SIGNAL 2:
      SEE PLAN, SEE SCHEDULE, SEE ELEVATION — these are quantity notes, not detail dependencies
      SEE CIVIL, SEE ARCH, SEE MECH — these reference other disciplines, not structural details
      PER CODE, PER SPEC — specification references, not detail callouts
- Either SIGNAL 1 or SIGNAL 2 alone is enough to classify as DEPENDENT_DETAIL.
- The detail's own title bubble appears at the BOTTOM as a label, not scattered inside.

INDEPENDENT_DETAIL — A single self-contained close-up detail/Schedule:
- Shows ONE specific connection, assembly, or component (base plate, ladder, footing, shear wall)
- Has its own title like "LADDER DETAIL", "BASE PLATE DETAIL", "TOP CONNECTION"
- Contains leader arrows pointing to material labels (e.g. "SIDE RAILS - SEE PLAN", "3/16 WELD")
- Shows close-up dimensions like 1'-3", 4'-0", 3/16", 10"
- May say "SEE PLAN" or "SEE SCHEDULE" — but these refer to quantities only, NOT to other
  details for materials. The steel members in this detail are fully labeled here.
- Example: Ladder detail with channels, rungs, anchor bolts all labeled directly

IGNORE — Not a drawing, just a symbol or tag:
- A small circle or hexagon containing ONLY a reference like "5/S-3.2" or "1/S-1.0"
- A title block, sheet border, company logo, north arrow, or scale bar
- Image is mostly white/blank with just a tiny symbol in the center
- Contains NO dimension lines, NO leader arrows, NO structural elements drawn

CRITICAL DECISION RULES:
1. Grid letters AND numbers forming a layout with walls/columns → PLAN_VIEW
2. Leader text has "PER X/Y" or "SEE X/Y" patterns pointing to other details → DEPENDENT_DETAIL
3. Graphical callout bubbles scattered INSIDE the drawing → DEPENDENT_DETAIL
4. Close-up single component, all materials fully labeled here, no PER/SEE detail refs → INDEPENDENT_DETAIL
5. Just a reference bubble/tag with no drawing content → IGNORE

Return ONLY valid JSON with no explanation:

{"type": "PLAN_VIEW"}
{"type": "DEPENDENT_DETAIL"}
{"type": "INDEPENDENT_DETAIL"}
{"type": "IGNORE"}"""


def prompt_for_extract_single_detail(group_title: str, group_detail_id: str):
    return f"""
    You are a Structural Steel Detailer performing a Material Takeoff (MTO).

    Detail Title : "{group_title}"
    Detail ID    : "{group_detail_id}"

    All images and text blocks supplied belong ONLY to this detail.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    YOUR ROLE
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    You are a READER and CATALOGUER — not a calculator.

    You MUST:
    - Read every steel material callout that is visibly labeled.
    - Record the exact size as written on the drawing.
    - Record piece_length_ft ONLY if a dimension is explicitly written next to the material.
    - Record qty_rule exactly as the drawing implies — no arithmetic.
    - Flag "SEE SCHEDULE", "SEE PLAN", "SEE TABLE" references so they can be resolved later.
    - Detect callout bubbles INSIDE this drawing and record them as dependencies.
    - Record finish treatment per material.

    You MUST NOT:
    ✗ Calculate rung counts or spacing formulas
    ✗ Compute stair stringer lengths or hypotenuse
    ✗ Calculate seep ring circumferences
    ✗ Sum up or estimate total linear feet
    ✗ Infer materials that are not explicitly labeled
    ✗ Extract concrete, rebar, anchor bolts cast in concrete, mesh, hardware items

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    STEP 1 — WHAT TO EXTRACT (STEEL ONLY)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Extract ONLY structural steel items with a visible leader arrow or label:

      INCLUDE:
      • Wide flange beams/columns      W8X13, W24X62, WF12X26
      • HSS tube steel                 HSS5X5X5/16, HSS4X2X3/16
      • Channels                       C8X11.5, MC6X15.1
      • Angles                         L4X4X1/4, L3X2X3/16
      • Flat bar                       FB 1/4X3, 3/4X3 FB
      • Plate                          PL 1/2, 1/4" PL, BASE PLATE
      • Pipe (structural)              PIPE 3" SCH 40, 2" STD WT PIPE
      • Threaded rod / tie rod         ROD3/4, 3/4" DIA ROD
      • Stair stringers (C or MC)      C10X15.3 STRINGER
      • Tube handrail / guardrail      1-1/2" SCH 40 PIPE RAIL
      • Seep rings / waterstop rings   1/4"X3" SEEP RING

      EXCLUDE (do not extract these at all):
      • Concrete, grout, rebar (#3, #4, #5, hoops, dowels)
      • Anchor bolts embedded in concrete
      • Masonry, wood, plywood, sheathing
      • Hardware (screws, hinges, hasps, padlocks, screens)
      • Manufacturer standard items (guardrail post caps, etc.)
      • Grating (unless it is fabricated steel platform grating)

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    STEP 2 — HOW TO FORMAT item_name
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Extract the SIZE only — never combine size and length in item_name.

      Angles:     L4X4X1/4    (always capitalize x → X)
      W shapes:   W8X13
      HSS:        HSS5X5X5/16
      Channel:    C8X11.5  or  MC6X15.1
      Flat bar:   FB1/4X3
      Plate:      PL1/2
      Pipe:       PIPE3SCH40   (diameter + schedule joined)
      Rod:        ROD3/4
      Seep ring:  SEEP RING 1/4"X3"

      If the drawing writes a length after the size (e.g. "L4X4X1/4 X 0'-3\""):
      → item_name  = L4X4X1/4
      → piece_length_ft = 0.25   (convert: 0'-3" = 0.25 ft)

      Length conversion:
        0'-3"  → 0.25      3"   → 0.25
        10"    → 0.833     1'-6" → 1.5
        2'-0"  → 2.0

      If NO explicit length is written:
      → piece_length_ft = null

      Spacing (@ 12" O.C.) is NOT a length → piece_length_ft = null

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    STEP 3 — qty_rule: what the drawing says
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Write what the drawing explicitly states, no arithmetic:

      FIXED count visible:
        "FIXED: 2"       (drawing says "2 RAILS" or shows 2 members)
        "FIXED: 4"       (4 bolts shown)

      Count variable / depends on dimension:
        "VARIABLE: SEE PLAN"
        "VARIABLE: spacing @ 12\" O.C."
        "VARIABLE: height of ladder"
        "VARIABLE: stair run length"

      References a schedule or table — CRITICAL:
        If the drawing says "SEE SHEAR WALL SCHEDULE", "PER LINTEL SCHEDULE",
        "SEE RAFTER TABLE", or any similar reference:
        → qty_rule = "SEE SCHEDULE: <exact schedule name as written>"
        → Example: "SEE SCHEDULE: SHEAR WALL SCHEDULE"
        → Example: "SEE SCHEDULE: LINTEL SCHEDULE"
        This tells Agent 4 to do a semantic search in the knowledge graph
        to resolve the actual specification.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    STEP 4 — notes field: finish + dependencies
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    The notes field carries two types of information, pipe-separated:

      A) FINISH TREATMENT (mandatory for every material):
        GALVANIZED   — if callout says GALV, HDG, HOT DIP GALVANIZE
        SS316        — if callout says SS, STN STL, STAINLESS, 316, 304
        COATED       — if callout says POWDER COAT, EPOXY COAT
        RAW          — if nothing is specified

      B) DEPENDENCY — two sources, both must be captured:

         SOURCE 1 — GRAPHICAL CALLOUT BUBBLE (already handled):
          If you see a circle-over-triangle symbol INSIDE this drawing:
          → add  DEPENDS_ON: 4/S-3.2

         SOURCE 2 — TEXT-BASED REFERENCE (NEW):
          If a leader line text or callout note contains any of these patterns:
            "PER 303/ST10"        → DEPENDS_ON: 303/ST10
            "SEE 305/ST10"        → DEPENDS_ON: 305/ST10
            "PER DETAIL 7/S522"   → DEPENDS_ON: 7/S522
            "SEE DETAIL 2/S3-01"  → DEPENDS_ON: 2/S3-01
            "PER 301/ST10"        → DEPENDS_ON: 301/ST10

          PATTERN RULE: Look for the keyword PER or SEE followed immediately
          by a value matching the format NUMBER/SHEET (e.g. 303/ST10, 7/S-3.2).
          Extract that NUMBER/SHEET as the dependency ID.

          If the text says "SEE ROOF PLAN" or "SEE PLAN" with NO detail number:
          → add  DEPENDS_ON: SEE_PLAN
          → This flags it for Agent 4 to handle via plan image scan.

          If the text says "PER SHEET ST6" or "PER SHEET S522" with NO detail number:
          → add  DEPENDS_ON: SHEET_ST6  or  DEPENDS_ON: SHEET_S522

      Combined example:
        notes = "GALVANIZED | DEPENDS_ON: 4/S-3.2, 5/S-3.2"
        notes = "RAW | Side rails"
        notes = "SS316 | Handrail pipe"

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    STEP 5 — FABRICATION METRICS (CRITICAL — READ ALL 3 CAREFULLY)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
    This step is MANDATORY. You must actively scan the drawing for every
    bolt, hole, and weld. Do not leave these at 0 unless you have confirmed
    there are truly none in this detail.
 
    ─────────────────────────────────────────
    5A — BOLT COUNT  (bolt_count)
    ─────────────────────────────────────────
 
    Structural drawings show bolts in three ways — read ALL of them:
 
    METHOD 1 — EXPLICIT COUNT TEXT:
      Look for text like:
        "(4) 3/4" A325 BOLTS"     → bolt_count = 4
        "2 - 3/4" BOLTS"          → bolt_count = 2
        "(8) 1/2" H.S. BOLTS"    → bolt_count = 8
        "4 BOLTS EACH SIDE"       → bolt_count = 8  (4 × 2 sides)
      Rule: Read the number directly. If "EACH SIDE", "EA. SIDE", "E.S." multiply by 2.
 
    METHOD 2 — BOLT PATTERN / ARRAY:
      Look for a grid of small circles or X marks representing bolt holes on a plate.
      Count the visible dots/circles in the bolt pattern.
        2 rows × 2 columns = 4 bolts
        3 rows × 2 columns = 6 bolts
      Rule: Count all visible bolt symbols in the pattern. One symbol = one bolt.
 
    METHOD 3 — SPACING RULE:
      If the drawing says "@ 12\" O.C." on a connection plate or flange:
      → bolt_count = 0  (VARIABLE — Agent 4 will compute from wall length)
      → qty_rule for that bolt item should capture the spacing
 
    SPECIAL RULES:
      • "TYP." after a bolt count means that count applies to EACH instance shown.
        If "4 BOLTS TYP." appears and 2 connections are drawn → bolt_count = 8
      • "EACH END" means multiply count by 2.
      • High-Strength (H.S.), A325, A490, A307 — all count as bolts regardless of grade.
 
    ─────────────────────────────────────────
    5B — HOLE COUNT  (hole_count)
    ─────────────────────────────────────────
 
    Holes are drilled through steel plates/members to receive bolts.
 
    DEFAULT RULE (apply unless drawing specifies otherwise):
      hole_count = bolt_count × 2
      Reason: Most bolted connections pass through TWO steel plates
              (e.g., gusset + flange), creating one hole in each.
 
    SINGLE-PLATE RULE:
      If the detail clearly shows a bolt going into only ONE plate
      (e.g., anchor bolt into a base plate only, not through a second member):
      → hole_count = bolt_count × 1
 
    EXPLICIT HOLE CALLOUT:
      If drawing says "(4) HOLES", "DRILL 6 HOLES", "8 PUNCHED HOLES":
      → hole_count = exact number stated. Ignore the default formula.
 
    SLOTTED / OVERSIZED HOLES:
      "SLOTTED HOLES", "OVERSIZED HOLES" — still count as holes, one per bolt location.
 
    ─────────────────────────────────────────
    5C — WELD INCHES  (weld_inches)
    ─────────────────────────────────────────
 
    Structural drawings show welds in THREE ways — you must read all three:
 
    METHOD 1 — WELD SYMBOL ON DRAWING:
      The standard weld symbol has:
        • An ARROW pointing to the weld location
        • A HORIZONTAL REFERENCE LINE
        • A SIZE written below-left (e.g., 3/16, 1/4, 5/16)
        • A LENGTH written to the right of the reference line (e.g., 6, 12, 4)
        • A shape symbol (triangle = fillet, rectangle = groove, etc.)
 
      Reading rule:
        SIZE = weld throat size in inches (e.g., 3/16 = 0.1875")
        LENGTH = weld run length in INCHES (e.g., 6 = 6 inches)
 
      You MUST extract the LENGTH, not the size.
      weld_inches contribution = LENGTH value in inches.
 
      Examples from drawing weld symbols:
        "3/16" with "6" → weld run = 6 inches
        "1/4" with "12" → weld run = 12 inches
        "3/16" with no length → weld is CONTINUOUS (see below)
 
    METHOD 2 — TEXT CALLOUT:
      Look for text like:
        "ALL WELDS 3/16" FILLET"          → continuous weld, length = perimeter of all joints
        "3/16" FILLET WELD BOTH SIDES"    → weld_inches = joint_length × 2
        "WELD 6" LONG"                    → 6 inches
        "WELD 4" EA. SIDE"                → 8 inches (4 × 2 sides)
        "FULL PEN WELD"                   → full penetration, record length as joint length
        "1/4" FILLET WELD @ 12" O.C."    → intermittent weld, not continuous
        "(4) 2" LONG WELDS"               → 8 inches total (4 × 2")
        "CONTINUOUS WELD ALL AROUND"      → weld_inches = perimeter of the welded element
 
    METHOD 3 — CONTINUOUS / "ALL AROUND" WELD:
      If a weld symbol has a CIRCLE at the reference line junction:
        → this means "weld all around" = continuous weld on all sides of the joint
        → weld_inches = perimeter of the member cross-section at that joint
        → If dimensions not available → record weld_inches for that joint as the
           member perimeter implied by its size (e.g., HSS4X4 all-around = 4×4=16")
 
    ACCUMULATION RULE — CRITICAL:
      weld_inches = SUM of ALL weld lengths in this entire detail.
 
      For each weld location found:
        1. Read its length in inches.
        2. If "BOTH SIDES" → multiply by 2.
        3. If "EACH END" → multiply by 2.
        4. If "TYP." and N instances shown → multiply by N.
        5. Add to running total.
 
      Final weld_inches = total accumulated inches across the whole detail.
 
    WELD-ONLY NOTE LINE RULE:
      Notes like "ALL WELDS SHALL BE 3/16" FILLET" are GLOBAL notes —
      they set the weld SIZE but do NOT give you weld LENGTHS.
      Do NOT add weld_inches from size-only notes unless a length is also visible.
 
    ─────────────────────────────────────────
    5D — FABRICATION OUTPUT FORMAT
    ─────────────────────────────────────────
 
    Always return all three fields. Never omit:
 
      {{
        "bolt_count": <integer>,
        "hole_count": <integer>,
        "weld_inches": <float>
      }}
 
    If a field is genuinely zero (none visible after careful scan): use 0.
    If you found bolts but no explicit hole count: apply the 2× default.
    If you found a weld size but no length: use 0 for that weld (size ≠ length).
 
    WORKED EXAMPLES:
 
    Example A — Ladder Detail:
      Drawing shows: "(4) 3/4" A325 BOLTS" at base plate
                     "3/16" FILLET WELD 6" LONG" on each side rail clip (2 clips)
      → bolt_count = 4
      → hole_count = 4 × 2 = 8  (through base plate + rail web)
      → weld_inches = 6 + 6 = 12  (two clips, 6" each)
 
    Example B — Base Plate Connection:
      Drawing shows: bolt pattern of 4 circles on plate
                     "ALL WELDS 3/16" FILLET" (no length given)
                     plate is 8"×8" base plate
      → bolt_count = 4
      → hole_count = 4 × 2 = 8
      → weld_inches = 0  (global size note only, no length given for this detail)
 
    Example C — Shear Wall Clip:
      Drawing shows: "(2) 5/8" BOLTS EA. SIDE"
                     "1/4" × 4" WELD BOTH SIDES"
      → bolt_count = 2 × 2 = 4  (2 each side)
      → hole_count = 4 × 2 = 8
      → weld_inches = 4 × 2 = 8  (4" each side, both sides)

     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    STEP 6 — visual_reasoning (chain of thought)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
    Before listing materials, write a short paragraph covering:
    - What type of detail is this? (ladder, stair, connection, shear wall, beam, etc.)
    - What is the general structural system shown?
    - What key dimensions or callout references are visible?
    - Are there any "SEE SCHEDULE" or "SEE PLAN" references?
    - Are there internal callout bubbles (dependencies)?
    - What bolt patterns, weld symbols, or hole callouts did you find?
      Describe each one explicitly before listing your final counts.
 
    This reasoning IS your audit trail — be specific and technical.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    FINAL VALIDATION — before returning
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    For each material in your list, ask:
      1. Is there a visible leader arrow or label pointing to this material? → keep
      2. Is it a concrete/rebar/hardware item? → remove
      3. Did I put a length in item_name? → move it to piece_length_ft, clean item_name
      4. Did I write a formula or computed number in qty_rule? → replace with "VARIABLE: ..."
      5. Is there a "SEE SCHEDULE" reference? → use "SEE SCHEDULE: <name>" format

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    OUTPUT FORMAT
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Return a single DetailExtraction JSON object:

    {{
      "detail_id": "{group_detail_id.split('/')[0] if '/' in group_detail_id else group_detail_id}",
      "title": "{group_title}",
      "visual_reasoning": "This is a ladder detail showing MC6x15.1 side rails with 3/4\" rod rungs at 12\" O.C. Height not stated — marked VARIABLE. Internal callout 4/S-3.2 references the base connection detail.",
      "materials": [
        {{
          "item_name": "MC6X15.1",
          "material_type": "C",
          "piece_length_ft": null,
          "qty_rule": "FIXED: 2",
          "notes": "GALVANIZED | Side rails. DEPENDS_ON: 4/S-3.2"
        }},
        {{
          "item_name": "ROD3/4",
          "material_type": "ROD",
          "piece_length_ft": 1.5,
          "qty_rule": "VARIABLE: spacing @ 12\" O.C.",
          "notes": "RAW | Rungs, width 1'-6\" stated"
        }},
        {{
          "item_name": "L4X4X1/4",
          "material_type": "L",
          "piece_length_ft": 0.25,
          "qty_rule": "FIXED: 2",
          "notes": "GALVANIZED | Base clips. SEE SCHEDULE: ANCHOR BOLT SCHEDULE"
        }}
      ],
      "fabrication": {{
        "bolt_count": 4,
        "hole_count": 4,
        "weld_inches": 6.0
      }}
    }}

    Rules:
    - JSON only, no markdown, no commentary
    - Do not hallucinate materials
    - If no steel materials are visible → return empty materials list
    - piece_length_ft must be a number in decimal feet or null
    - Never write a formula result in qty_rule — write what the drawing says
"""

# ------ AGENT 4. AGENT MERGER ----------

def prompt_extract_floor_plan_keywords():
    return """You are reading a structural engineering floor plan.
Your job is to extract schedule reference codes that need definition lookup.

INCLUDE:
- Short schedule/spec codes like: SC-6A, FS7.0, FC2.0, MW-8A, MC-1, CP-4, CW-12
- These follow patterns like: 2-3 letters + optional number/decimal
- They appear as text labels near structural elements

DO NOT INCLUDE:
- Detail callout bubbles with slash: 2/S201, 5/S-3.2, A/S503
- Symbol letters inside triangles/diamonds/circles: A, B, C, R, Q, P, N
- Dimensions: 15'-0", T.O.F=93'-4"
- General notes: TYP, SEE ARCH, BLOCK-OUT, SLOPE SLAB
- Grid lines: J, H, G, F, E, D, C (single letters at drawing border)

Return ONLY a flat JSON array of unique strings. No explanation.
Example: ["SC-6A", "FS7.0", "FC2.0", "MW-8A", "MC-1", "CP-4", "CW-12"]
If nothing found return: []"""

def prompt_for_agent_4_merger(DETECTED_SYMBOLS: str, valid_materials_str: str, sheet_number: str,SHEET_DEFINITIONS: str = "[]"):
    """
    Accepts the list of dictionaries returned by Neo4j (graph_data).
    Each symbol dict contains:
      - text_content      : symbol label e.g. "3/S3-01" or "hex-1"
      - bbox              : [x1, y1, x2, y2] pixel location on this floor plan image
      - linked_definition : dict from Neo4j with keys:
            BOM           : list of MaterialItem dicts from Agent 3
                            Each item has: item_name, material_type, piece_length_ft, qty_rule, notes
            fabrication   : dict with bolt_count, hole_count, weld_inches  ← PER DETAIL INSTANCE
            Title         : detail title string
            ID            : detail id string
    """
    prompt = f"""
    You are the Senior Structural Steel Estimator for DAX Manufacturing.

    You have TWO sources of material data — you MUST use BOTH:

    SOURCE 1 — DETECTED SYMBOLS (from GraphRAG):
      Callout bubbles and hex symbols already detected on this plan, enriched with
      their linked detail BOM and fabrication metrics from Neo4j.
      These drive CASE A (detail assemblies) and CASE B (shear wall spacing rules).

    SOURCE 2 — FLOOR PLAN IMAGE (direct visual read):
      Material labels written directly on the plan — beams, columns, HSS, angles, plates
      called out with leader arrows or text alongside members (e.g. "W14X22", "HSS4X4X1/4").
      These are NOT captured by symbol detection. YOU must read them directly from the image.
      These drive CASE C (beams), CASE D (columns), CASE E (lintels), and any
      material label that appears on the plan without a callout bubble.

    SOURCE 3 — SHEET DEFINITIONS (schedules, notes, rules from this sheet + global structural specs):
      All schedule tables and keyed notes extracted from this sheet and stored in the
      knowledge graph. These define text marks you may see written directly on the plan
      — marks like CW1, CW3, MW1, F11, MJ1 that are NOT inside callout bubbles.

      When you see a text label on the plan (e.g. "CW1" on a wall, "MJ1" at a door jamb),
      look it up in the SHEET DEFINITIONS below to find its specification.
      
      Also included are GLOBAL structural specifications extracted from the project's
      general notes pages (concrete strengths, steel grades, bolt standards, etc.).
      Use these to validate or resolve any material spec that is not explicitly called
      out on the floor plan itself.

      Use this data to:
      - Resolve wall marks (CW1 → 24" thick, 13'-0" height, #6 AT 6" O.C.)
      - Resolve footing/note references (F11 → concrete lintel spec)
      - Resolve any mark that matches a schedule row ID below

      SHEET DEFINITIONS:

    You are EXECUTING STRUCTURAL ESTIMATION LOGIC across both sources
    to produce a complete Final Bill of Materials.

    Your output feeds directly into DAX's 6-point takeoff report:
      1. Material Takeoff     → description + material_size + total_linear_feet
      2. Total Material Weight → total_weight_lbs  (computed post-LLM, you supply total_linear_feet)
      3. Member / Part Count  → quantity
      4. Inches of Weld       → total_weld_inches  ← YOU MUST COMPUTE THIS
      5. Hole Count           → total_holes        ← YOU MUST COMPUTE THIS
      6. Bolt Count           → total_bolts        ← YOU MUST COMPUTE THIS

    Fields 4, 5, and 6 MUST be populated for every item where fabrication data exists.
    Leaving them at 0 when data is available is a CRITICAL ERROR.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    INPUT DATA
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    DETECTED SYMBOLS (enriched from Neo4j GraphRAG):
    {DETECTED_SYMBOLS}

    VALID MATERIALS LIST:
    {valid_materials_str}

    CURRENT DRAWING SHEET: {sheet_number}

    SHEET DEFINITIONS (schedules & notes for this sheet):
    {SHEET_DEFINITIONS}

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    STEP 1 — INVENTORY EVERY SYMBOL ON THE PLAN
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Before computing anything:

    1. List every symbol from DETECTED_SYMBOLS.
    2. For each symbol note:
      - Its text_content (e.g. "3/S3-01", "hex-1")
      - Whether it has a linked_definition (YES / NO)
      - Its occurrence_count field — this is N, your multiplier.
        This has been pre-computed for you. Use it directly. Do NOT recount from the list.

    N is the OCCURRENCE COUNT for that symbol. Every metric you compute gets multiplied by N.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    STEP 2 — FULL VISUAL SCAN OF THE FLOOR PLAN IMAGE
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Scan the ENTIRE floor plan image in two passes:

    PASS A — SYMBOL-ANCHORED SCAN:
    For each detected symbol, use its bbox as a spatial anchor.
    Look around that location for:
      • Dimension text (e.g. "13'-10\"", "27'-0\"", "4'-0\" R.O.")
      • Grid line labels (A, B, C / 1, 2, 3)
      • Elevation notes (e.g. "T.O.S. EL. 18'-0\"", "F.F. EL. 0'-0\"")

    Record the nearest relevant dimension for each symbol.
    If no dimension is visible → note "dimension not found" and fall through
    to grid-span estimation or flag as "field measure required".

    CRITICAL: You MUST output a separate BOM row for EVERY item in 
      linked_definition.BOM. Do NOT skip PL, L, or FB items. Even if you 
      read a visual label (e.g. MC6X15.1) for the rails — output BOTH the 
      visual item AND all definition items. They are different components 
      of the same assembly.

    PASS B — DIRECT MATERIAL LABEL SCAN (critical — do not skip):
    Independently scan the entire image for material labels written directly on members.
    These appear WITHOUT a callout bubble — just text alongside a structural member.

    Look for:
      • Steel section labels on beams/columns: "W14X22", "HSS4X4X1/4", "MC6X15.1"
      • Plate callouts: "1/2\" PL", "BASE PL 3/4\""
      • Angle callouts: "L4X4X1/4"
      • Any steel size written with a leader arrow pointing to a drawn member

    For each direct label found:
      1. Record the material size exactly as written.
      2. Count how many times that label appears (each occurrence = one member).
      3. Read the span/length dimension nearest to that member on the plan.
      4. Add these as BOM items just like symbol-triggered items — they are equally valid.

    These direct-label materials are NOT in DETECTED_SYMBOLS — they come purely from
    your visual read of the image. Do not skip them.

    IMPORTANT: PASS B is strictly for material labels written directly on members
    WITHOUT any callout bubble. Do NOT use PASS B to re-extract materials that 
    are already defined in linked_definition.BOM — those come from CASE A.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    STEP 3 — LENGTH RESOLUTION  (strict priority order)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    For each material item inside a linked_definition.BOM:

    PRIORITY 1 — piece_length_ft is a number (not null):
      → total_linear_feet = piece_length_ft × quantity × N
      → No further dimension lookup needed for this item.

    PRIORITY 2 — piece_length_ft is null AND qty_rule is VARIABLE:
      → Read qty_rule text to determine what drives the length:
          "VARIABLE: height of ladder"    → use ladder_height_ft from Step 2 scan
          "VARIABLE: spacing @ 12\" O.C." → compute count from dimension ÷ spacing (see CASE B)
          "VARIABLE: stair run length"    → use stair_run_ft from Step 2 scan
          "VARIABLE: SEE PLAN"            → use nearest dimension from Step 2
      → Apply the appropriate formula below (CASES A–H).
      → total_linear_feet = computed_length × N

    PRIORITY 3 — piece_length_ft is null AND material is plan-based (beam, column, lintel):
      → Measure directly from plan image using grid span or dimension text from Step 2.
      → total_linear_feet = span_length_ft × quantity × N

    PRIORITY 4 — No dimension found anywhere:
      → total_linear_feet = 0
      → Set logic_trace: "Length unknown — field measure required"
      → Still output the item with quantity and fabrication metrics.

    RULE: NEVER invent a length. NEVER use bolt/washer dimensions as member lengths.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    STEP 4 — FABRICATION ROLLUP  ← THIS IS THE CRITICAL STEP
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    For EVERY symbol that has a linked_definition with a fabrication block:

      linked_definition["fabrication"] contains:
        bolt_count   — bolts per ONE instance of this detail
        hole_count   — holes per ONE instance of this detail
        weld_inches  — weld inches per ONE instance of this detail

      YOUR JOB: Scale by N (the occurrence count from Step 1).

      FORMULAS:
        total_bolts       = bolt_count   × N
        total_holes       = hole_count   × N
        total_weld_inches = weld_inches  × N

      CRITICAL RULES:
      • If bolt_count > 0 but hole_count = 0 → apply default: hole_count = bolt_count × 2, THEN scale by N.
        (This catches cases where Agent 3 missed holes — apply the 2× safety default.)
      • If fabrication block is missing or null → set all three to 0, note in logic_trace.
      • NEVER leave total_bolts, total_holes, total_weld_inches blank or missing from output.
      • These fields belong at the MATERIAL ITEM level — each BOM row gets its own scaled values.

      AGGREGATION ACROSS MATERIALS IN THE SAME DETAIL:
      When one detail has multiple materials (e.g., rails + rungs + base clips), the fabrication
      metrics apply to the DETAIL as a whole, not per material line.
      Strategy: Assign all fabrication metrics to the PRIMARY structural member of that detail
      (the first non-ROD, non-plate item in the BOM list). Set other items in same detail to 0.
      Note this in logic_trace: "Fabrication metrics assigned to primary member. Rails: bolt=4, hole=8, weld=12in."

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    STEP 5 — SYMBOL TYPE LOGIC BRANCHING
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ─────────────────────────────────────────
    CASE A — DETAIL CALLOUT (e.g., 3/S3-01)
    ─────────────────────────────────────────

    linked_definition.BOM gives the component list.
    For each component:
      • If piece_length_ft is a number → use Priority 1 formula.
      • If qty_rule = "FIXED: N" → quantity = N per instance × symbol occurrences.
      • If qty_rule = "VARIABLE: ..." → apply the matching sub-case below.
    Apply fabrication rollup from Step 4 to primary member.

    ─────────────────────────────────────────
    CASE B — SHEAR WALL / SPACING RULE (hex symbol)
    ─────────────────────────────────────────

    The linked_definition.Rows may contain a mixed schedule row with wood, nails, and an anchor bolt spec together.
    Ignore everything except the anchor bolt — find any field whose key contains "ANCHOR" and "BOLT", parse it.

    Extract from that field:
      - diameter → e.g. "5/8\" DIA" → material_size = ROD5/8
      - spacing  → e.g. "@ 16\" O.C." → spacing_in = 16
      - embed    → e.g. "MIN EMBED 4\"" → bolt_length_ft = 4/12

    Formula:
      wall_length_in = wall_length_ft × 12
      bolt_count_per_wall = ceil(wall_length_in / spacing_in) + 1
      total_bolts = bolt_count_per_wall × N
      total_linear_feet = total_bolts × bolt_length_ft
      total_holes = total_bolts × 2
      total_weld_inches = 0

    ─────────────────────────────────────────
    CASE C — BEAMS (W-shape or HSS horizontal)
    ─────────────────────────────────────────

    Use visible span dimension or grid-to-grid distance.
      total_linear_feet = span_ft × beam_count × N
      quantity = beam_count × N

    Fabrication from linked_definition.fabrication if present, else 0.

    ─────────────────────────────────────────
    CASE D — COLUMNS (HSS or W vertical)
    ─────────────────────────────────────────

    Count column instances on plan.
    Height = Top of Steel elevation − Base elevation (read from plan notes or elevation tag).
      total_linear_feet = height_ft × column_count × N
      quantity = column_count × N

    ─────────────────────────────────────────
    CASE E — LINTELS (angle over opening)
    ─────────────────────────────────────────

    Find R.O. (Rough Opening) dimension near symbol.
      lintel_length_ft = RO_width_ft + 1.33  (standard 8" bearing each side)
      total_linear_feet = lintel_length_ft × 2 × N  (two angles per lintel, unless noted otherwise)
      quantity = 2 × N

    ─────────────────────────────────────────
    CASE F — PIPE & PIPING COMPONENTS
    ─────────────────────────────────────────

    PIPE (piece_length_ft is a number):
      total_linear_feet = piece_length_ft × N
      quantity = N

    PIPE (piece_length_ft is null / "VIF" / "AS REQ'D"):
      total_linear_feet = 0
      logic_trace: "Spool length field measure required"

    FITTINGS (material_type = "FITTING"):
      total_linear_feet = 0  (count-based only)
      quantity = fitting_count_per_detail × N

    SEEP RINGS (item_name contains "SEEP RING"):
      pipe_OD_in = diameter from item_name (e.g. 8 for 8" pipe)
      circumference_ft = pipe_OD_in × 3.14159 / 12
      total_linear_feet = circumference_ft × penetration_count × N
      quantity = penetration_count × N

    ─────────────────────────────────────────
    CASE G — LADDER
    ─────────────────────────────────────────

    Read ladder_height_ft from plan (elevation difference or dimension note near symbol).

    RAILS (material_type = "C", "MC", or item has "RAIL" in notes):
      quantity = 2 × N
      total_linear_feet = ladder_height_ft × 2 × N

    RUNGS (material_type = "ROD", qty_rule contains "spacing @ 12\" O.C."):
      rung_count = ceil(ladder_height_ft × 12 / 12) + 1 = ceil(ladder_height_ft) + 1
      rung_width_ft = piece_length_ft  (from BOM, default 1.5 if null)
      total_linear_feet = rung_count × rung_width_ft × N
      quantity = rung_count × N

    CONNECTION PLATES / BASE ANGLES:
      quantity = 4 × N  (2 top + 2 bottom per ladder)

    Fabrication: assign bolt_count, hole_count, weld_inches from linked_definition.fabrication × N
    to the RAILS item (primary member).

    ─────────────────────────────────────────
    CASE H — STAIRS
    ─────────────────────────────────────────

    Read stair_rise_ft and stair_run_ft from plan.

    STRINGERS (C or MC channel):
      stringer_length_ft = sqrt(stair_rise_ft² + stair_run_ft²)
      quantity = 2 × N
      total_linear_feet = stringer_length_ft × 2 × N

    GUARDRAIL / HANDRAIL (pipe):
      total_linear_feet = (stair_run_ft + 2.0) × N  (1'-0" extension each end)
      quantity = N

    ─────────────────────────────────────────
    CASE I — FINISH TREATMENT
    ─────────────────────────────────────────

    For every BOM item, check the notes field from linked_definition.BOM:
      "GALVANIZED" or "GALV" or "HDG" → append "(GALV)" to description, note in logic_trace
      "SS316" or "STAINLESS"          → append "(SS 316)" to description, note "SS 316 rates apply"
      "COATED"                        → append "(COATED)" to description
      "RAW" or nothing                → no suffix needed

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    STEP 6 — MATERIAL NAME NORMALIZATION
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    • If material_size matches a value in VALID MATERIALS LIST → use it exactly (uppercase).
    • If NOT in list → KEEP the original size as written. Do NOT discard.
      Set: lb_per_ft = null, total_weight_lbs = null, charge_per_lb = null
      Note in logic_trace: "Not in valid materials list — kept as-is"
    • NEVER remove a material for a mismatch.

    SS PREFIX:
      "W8x13 SS" or "SS W8x13" → material_size = "W8X13", description includes "(SS 316)"
      Strip "SS" from the shape name before matching. "SS" is a finish flag, not a shape type.
    
    CRITICAL: PL items from linked_definition.BOM must keep their exact item_name as material_size.
    Do NOT normalize PL1/4X4X10 to FB or any other type.
    PL = Plate. FB = Flat Bar. These are different products.
    If the item_name starts with PL → material_size must start with PL.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    STEP 7 — AGGREGATION
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Group rows with IDENTICAL material_size and description together:
      • Sum total_linear_feet
      • Sum quantity
      • Sum total_bolts
      • Sum total_holes
      • Sum total_weld_inches

    If two items have the same material_size but different descriptions (e.g., "Rail at Ladder A"
    vs "Rail at Ladder B") → keep as separate rows for traceability.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    STEP 8 — LOGIC TRACE (mandatory for every row)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    For every BOM row, write a concise logic_trace that explains:
      • Which symbol triggered it (e.g., "Symbol 3/S3-01 × 2 occurrences")
      • Which dimension or elevation was used
      • Which formula was applied
      • The exact arithmetic that produced quantity, total_linear_feet, total_bolts, total_holes, total_weld_inches

    Example:
      "Symbol 2/S3-01 × 3 occurrences. Ladder height 14'-6\" from plan elevation note.
      Rails: 2 × 14.5ft × 3 = 87 LF, qty=6.
      Fabrication from detail: bolt_count=4, hole_count=8, weld_inches=12.
      Scaled: total_bolts=12, total_holes=24, total_weld_inches=36.0"

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    EXCLUSION LIST — remove these from final BOM
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Do NOT include:
      • Concrete rebar (#3, #4, #5, hoops, dowels, T-bars in concrete)
      • Hardware (screws, hinges, hasps, padlocks, screens, mesh)
      • Manufactured products (guardrail post caps, standard hardware items)
      • Grating (unless fabricated structural steel platform grating)

    INCLUDE ONLY fabricated structural steel:
      W, C, MC, L, HSS, PL, FB, PIPE (structural), ROD (structural tie rods)
      Stair stringers, base plates, closure plates, connection plates
    
    If material_size is a bare type prefix only (W, HSS, PL, L, C, MC, FB, ROD, PIPE) with no dimensions following it 
    — do not output this item. A material with no size specification cannot be ordered or fabricated. 
    If material_size contains no numeric characters at all (no digits) — drop it.
    Drop it entirely and note it was dropped in project_summary.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    OUTPUT FORMAT — STRICT JSON ONLY
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Return EXACTLY this structure. No markdown. No explanation outside JSON.

    {{
      "project_summary": "Estimated structural steel for [sheet description]. [N] symbols processed across [sheet_number]. Primary categories: [list key material types found].",
      "final_bill_of_materials": [
        {{
          "description": "MC6X15.1 Side Rails — Ladder at Grid B-3",
          "material_size": "MC6X15.1",
          "quantity": 6,
          "total_linear_feet": 87.0,
          "total_bolts": 12,
          "total_holes": 24,
          "total_weld_inches": 36.0,
          "logic_trace": "Symbol 2/S3-01 × 3 occurrences. Ladder height 14.5ft from elev note. Rails: 2×14.5×3=87 LF. Fabrication per detail: bolt=4, hole=8, weld=12in. Scaled ×3: bolts=12, holes=24, weld=36in.",
          "source_drawing": "2/S3-01",
          "source_sheet": "S3-01",
          "source_symbol": "2/S3-01"
        }},
        {{
          "description": "ROD3/4 Rungs — Ladder at Grid B-3",
          "material_size": "ROD3/4",
          "quantity": 45,
          "total_linear_feet": 67.5,
          "total_bolts": 0,
          "total_holes": 0,
          "total_weld_inches": 0.0,
          "logic_trace": "Symbol 2/S3-01 × 3 occurrences. Height 14.5ft → ceil(14.5)+1=16 rungs each. Width 1.5ft. LF: 16×1.5×3=72. Qty: 16×3=48. Fabrication on primary member (rails).",
          "source_drawing": "2/S3-01",
          "source_sheet": "S3-01",
          "source_symbol": "2/S3-01"
        }},
        {{
          "description": "5/8\" DIA. ANCHOR ROD — Shear Wall hex-1",
          "material_size": "ROD5/8",
          "quantity": 22,
          "total_linear_feet": 33.0,
          "total_bolts": 22,
          "total_holes": 44,
          "total_weld_inches": 0.0,
          "logic_trace": "hex-1 shear wall × 2 occurrences. Wall 13'-10\" each. Spacing 16\" OC. Each wall: ceil(166/16)+1=11 bolts. Total: 11×2=22. LF: 22×1.5=33. Holes: 22×2=44. No weld on anchor rods.",
          "source_drawing": "hex-1",
          "source_sheet": "{sheet_number}",
          "source_symbol": "hex-1"
        }}
      ]
    }}

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    FINAL VALIDATION CHECKLIST — run before returning
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    For every row in final_bill_of_materials:
      [ ] total_linear_feet is a number (not null, not string) — 0 only if truly unknown
      [ ] quantity is an integer ≥ 1
      [ ] total_bolts is populated if linked_definition.fabrication.bolt_count > 0
      [ ] total_holes is populated — NEVER 0 if total_bolts > 0 (apply 2× minimum)
      [ ] total_weld_inches is populated if linked_definition.fabrication.weld_inches > 0
      [ ] logic_trace shows the symbol name, occurrence count N, dimension source, and all arithmetic
      [ ] source_sheet and source_symbol are both filled
      [ ] material_size does not contain "SS " prefix (strip it, put in description)
      [ ] No concrete, rebar, or hardware items included

    STRICT RULES:
      • No markdown in output
      • No explanation outside the JSON
      • No hallucinated materials
      • All math must be shown in logic_trace
      • Use decimal feet for all lengths
      • Round bolt counts UP (ceiling)
      • Preserve fractions in material names

    You are executing structural estimation logic. Not summarizing. Not guessing. Compute precisely.
"""
    return prompt


def SYMBOL_OCR_PROMPT():
    return """
You are reading a structural engineering drawing callout symbol.

There are only two valid symbol types:

1) HEXAGON with a number inside:
   Output exactly: hex-N
   Examples: hex-1, hex-42

2) CIRCLE with a number inside:
   Output exactly: cir-N
   Examples: cir-1, cir-7

2) DETAIL CALLOUT — circle on top of triangle with:
   - Top half: a number or alphanumeric label
   - Bottom half: a sheet reference

   Output exactly: LABEL/SHEET

   The sheet reference can look many different ways:
     S-3.2    S3-01    S5-07    ST-DT-0003    S-4.0
   
   IMPORTANT: After the slash, if you see a digit that looks
   like 5 at the start of a sheet number, it is likely the
   letter S — read it as S, not 5.

   Real examples from structural drawings:
     3/S-3.2
     4/S-4.0
     2/S3-01
     7A/S5-05
     4/ST-DT-0003
     1A/S5-01

Rules:
- NO spaces, NO newlines, NO explanation, NO markdown
- If you cannot confidently read it: output Unknown
- Valid outputs: hex-N  or  LABEL/SHEET  or  Unknown
- Never write "S5" when the actual text is just "S" followed by a digit like "3". 
  Example: "2/S3-01" must NEVER become "2/S53-01" — the 5 does not exist.
"""


def prompt_bom_validator(bom_json: str) -> str:
    return f"""
You are a structural steel estimator reviewing a raw Bill of Materials for quality control.

Your job: review each item and decide KEEP or DROP.

DROP an item if it meets ANY ONE of these conditions:

1. DUPLICATE — exact same material_size + quantity + source_symbol appears elsewhere.
   Keep the symbol-detected version (cir-N, hex-N, detail ref), drop "Schedule:" version.

2. NON-STRUCTURAL — item is CLEARLY one of:
   - Concrete rebar: #3, #4, #5, #6 bars, DBA BAR, hoops, dowels
   - Insulation, ceramic fiber, wood, concrete products
   - Manufactured hardware: screws, hinges, padlocks, mesh, screens
   - Non-fabricated grating

3. HALLUCINATION — material_size contains zero numeric characters AND
   description gives no real spec.

NEVER DROP these regardless of context:
- L shapes (angles) — always structural connection hardware
- PL (plates) — always structural
- W, C, MC, HSS, FB shapes — always structural
- ROD with dimensions (e.g. ROD3/4) — structural tie rod
- Any item with a valid detail ref source_symbol (e.g. 7/S-3.2)

KEEP everything else including items flagged "not in valid materials list".

For duplicates: keep symbol-detected version (cir-N, hex-N, detail ref),
drop the schedule-scan version (source_symbol starts with "Schedule:").

RAW BOM (0-indexed):
{bom_json}

Return your decision for EVERY item. Index must match exactly.
"""