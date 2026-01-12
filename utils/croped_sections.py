from langgraph.graph import StateGraph, START, END
import base64
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from typing import TypedDict, List
from dotenv import load_dotenv
import os
import fitz
import pdfplumber
from pdfplumber import open as pdf_open
from PIL import Image, ImageDraw, ImageFont
from utils.pdf_page_to_png import convert_specific_page_to_png
from utils.pdf_create import extracte_pdf
from pydantic import BaseModel
from typing import Literal
import cv2 
from PIL import Image
import pdfplumber
import cv2

load_dotenv()

llm=ChatGoogleGenerativeAI(
        model='gemini-2.5-pro'
)

class TitleChoice(BaseModel):
    choice: int  

def load_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def scale_coords_pdf_to_image(coords_dict, pdf_path, image_path):
    img = Image.open(image_path)
    img_width, img_height = img.size

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        pdf_width = page.width
        pdf_height = page.height

    scale_x = img_width / pdf_width
    scale_y = img_height / pdf_height

    scaled = {}
    for title, c in coords_dict.items():
        scaled[title] = {
            "x1": int(c["x1"] * scale_x),
            "y1": int(c["y1"] * scale_y),
            "x2": int(c["x2"] * scale_x),
            "y2": int(c["y2"] * scale_y),
        }

    return scaled


def extract_text_from_gemini_response(response) -> str:
    """
    Safely extract text from Gemini / LangChain response.
    Handles string and multimodal list formats.
    """
    if isinstance(response.content, list):
        return " ".join(
            part.get("text", "")
            for part in response.content
            if isinstance(part, dict)
        ).strip()
    return str(response.content).strip()


def find_title(imgURL: str):
    image_base64 = load_image_base64(imgURL)

    prompt = """
You are looking at an architectural / structural drawing sheet.
Task:
- Extract ONLY the section titles present in the drawing Sheet.
Rules:
- Titles are bold, uppercase, and placed below each detail.
- Ignore dimensions, notes, callouts, and material labels.
- Do NOT explain anything.
- Do NOT guess if you cannot see.
- Return each title on new line as it is.
"""

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                },
            },
        ]
    )

    response = llm.invoke([message])


    if isinstance(response.content, list):
        text = "\n".join(
            part["text"]
            for part in response.content
            if isinstance(part, dict) and "text" in part
        )
    else:
        text = response.content

    titles = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]

    return titles
def disambiguate_repeated_titles(image_path, title_coords_candidates):
    final_coords = {}

    for title, candidates in title_coords_candidates.items():
        if len(candidates) == 1:
            final_coords[title] = candidates[0]
            continue

        candidate_str = "\n".join(
            [f"{i+1}: x1={c['x1']}, y1={c['y1']}, x2={c['x2']}, y2={c['y2']}"
             for i, c in enumerate(candidates)]
        )

        prompt = f"""
You are looking at an architectural drawing sheet (image provided).

The title "{title}" appears multiple times.
Select the correct candidate.

Candidates:
{candidate_str}

Rules:
- Return ONLY a JSON object
- The number MUST be between 1 and {len(candidates)}

Example:
{{ "choice": 2 }}
"""

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{load_image_base64(image_path)}"
                    },
                },
            ]
        )

        # 🔒 Schema-enforced invoke
        response = llm.with_structured_output(TitleChoice).invoke([message])

        choice = response.choice - 1  # convert to 0-based

        # Absolute safety (should never trigger now)
        if not (0 <= choice < len(candidates)):
            choice = min(range(len(candidates)), key=lambda i: candidates[i]["y1"])

        final_coords[title] = candidates[choice]

    return final_coords


def extract_words_with_coords(pdf_path, page_num):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num - 1]
        return page.extract_words(
            use_text_flow=True,
            keep_blank_chars=False
        )

def find_all_title_coordinates(words, titles):
    results = {}  # title -> list of coordinates
    word_texts = [w["text"].upper() for w in words]

    for title in titles:
        title_words = title.upper().split()
        n = len(title_words)
        candidates = []

        for i in range(len(word_texts) - n + 1):
            if word_texts[i:i+n] == title_words:
                boxes = words[i:i+n]
                x0 = min(w["x0"] for w in boxes)
                x1 = max(w["x1"] for w in boxes)
                top = min(w["top"] for w in boxes)
                bottom = max(w["bottom"] for w in boxes)

                candidates.append({
                    "x1": x0,
                    "y1": top,
                    "x2": x1,
                    "y2": bottom
                })

        if candidates:
            results[title] = candidates

    return results



def find_title_coordinates_from_image_and_pdf(pdf_path):
    results = {}
    os.makedirs("pages", exist_ok=True)
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):

            # 1. Render page image for LLM
            image_path = f"pages/page_{page_num}.png"
            page.to_image(resolution=300).save(image_path)

            # 2. LLM reads titles from image
            titles = find_title(image_path)   # List[str]

            # 3. Extract PDF words + coords
            words = page.extract_words(
                use_text_flow=True,
                keep_blank_chars=False
            )

            # 4. Collect all candidate coordinates for each title
            title_coords_candidates = find_all_title_coordinates(words, titles)

            # 5. Resolve duplicates using Gemini if needed
            final_coords = disambiguate_repeated_titles(image_path, title_coords_candidates)

            results[f"page_{page_num}"] = final_coords

    return results


def draw_bounding_boxes_on_image(
    image_path: str,
    pdf_path: str,
    coords_dict: dict,
    outline_color: str = "red",
    line_width: int = 4
):
    """
    Scales PDF-space coordinates to image-space and draws bounding boxes
    on the image. Overwrites the same image.
    """

    # Load image
    img = Image.open(image_path).convert("RGB")
    img_width, img_height = img.size
    draw = ImageDraw.Draw(img)

    # Read PDF page size (single-page PDF)
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        pdf_width = page.width
        pdf_height = page.height

    # Scaling factors
    scale_x = img_width / pdf_width
    scale_y = img_height / pdf_height

    # Draw boxes
    for title, c in coords_dict.items():
        x1 = int(c["x1"] * scale_x)
        y1 = int(c["y1"] * scale_y)
        x2 = int(c["x2"] * scale_x)
        y2 = int(c["y2"] * scale_y)

        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline=outline_color,
            width=line_width
        )

    img.save(image_path)
# 
def crop_sections_from_page(
    coords_dict: dict,          # PDF-space coords
    page_image_path: str,
    pdf_path: str,
    page_name: str,
    base_output_dir: str = "cropped_sections"
):
    """
    Crops architectural / structural sections from a page image
    using title coordinates extracted from PDF.

    FIX:
    - Convert PDF coords → image coords
    - Use ONLY image-space (scaled) coords for cropping
    """

    # ---------------------------
    # Create per-page output dir
    # ---------------------------
    page_output_dir = os.path.join(base_output_dir, page_name)
    os.makedirs(page_output_dir, exist_ok=True)

    # ---------------------------
    # Load image
    # ---------------------------
    page_image = Image.open(page_image_path)
    img_width, img_height = page_image.size

    # ---------------------------
    # Scale coordinates PDF → Image
    # ---------------------------
    scaled = scale_coords_pdf_to_image(
        coords_dict,
        pdf_path,
        page_image_path
    )

    # ---------------------------
    # Row-wise cropping (by y2) — IMAGE SPACE
    # ---------------------------
    cropped_sections = []

    rows_y2 = sorted(set(c["y2"] for c in scaled.values()))
    prev_y = 0

    for row_idx, y2 in enumerate(rows_y2, start=1):

        if y2 <= prev_y:
            continue

        cropped_row = page_image.crop((0, prev_y, img_width, y2))

        # Titles in this row (IMAGE SPACE)
        titles_in_row = [
            t for t, c in scaled.items() if c["y2"] == y2
        ]

        # Sort left → right (IMAGE SPACE)
        titles_in_row_sorted = sorted(
            titles_in_row,
            key=lambda t: scaled[t]["x1"]
        )

        left_margin = 400
        right_margin = 200

        for col_idx, title in enumerate(titles_in_row_sorted):

            curr_x1 = scaled[title]["x1"]
            x_start = max(0, curr_x1 - left_margin)

            if col_idx < len(titles_in_row_sorted) - 1:
                next_title = titles_in_row_sorted[col_idx + 1]
                next_x1 = scaled[next_title]["x1"]
                x_end = max(x_start + 1, next_x1 - right_margin)
            else:
                x_end = img_width

            if x_end <= x_start:
                continue

            cropped_section = cropped_row.crop(
                (x_start, 0, x_end, cropped_row.height)
            )

            safe_title = title.replace("/", "_").replace(" ", "_")
            save_path = os.path.join(
                page_output_dir,
                f"{safe_title}.png"
            )

            cropped_section.save(save_path)

            cropped_sections.append({
                "title": title,
                "image_path": save_path,
                "coords": (x_start, prev_y, x_end, y2)
            })

        prev_y = y2 + 110  # spacing buffer

    return cropped_sections



    
# --- Run the pipeline ---

if __name__=="__main__":
        pdf_path = "utils/all_section.pdf"
        output_path='section'
        try :
          if not os.path.exists(output_path):
              os.mkdir(output_path)
          doc=fitz.open(pdf_path)
          for i,page in enumerate(doc):
              page_image_path=f'section/page_{i}.png'
              page_section_pdf=f'section/page_{i}.pdf'
              extracte_pdf(pdf_path,page_section_pdf,i+1,i+1)
              convert_specific_page_to_png(page_section_pdf,0,page_image_path)
              coords_dict = find_title_coordinates_from_image_and_pdf(page_section_pdf)
              page_key = list(coords_dict.keys())[0]
              scale_coords_dict=scale_coords_pdf_to_image(coords_dict[page_key],page_section_pdf,page_image_path)
              print(scale_coords_dict)
              print("-------")
              draw_bounding_boxes_on_image(page_image_path,page_section_pdf,coords_dict[page_key])
              crop_sections_from_page(coords_dict[page_key],page_image_path,page_section_pdf,f"page_{i}",)
        except Exception as e:
         raise ValueError(e)



#      SCALED_COORDS = [
#     {
#         "page_name": "page_0",
#         ""
#         "image_path": "section/page_0.png",
#         "coords": {
#             "TYP. REINF. ARRANGMENTS AT CORNERS": {"x1": 853, "y1": 2094, "x2": 3869, "y2": 2198},
#             "TYP. STEPPED FOOTING": {"x1": 6990, "y1": 2094, "x2": 8734, "y2": 2198},
#             "TYP. UTILITIES UNDER SLAB OR WALL FOOTING": {"x1": 853, "y1": 3844, "x2": 3180, "y2": 4094},
#             "TYP. COLUMN ISOLATION JOINTS": {"x1": 3922, "y1": 3990, "x2": 6361, "y2": 4094},
#             "TYP. FOUNDATION INFLUENCE DETAIL": {"x1": 6990, "y1": 3844, "x2": 9218, "y2": 4094},
#             "TYP. JOINTS IN SLAB ON GRADE": {"x1": 853, "y1": 6764, "x2": 3201, "y2": 6868},
#             "TYP. EXCAVATIONS PARALLEL TO WALL FOOTING": {"x1": 3922, "y1": 6618, "x2": 6384, "y2": 6868},
#             "TYP. INTERIOR FOOTING": {"x1": 6990, "y1": 6764, "x2": 8777, "y2": 6868},
#         }
#     },
#     {
#         "page_name": "page_1",
#         "image_path": "section/page_1.png",
#         "coords": {
#             "HOLDOWN SCHEDULE": {"x1": 851, "y1": 3993, "x2": 2431, "y2": 4097},
#             "TYP. SLAB AT FLOOR DRAIN": {"x1": 7013, "y1": 3993, "x2": 9043, "y2": 4097},
#             "TYP. CONCRETE SLAB DETAILS": {"x1": 851, "y1": 6765, "x2": 3112, "y2": 6869},
#             "TYP. TRENCH DETAIL": {"x1": 3943, "y1": 6765, "x2": 5490, "y2": 6869},
#             "TYP. DEPRESSED SLAB ON GRADE": {"x1": 7013, "y1": 6765, "x2": 9482, "y2": 6869},
#         }
#     },
#     {
#         "page_name": "page_2",
#         "image_path": "section/page_2.png",
#         "coords": {
#             "TYP. EXT. WALL FOOTING": {"x1": 867, "y1": 2627, "x2": 2726, "y2": 2731},
#             "WALL FOOTING @ HOLD DOWN": {"x1": 3943, "y1": 2627, "x2": 6129, "y2": 2731},
#             "WALL FOOTING @ SHEAR WALL": {"x1": 6996, "y1": 2627, "x2": 9232, "y2": 2731},
#             "WALL FOOTING AT FURRING WALL": {"x1": 867, "y1": 4852, "x2": 3369, "y2": 4956},
#             "FOUNDATION @ INT. SHEAR WALL": {"x1": 3943, "y1": 4852, "x2": 6380, "y2": 4956},
#             "TYP. EXT. FOOTING AT PARTIAL SLAB": {"x1": 6996, "y1": 4706, "x2": 9314, "y2": 4956},
#             "LADDER DETAIL": {"x1": 6996, "y1": 6737, "x2": 8147, "y2": 6841},
#         }
#     },
#     {
#         "page_name": "page_3",
#         "image_path": "section/page_3.png",
#         "coords": {
#             "TYP. OPENING ELEVATION": {"x1": 813, "y1": 2197, "x2": 2737, "y2": 2301},
#             "BUILT UP POST": {"x1": 4674, "y1": 2197, "x2": 5792, "y2": 2301},
#             "TYP. NON-LOAD BEARING WALL": {"x1": 6965, "y1": 2197, "x2": 9255, "y2": 2301},
#             "TYP. SHEAR WALL ELEVATION": {"x1": 813, "y1": 4482, "x2": 3010, "y2": 4586},
#             "SECTION AT CANOPY": {"x1": 4067, "y1": 4482, "x2": 5609, "y2": 4586},
#             "BUILT UP BEAM NAILING PATTERN": {"x1": 6965, "y1": 4482, "x2": 9494, "y2": 4586},
#             "CORNER FRAMING DETAIL": {"x1": 813, "y1": 6762, "x2": 2714, "y2": 6866},
#             "ROOF DIAPHRAGM NAILING": {"x1": 3144, "y1": 6762, "x2": 5134, "y2": 6866},
#             "TYP. TOP PLATE SPLICE": {"x1": 5576, "y1": 6762, "x2": 7320, "y2": 6866},
#             "TYP. SPLICE SECTION": {"x1": 7693, "y1": 6762, "x2": 9278, "y2": 6866},
#         }
#     },
#     {
#         "page_name": "page_4",
#         "image_path": "section/page_4.png",
#         "coords": {
#             "TYP. WALL SECTION": {"x1": 792, "y1": 1989, "x2": 2261, "y2": 2093},
#             "TYP. WALL SECTION @ GIRDER": {"x1": 3984, "y1": 1989, "x2": 6190, "y2": 2093},
#             "SECTION @ INT. SHEAR WALL": {"x1": 6935, "y1": 1989, "x2": 9047, "y2": 2093},
#             "SECTION @ FURRING WALL": {"x1": 792, "y1": 4147, "x2": 2733, "y2": 4251},
#             "SECTION @ ENTRANCE": {"x1": 3984, "y1": 4147, "x2": 5615, "y2": 4251},
#             "SECTION @ LIGHT": {"x1": 792, "y1": 5849, "x2": 2071, "y2": 5953},
#             "TYP. INT. TRUSS BRG.": {"x1": 3162, "y1": 5849, "x2": 4760, "y2": 5953},
#             "TYP. END COLUMN": {"x1": 5549, "y1": 5849, "x2": 6903, "y2": 5953},
#             "SECTION @ ROOF HATCH": {"x1": 7746, "y1": 5849, "x2": 9534, "y2": 5953},
#             "MECH UNIT SUPPORT": {"x1": 5549, "y1": 6769, "x2": 7114, "y2": 6873},
#         }
#     },
#     {
#         "page_name": "page_5",
#         "image_path": "section/page_5.png",
#         "coords": {
#             "DUMPSTER ENCLOSURE FOUNDATION PLAN": {"x1": 812, "y1": 2982, "x2": 2557, "y2": 3232},
#             "DUMPSTER ENCLOSURE WALL": {"x1": 3500, "y1": 3128, "x2": 5690, "y2": 3232},
#             "DUMPSTER SLAB EDGE": {"x1": 6953, "y1": 3128, "x2": 8625, "y2": 3232},
#             "TYP. DETAIL OF LOW-LIFT REINFORCED CMU": {"x1": 3500, "y1": 6579, "x2": 5367, "y2": 6829},
#             "TYPICAL CMU WALL CORNERS AND INTERSECTIONS": {"x1": 6953, "y1": 6579, "x2": 9503, "y2": 6829},
#         }
#     },
# ]
#      for page in SCALED_COORDS:
#         crop_sections_from_page(
#         coords_dict=page["coords"],
#         page_image_path=page["image_path"],
#         page_name=page["page_name"]
#     )

# # Load page image
 # your exported image




# cropped_sections = crop_sections_from_page(
#     coords_dict['page_0'],
#     page_image_path,
#     pdf_path
# )

# for cs in cropped_sections:
#     print(cs["title"], cs["image_path"], cs["coords"])







# coords_dict = {
#     'page_0':
#     {
#     'TYP. OPENING ELEVATION': {'x1': 195.2399, 'y1': 527.37646, 'x2': 657.0025, 'y2': 552.2964},
#     'BUILT UP POST': {'x1': 1121.8796, 'y1': 527.37646, 'x2': 1390.09, 'y2': 552.2964},
#     'TYP. NON-LOAD BEARING WALL': {'x1': 1671.7193, 'y1': 527.37646, 'x2': 2221.33, 'y2': 552.2964},
#     'TYP. SHEAR WALL ELEVATION': {'x1': 195.2399, 'y1': 1075.77626, 'x2': 722.4038, 'y2': 1100.6962},
#     'SECTION AT CANOPY': {'x1': 976.3196, 'y1': 1075.77626, 'x2': 1346.25, 'y2': 1100.6962},
#     'BUILT UP BEAM NAILING PATTERN': {'x1': 1671.7193, 'y1': 1075.77626, 'x2': 2278.645, 'y2': 1100.6962},
#     'CORNER FRAMING DETAIL': {'x1': 195.2399, 'y1': 1623.09596, 'x2': 651.5738, 'y2': 1648.01596},
#     'ROOF DIAPHRAGM NAILING': {'x1': 754.7997, 'y1': 1623.09596, 'x2': 1232.2646, 'y2': 1648.01596},
#     'TYP. TOP PLATE SPLICE': {'x1': 1338.4795, 'y1': 1623.09596, 'x2': 1756.8379, 'y2': 1648.01596},
#     'TYP. SPLICE SECTION': {'x1': 1846.5593, 'y1': 1623.09596, 'x2': 2226.8408, 'y2': 1648.01596}
# },
# }