# from groq import Groq
# import base64
# import os
# from dotenv import load_dotenv
# load_dotenv() 
# # Function to encode the image
# def encode_image(image_path):
#   with open(image_path, "rb") as image_file:
#     return base64.b64encode(image_file.read()).decode('utf-8')

# # Path to your image
# image_path = "output_temp/section_page_5/auto/images/0d2108753e3f2eec76a40dc910e3a6c0a06ef8f254ff0f381627a80dc36b5795.jpg"

# # Getting the base64 string
# base64_image = encode_image(image_path)

# client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": "Extract the dimentions and respected elements like wf, hss ,etc. we want to do the steel estimation using the floor plan . if you see the section detail symbol example 4/s-3.2 than do extract too. if you see other symbol like hexagon 1 then do extract that to . give me output in json format"},
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"data:image/jpeg;base64,{base64_image}",
#                     },
#                 },
#             ],
#         }
#     ],
#     model="meta-llama/llama-4-scout-17b-16e-instruct",
# )

# print(chat_completion.choices[0].message.content)


# # import os
# # import json
# # from google import genai
# # from google.genai import types
# # from dotenv import load_dotenv

# # load_dotenv()

# # client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
# # MODEL = "gemini-2.5-flash" 

# # def extract_detail_components_with_crops(pdf_layout_path: str,json_path:str,images_dir:str):
    

# #     # 2. Load JSON (Text Data)
# #     with open(json_path, 'r') as f:
# #         json_data = json.load(f)
# #         # Filter for text only to keep prompt clean
# #         text_data = [item for item in json_data if item["type"] == "text"]
# #         json_string = json.dumps(text_data, indent=2)

# #     # 3. Prepare Content List for Gemini
# #     contents_payload = []

# #     # A. Add the Prompt
   
# #     prompt = f"""
# #     You are a Senior Structural Detailer creating a "Standard Definition Library".
    
# #     ### INPUTS PROVIDED:
# #     1. **Layout PDF:** Shows the full page structure.
# #     2. **Cropped Images:** High-resolution zooms. Filenames are provided.
# #     3. **Text JSON:** OCR text found on the page.

# #     ### YOUR GOAL
# #     Visually group the inputs into distinct **"Detail Units"** and create a **Recipe Card** for each one.

# #     ---
# #     ### MULTIMODAL CHAIN-OF-THOUGHT PROCESS:

# #     **STEP 1: IDENTIFY & GROUP (The Trace)**
# #     - Look at the Layout PDF. Find a Title.
# #     - Find the Cropped Image that matches this Title.
# #     - *Trace:* "I matched Title 'LADDER DETAIL' to Image 'crop_005.jpg' because it is located directly above the text."

# #     **STEP 2: EXTRACT INGREDIENTS (Verbatim)**
# #     - Read the text on the leader lines.
# #     - **CRITICAL RULE:** Extract the material name **EXACTLY AS WRITTEN** on the drawing.
# #         - *Bad:* "Angle 4x4"
# #         - *Good:* "L4x4x1/4"
# #         - *Bad:* "3/4 inch Rod"
# #         - *Good:* "3/4\" DIA. ROD"
# #     - Do not expand abbreviations. Do not convert units.

# #     **STEP 3: DEFINE LOGIC (Fixed vs Variable)**
# #     - Decide if the item count is constant (Fixed) or depends on height/width (Variable).

# #     ---
# #     ### OUTPUT FORMAT (JSON List)
# #     [
# #       {{
# #         "detail_id": "7/S-3.2",
# #         "title": "LADDER DETAIL",
# #         "source_trace": "Matched Title at [x,y] to Image 'crop_005.jpg'",
# #         "visual_reasoning": "I identified this as a Ladder. It contains fixed clips and variable rails.",
# #         "materials": [
# #           {{
# #             "item_name": "L4x4x1/4x0'-3\"",  <-- EXACT TEXT FROM PDF
# #             "material_type": "L",
# #             "qty_rule": "FIXED: 2",
# #             "notes": "Base connection clips"
# #           }}
# #         ],
# #         "fabrication": {{ ... }}
# #       }}
# #     ]
# #     """
# #     contents_payload.append(types.Part.from_text(text=prompt))

# #     # B. Add the Layout PDF (Global Context)
# #     with open(pdf_layout_path, "rb") as f:
# #         contents_payload.append(types.Part.from_bytes(
# #             data=f.read(),
# #             mime_type='application/pdf'
# #         ))

# #     # C. Add All Cropped Images (Local Precision)
# #     if os.path.exists(images_dir):
# #         image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
# #         print(f"Loading {len(image_files)} cropped images...")
        
# #         for img_file in image_files:
# #             img_path = os.path.join(images_dir, img_file)
# #             with open(img_path, "rb") as f:
# #                 # We add the filename as text context so the model knows which image is which
# #                 contents_payload.append(types.Part.from_text(text=f"Image File: {img_file}"))
# #                 contents_payload.append(types.Part.from_bytes(
# #                     data=f.read(),
# #                     mime_type='image/jpeg'
# #                 ))

# #     # D. Add the JSON Text Data
# #     contents_payload.append(types.Part.from_text(text=f"OCR Text Data:\n{json_string}"))

# #     # 4. Call Gemini
# #     print("Sending to Gemini (This may take a moment due to image count)...")
    
# #     response = client.models.generate_content(
# #         model=MODEL,
# #         contents=[types.Content(parts=contents_payload)]
# #     )
    
# #     print("\n--- EXTRACTED LIBRARY DATA ---")
# #     print(response.text)

# # if __name__ == "__main__":
# #     # Update this to your actual output folder

# #     pdf_layout_path = f"output/section_page_5/auto/section_page_5_layout.pdf"
# #     json_path = f"output/section_page_5/auto/section_page_5_content_list.json"
# #     images_dir = f"output/section_page_5/auto/images"
    
# #     if os.path.exists(pdf_layout_path):
# #         extract_detail_components_with_crops(pdf_layout_path,json_path,images_dir)
# #     else:
# #         print("Folder not found.")

# # # import json
# # # import os
# # # from PIL import Image

# # # def crop_union_tables(json_path, image_path, output_dir="debug_crops"):
    
# # #     os.makedirs(output_dir, exist_ok=True)
    
# # #     if not os.path.exists(image_path):
# # #         print(f"Error: Image not found at {image_path}")
# # #         return
    
# # #     full_img = Image.open(image_path)
# # #     img_w, img_h = full_img.size
# # #     print(f"Loaded Image: {img_w}x{img_h}")

# # #     with open(json_path, 'r') as f:
# # #         content_list = json.load(f)
# # #         if isinstance(content_list, list) and len(content_list) > 0 and isinstance(content_list[0], list):
# # #             content_list = content_list[0]

# # #     # --- CALCULATE SCALE FACTOR ---
# # #     # Find the max X and Y in the JSON to determine the PDF size
# # #     max_json_x = 0
# # #     max_json_y = 0
# # #     for item in content_list:
# # #         if item.get("bbox"):
# # #             max_json_x = max(max_json_x, item["bbox"][2])
# # #             max_json_y = max(max_json_y, item["bbox"][3])
    
# # #     # Avoid division by zero
# # #     if max_json_x == 0: max_json_x = 1000 
    
# # #     scale_x = img_w / max_json_x
# # #     scale_y = img_h / max_json_y
    
# # #     print(f"Detected Scale Factor: X={scale_x:.2f}, Y={scale_y:.2f}")

# # #     skip_next = False

# # #     for i, item in enumerate(content_list):
# # #         if skip_next:
# # #             skip_next = False
# # #             continue

# # #         item_type = item.get("type")
# # #         bbox = item.get("bbox") 

# # #         # --- RELAXED CHECK: Allow 'text' to be a title ---
# # #         if item_type in ["title", "text"]:
            
# # #             # Extract text for filename
# # #             try:
# # #                 if item_type == "title":
# # #                     title_text = item["content"]["title_content"][0]["content"]
# # #                 else:
# # #                     title_text = item.get("text", "")
                
# # #                 # Skip long paragraphs (likely not titles)
# # #                 if len(title_text) > 100: continue
                
# # #                 safe_title = "".join(x for x in title_text if x.isalnum() or x == " ")[:30].strip()
# # #             except:
# # #                 safe_title = f"Item_{i}"

# # #             # Look Ahead
# # #             if i + 1 < len(content_list):
# # #                 next_item = content_list[i+1]
# # #                 next_type = next_item.get("type")
# # #                 next_bbox = next_item.get("bbox")

# # #                 # --- RELAXED CHECK: Allow 'image' as table ---
# # #                 if next_type in ["table", "list", "image"]:
                    
# # #                     # Check Vertical Gap
# # #                     gap = next_bbox[1] - bbox[3]
                    
# # #                     # Scale the gap threshold too (e.g. 100 PDF points)
# # #                     if gap < 100:
# # #                         print(f"MATCH! '{safe_title}' -> '{next_type}' (Gap: {gap:.1f})")
                        
# # #                         # --- CALCULATE UNION BOX (PDF Coords) ---
# # #                         union_x1 = min(bbox[0], next_bbox[0]) - 60 # Padding for Symbol
# # #                         union_y1 = bbox[1] - 10
# # #                         union_x2 = max(bbox[2], next_bbox[2]) + 10
# # #                         union_y2 = next_bbox[3] + 10
                        
# # #                         # --- SCALE TO IMAGE PIXELS ---
# # #                         crop_box = (
# # #                             int(union_x1 * scale_x),
# # #                             int(union_y1 * scale_y),
# # #                             int(union_x2 * scale_x),
# # #                             int(union_y2 * scale_y)
# # #                         )
                        
# # #                         # Clamp
# # #                         crop_box = (
# # #                             max(0, crop_box[0]), max(0, crop_box[1]),
# # #                             min(img_w, crop_box[2]), min(img_h, crop_box[3])
# # #                         )

# # #                         # Crop & Save
# # #                         try:
# # #                             crop_img = full_img.crop(crop_box)
# # #                             save_path = os.path.join(output_dir, f"UNION_{safe_title}.png")
# # #                             crop_img.save(save_path)
# # #                             print(f"  -> Saved: {save_path}")
# # #                             skip_next = True
# # #                         except Exception as e:
# # #                             print(f"  ! Crop Failed: {e}")
# # #                     else:
# # #                         pass # print(f"  Gap too large: {gap}")
# # #                 else:
# # #                     pass # print(f"  Next item is {next_type}")

# # # if __name__ == "__main__":
# # #     # json_file = "output/Foundation_plan/vlm/Foundation_plan_content_list_v2.json"
# # #     json_file="output/floor_plan/vlm/floor_plan_content_list_v2.json"
# # #     image_file = "output_temp/floor_1.png"
# # #     crop_union_tables(json_file, image_file)

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# 1. Setup the model you are using
embedder = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# 2. Generate a test vector
vector = embedder.embed_query("Test string")

# 3. Print the size
print(f"Model: gemini-embedding-001")
print(f"Actual Vector Dimension: {len(vector)}")



from langgraph_temp_workflow.common.state import ProjectState
import json
