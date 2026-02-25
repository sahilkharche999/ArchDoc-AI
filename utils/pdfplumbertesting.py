# import pdfplumber
# import os

# def extract_pdf_content(pdf_path, output_folder="pdf_output"):
#     os.makedirs(output_folder, exist_ok=True)

#     all_text = ""

#     with pdfplumber.open(pdf_path) as pdf:
#         print(f"Total pages: {len(pdf.pages)}")

#         for page_number, page in enumerate(pdf.pages, start=1):
#             print(f"\nProcessing Page {page_number}")

#             # ---- Extract Text ----
#             text = page.extract_text()
#             if text:
#                 all_text += f"\n\n--- Page {page_number} ---\n"
#                 all_text += text
#                 print("Text extracted.")

#             # ---- Extract Tables ----
#             tables = page.extract_tables()
#             if tables:
#                 for table_index, table in enumerate(tables, start=1):
#                     table_file = os.path.join(
#                         output_folder,
#                         f"page_{page_number}_table_{table_index}.csv"
#                     )

#                     with open(table_file, "w", encoding="utf-8") as f:
#                         for row in table:
#                             row = [str(cell) if cell else "" for cell in row]
#                             f.write(",".join(row) + "\n")

#                     print(f"Saved table: {table_file}")

#             # ---- Extract Images ----
#             for img_index, img in enumerate(page.images, start=1):
#                 x0, top, x1, bottom = (
#                     img["x0"], img["top"], img["x1"], img["bottom"]
#                 )

#                 # Crop image from page
#                 cropped = page.within_bbox((x0, top, x1, bottom))
#                 image_obj = cropped.to_image(resolution=300)

#                 image_path = os.path.join(
#                     output_folder,
#                     f"page_{page_number}_image_{img_index}.png"
#                 )
#                 image_obj.save(image_path)
#                 print(f"Saved image: {image_path}")

#     # ---- Save All Text ----
#     text_file = os.path.join(output_folder, "extracted_text.txt")
#     with open(text_file, "w", encoding="utf-8") as f:
#         f.write(all_text)

#     print(f"\nAll text saved to: {text_file}")
#     print("Extraction complete.")


# # 🔹 Example usage
# pdf_path = "S302.pdf"  # change this to your PDF path
# extract_pdf_content(pdf_path)

# import camelot

# tables = camelot.read_pdf("S302.pdf", pages="all")

# print(f"Total tables found: {len(tables)}\n")

# for i, table in enumerate(tables, start=1):
#     print(f"\n========== Table {i} ==========\n")
    
#     # table.df is already a pandas DataFrame
#     print(table.df.to_string(index=False))  # Clean formatted print

# print("\nDone extracting tables")



import pdfplumber
import pandas as pd

pdf_path ="S302.pdf" 

table_settings = {
    "vertical_strategy": "lines",        # detect vertical lines
    "horizontal_strategy": "lines",      # detect horizontal lines
    "intersection_tolerance": 5,
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "edge_min_length": 3,
    "min_words_vertical": 1,
    "min_words_horizontal": 1,
}

with pdfplumber.open(pdf_path) as pdf:
    for page_number, page in enumerate(pdf.pages, start=1):
        
        tables = page.extract_tables(table_settings)

        print(f"\n===== Page {page_number} =====")
        print(f"Tables found: {len(tables)}")

        for i, table in enumerate(tables, start=1):
            
            # Convert to DataFrame
            df = pd.DataFrame(table)

            # Clean empty rows
            df = df.dropna(how="all")
            df = df.fillna("")

            print(f"\n--- Table {i} ---\n")
            print(df.to_string(index=False))
