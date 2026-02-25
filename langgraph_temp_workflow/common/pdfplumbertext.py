from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

def extract_structured_text(pdf_path):
    pages = []

    for page_num, page_layout in enumerate(extract_pages(pdf_path), start=1):
        page_data = {
            "page": page_num,
            "blocks": []
        }

        for element in page_layout:
            if isinstance(element, LTTextContainer):
                block = {
                    "text": element.get_text().strip(),
                    "bbox": element.bbox  # (x0, y0, x1, y1)
                }
                page_data["blocks"].append(block)

        pages.append(page_data)

    return pages

print(extract_structured_text("langgraph_temp_workflow/input.pdf"))