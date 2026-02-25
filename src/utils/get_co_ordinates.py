import fitz  # PyMuPDF

def find_co_ordinates(pdfPath:str):
   doc = fitz.open(pdfPath)
   with open("/Users/consultadd/Desktop/Dex/output.txt", "w", encoding="utf-8") as f:
    # Header
    f.write("page\ttext\tx1\ty1\tx2\ty2\n")

    for page_num, page in enumerate(doc, start=1):
        words = page.get_text("words")

        for w in words:
            x1, y1, x2, y2, text = w[:5]
            text = text.replace("\t", " ").replace("\n", " ")

            f.write(
                f"{page_num}\t{text}\t"
                f"{x1:.2f}\t{y1:.2f}\t{ x2:.2f}\t{y2:.2f}\n"
            )
    return 
   
# def extract_words_with_coords(pdf_path, page_num):
#     with pdfplumber.open(pdf_path) as pdf:
#         page = pdf.pages[page_num - 1]
#         return page.extract_words(
#             use_text_flow=True,
#             keep_blank_chars=False
#         )

find_co_ordinates('floor_plan.pdf')

