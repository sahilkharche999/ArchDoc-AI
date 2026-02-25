import pdfplumber
from pdfplumber import open as pdf_open
def extract_words_with_coords(pdf_path, page_num):
    try:
        with pdfplumber.open(pdf_path) as pdf:
         page = pdf.pages[page_num - 1]
         return page.extract_words(
            use_text_flow=True,
            keep_blank_chars=False
         )
        return True
    except Exception as e:
       raise ValueError(e)
    
if __name__=="__main__":
    extract_words_with_coords()
