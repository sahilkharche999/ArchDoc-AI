from pypdf import PdfReader, PdfWriter
import sys

def parse_pages(pages_str):
    """
    Converts a string like '1,3,5-7' into a sorted list of page indices (0-based).
    """
    pages = set()
    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            pages.update(range(int(start) - 1, int(end)))
        else:
            pages.add(int(part) - 1)
    return sorted(pages)

def extract_pdf_pages(input_pdf, pages_str, output_pdf):
    reader = PdfReader(input_pdf)
    writer = PdfWriter()

    pages = parse_pages(pages_str)

    for page_num in pages:
        if 0 <= page_num < len(reader.pages):
            writer.add_page(reader.pages[page_num])
        else:
            print(f"Skipping invalid page number: {page_num + 1}")

    with open(output_pdf, "wb") as f:
        writer.write(f)

    print(f"Created {output_pdf} with pages: {pages_str}")

if __name__ == "__main__":

    input_pdf ='extracted_pages.pdf'
    pages_str = "6:11"
    output_pdf = 'all_section.pdf'
    extract_pdf_pages(input_pdf, pages_str, output_pdf)
