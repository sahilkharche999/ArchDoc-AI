import fitz
def convert_specific_page_to_png(pdf_path,page_num,output_image_path,dpi=300):
    try:
        #open the pdf into one container
        doc=fitz.open(pdf_path)
        if 0<=page_num<doc.page_count:
            page=doc.load_page(page_num)
            pix=page.get_pixmap(dpi=dpi)
            pix.save(output_image_path)
            print(f"Successfully converted page {page_num } to {output_image_path}")
        else:
            print(f"Error: Page number {page_num } is out of range.")
        doc.close()
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__=='__main__':
 convert_specific_page_to_png('../extracted_pages.pdf', 8, 'output_page_8.png')



        