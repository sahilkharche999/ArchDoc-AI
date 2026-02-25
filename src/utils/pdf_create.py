from pypdf import PdfReader,PdfWriter
import fitz 
def draw_bounding_boxes(
    input_pdf: str,
    output_pdf: str,
    title_coordinates: dict
):
    doc = fitz.open(input_pdf)

    for page_key, titles in title_coordinates.items():
        page_index = int(page_key.split("_")[1]) - 1
        page = doc[page_index]

        for title, coords in titles.items():
            rect = fitz.Rect(
                coords["x1"],
                coords["y1"],
                coords["x2"],
                coords["y2"]
            )
            # Draw rectangle
            page.draw_rect(
                rect,
                color=(1, 0, 0),   # red
                width=1.5
            )
            # Optional: add title text above box
            page.insert_text(
                fitz.Point(coords["x1"], coords["y1"] - 5),
                title,
                fontsize=6,
                color=(1, 0, 0)
            )

    doc.save(output_pdf)
    doc.close()

def extracte_pdf(inputPdf,outputPdf,start_page,end_page):
    reader=PdfReader(inputPdf)
    writer=PdfWriter()
    for page_num in range(start_page-1,end_page):
        writer.add_page(reader.pages[page_num])
    with open(outputPdf,'wb') as f:
        writer.write(f)
    print(f"Pages {start_page} to {end_page} saved to {outputPdf}")


if __name__=="__main__":
    input_pdf = "Anderson WTP 1 Plans for ai.pdf"
    output_pdf = "S302.pdf"
    start_page = 65
    end_page = 65
    extracte_pdf(input_pdf,output_pdf,start_page,end_page)
    
# coords={'page_1': {'TYP. OPENING ELEVATION': {'x1': 195.2399, 'y1': 527.37646, 'x2': 657.002560776256, 'y2': 552.2964599999998}, 'BUILT UP POST': {'x1': 1121.8796, 'y1': 527.37646, 'x2': 1390.0900082770002, 'y2': 552.2964599999998}, 'TYP. NON-LOAD BEARING WALL': {'x1': 1671.7193, 'y1': 527.37646, 'x2': 2221.330257372736, 'y2': 552.2964599999998}, 'TYP. SHEAR WALL ELEVATION': {'x1': 195.2399, 'y1': 1075.77626, 'x2': 722.4038698877439, 'y2': 1100.6962600000002}, 'SECTION AT CANOPY': {'x1': 976.3196, 'y1': 1075.77626, 'x2': 1346.2506551686402, 'y2': 1100.6962600000002}, 'BUILT UP BEAM NAILING PATTERN': {'x1': 1671.7193, 'y1': 1075.77626, 'x2': 2278.645419051476, 'y2': 1100.6962600000002}, 'CORNER FRAMING DETAIL': {'x1': 195.2399, 'y1': 1623.09596, 'x2': 651.5738059032601, 'y2': 1648.01596}, 'ROOF DIAPHRAGM NAILING': {'x1': 754.7997, 'y1': 1623.09596, 'x2': 1232.2646061339362, 'y2': 1648.01596}, 'TYP. TOP PLATE SPLICE': {'x1': 1338.4795, 'y1': 1623.09596, 'x2': 1756.837967041984, 'y2': 1648.01596}, 'TYP. SPLICE SECTION': {'x1': 1846.5593, 'y1': 1623.09596, 'x2': 2226.840817081536, 'y2': 1648.01596}}}
# draw_bounding_boxes(
#     input_pdf="onepagesection.pdf",
#     output_pdf="onepagesection_with_boxes.pdf",
#     title_coordinates=coords
# )


