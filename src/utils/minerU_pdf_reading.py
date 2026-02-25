import subprocess
import json
import os

def minerU_pdf_creating_extration(pdf_path:str,output_dir:str,backend_type:str):
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "mineru",

        "--path", pdf_path,
        "--output", output_dir,
        "--backend", "pipeline",
        "--method", "auto",
        "--lang", "en",
        "--table", "true",
        "--formula", "false",
        "--backend",backend_type
    ]

    subprocess.run(cmd, check=True)
    
if __name__ == "__main__":
    pdf_path = "output_temp/section_page_5.pdf"       
    output_dir = "output"
    backend_type="pipeline"
    minerU_pdf_creating_extration(pdf_path,output_dir,backend_type)
