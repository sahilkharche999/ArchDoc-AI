import streamlit as st
import pandas as pd
import plotly.express as px
import time
import os
import shutil
from pypdf import PdfReader, PdfWriter
from pdf2image import convert_from_path

# --- IMPORT YOUR UTILS ---
# Ensure this path is correct relative to where you run 'streamlit run app.py'
from src.utils import minerU_pdf_creating_extration

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Steel Estimation Dashboard",
    page_icon="🏗️",
    layout="wide"
)

# --- SESSION STATE ---
if "processed" not in st.session_state:
    st.session_state.processed = False
if "layout_images" not in st.session_state:
    st.session_state.layout_images = [] # List of paths to layout images

# --- MOCK DATA (For the Dashboard part) ---
# def get_hardcoded_data():
#     return {
#       "project_summary": "Structural steel takeoff for Roof Framing and Shear Wall systems...",
#       "final_bill_of_materials": [
#         {
#           "category": "W", "description": "W24x62 Beam", "total_qty": 2, "total_linear_feet": 57.08, "total_weight_lbs": 3538.96, "total_bolts": 12, "total_holes": 24, "total_weld_inches": 0.0, "logic_trace": "Found 2 instances..."
#         },
#         {
#           "category": "HSS", "description": "HSS 5x5x5/16 Column", "total_qty": 3, "total_linear_feet": 54.87, "total_weight_lbs": 1046.9, "total_bolts": 12, "total_holes": 12, "total_weld_inches": 0.0, "logic_trace": "Identified at Grids B-2..."
#         }
#       ]
#     }

# --- DATA LOADING (Simulating your JSON Output) ---
def get_hardcoded_data():
    raw_json = {
      "project_summary": "Structural steel takeoff for Roof Framing and Shear Wall systems. Includes W-Shape beams, HSS Columns, MC Channel for ladders, Steel Angle lintels for openings, and Anchor Rods for shear walls. Wood framing and trusses excluded.",
      "final_bill_of_materials": [
        {
          "category": "W",
          "description": "W24x62 Beam",
          "total_qty": 2,
          "total_linear_feet": 57.08,
          "total_weight_lbs": 3538.96,
          "total_bolts": 12,
          "total_holes": 24,
          "total_weld_inches": 0.0,
          "logic_trace": "Found 2 instances with dimensions 27'-0\" and 30'-1\". Sum: 27.0 + 30.08 = 57.08 ft. Weight: 57.08 * 62."
        },
        {
          "category": "W",
          "description": "W14x22 Beam",
          "total_qty": 1,
          "total_linear_feet": 18.58,
          "total_weight_lbs": 408.76,
          "total_bolts": 6,
          "total_holes": 12,
          "total_weld_inches": 0.0,
          "logic_trace": "Found 1 instance with dimension 18'-7\". Weight: 18.58 * 22."
        },
        {
          "category": "W",
          "description": "W12x14 Beam",
          "total_qty": 1,
          "total_linear_feet": 12.0,
          "total_weight_lbs": 168.0,
          "total_bolts": 6,
          "total_holes": 12,
          "total_weld_inches": 0.0,
          "logic_trace": "Found 1 instance with dimension 12'-0\". Weight: 12.0 * 14."
        },
        {
          "category": "HSS",
          "description": "HSS 5x5x5/16 Column",
          "total_qty": 3,
          "total_linear_feet": 54.87,
          "total_weight_lbs": 1046.9,
          "total_bolts": 12,
          "total_holes": 12,
          "total_weld_inches": 0.0,
          "logic_trace": "Identified at Grids B-2, C-2, F-2. Height assumption 18.29' (Global). Total: 3 * 18.29 = 54.87 ft. Weight: 19.08 lbs/ft."
        },
        {
          "category": "HSS",
          "description": "HSS 5x5x1/4 Column",
          "total_qty": 2,
          "total_linear_feet": 36.58,
          "total_weight_lbs": 571.3,
          "total_bolts": 8,
          "total_holes": 8,
          "total_weld_inches": 0.0,
          "logic_trace": "Identified at Grids D-2, E-2. Height assumption 18.29' (Global). Total: 2 * 18.29 = 36.58 ft. Weight: 15.62 lbs/ft."
        },
        {
          "category": "C",
          "description": "MC6x15.1 Ladder Rails",
          "total_qty": 4,
          "total_linear_feet": 73.16,
          "total_weight_lbs": 1104.7,
          "total_bolts": 8,
          "total_holes": 16,
          "total_weld_inches": 0.0,
          "logic_trace": "Detail 7/S-3.2 at F-2 & G-2 implies 2 ladders. 2 rails per ladder. Height 18.29'. Total: 4 * 18.29 = 73.16 ft."
        },
        {
          "category": "L",
          "description": "L4x3.5x5/16 Loose Lintels",
          "total_qty": 13,
          "total_linear_feet": 83.25,
          "total_weight_lbs": 632.7,
          "total_bolts": 0,
          "total_holes": 0,
          "total_weld_inches": 0.0,
          "logic_trace": "Calculated from Window Opening symbols (4/S-4.1, 5/S-4.1). Logic: Opening Width + 1.33' bearing. Sum of 13 openings ranging from 3.33' to 6.42' width."
        },
        {
          "category": "L",
          "description": "L4x4x1/4 Ladder Angles",
          "total_qty": 8,
          "total_linear_feet": 2.0,
          "total_weight_lbs": 13.2,
          "total_bolts": 0,
          "total_holes": 8,
          "total_weld_inches": 32.0,
          "logic_trace": "Detail 7/S-3.2. 4 clips per ladder * 2 ladders. Length 0.25' per clip."
        },
        {
          "category": "ROD",
          "description": "5/8 Inch Anchor Rods",
          "total_qty": 95,
          "total_linear_feet": 142.5,
          "total_weight_lbs": 148.2,
          "total_bolts": 95,
          "total_holes": 0,
          "total_weld_inches": 0.0,
          "logic_trace": "Shear Wall Schedule Spacing (32\" o.c.). Summed wall lengths (approx 125' for Type 1, plus Type 2/3). Added Hold Downs. Total count 95. Length 1.5' ea."
        },
        {
          "category": "ROD",
          "description": "#6 Round Bar Rungs",
          "total_qty": 36,
          "total_linear_feet": 54.0,
          "total_weight_lbs": 81.0,
          "total_bolts": 0,
          "total_holes": 0,
          "total_weld_inches": 72.0,
          "logic_trace": "Ladder Rungs @ 12\" o.c. for 18.29' height = 18 rungs/ladder. 2 ladders. Width 1.5'. Total LF: 36 * 1.5."
        }
      ]
    }
    return raw_json


# --- REAL PROCESSING LOGIC ---
def process_pdf_pipeline(uploaded_file):
    """
    Splits PDF, runs MinerU on each page, and returns paths to Layout Images.
    """
    base_temp_dir = "temp_processing"
    if os.path.exists(base_temp_dir):
        shutil.rmtree(base_temp_dir) # Clean start
    os.makedirs(base_temp_dir, exist_ok=True)
    
    # 1. Save Uploaded File
    main_pdf_path = os.path.join(base_temp_dir, "input.pdf")
    with open(main_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    layout_results = []
    
    # 2. Split PDF into Pages
    reader = PdfReader(main_pdf_path)
    total_pages = len(reader.pages)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(total_pages):
        status_text.text(f"Processing Page {i+1} of {total_pages} with MinerU...")
        
        # Create single page PDF
        page_pdf_name = f"page_{i}.pdf"
        page_pdf_path = os.path.join(base_temp_dir, page_pdf_name)
        
        writer = PdfWriter()
        writer.add_page(reader.pages[i])
        with open(page_pdf_path, "wb") as f:
            writer.write(f)
            
        # 3. Run MinerU
        # Output dir for this page: temp_processing/output_page_0
        page_output_dir = os.path.join(base_temp_dir, f"output_page_{i}")
        
        try:
            minerU_pdf_creating_extration(page_pdf_path, page_output_dir)
            
            # 4. Find the Layout PDF
            # MinerU usually saves it as: output_dir/page_0/page_0_layout.pdf
            # Note: MinerU creates a subfolder based on filename (without extension)
            mineru_subfolder = os.path.splitext(page_pdf_name)[0] # "page_0"
            layout_pdf_path = os.path.join(page_output_dir, mineru_subfolder, f"{mineru_subfolder}_layout.pdf")
            
            if os.path.exists(layout_pdf_path):
                # 5. Convert Layout PDF to Image (for display)
                images = convert_from_path(layout_pdf_path, dpi=150)
                if images:
                    img_save_path = os.path.join(base_temp_dir, f"layout_view_{i}.png")
                    images[0].save(img_save_path, "PNG")
                    layout_results.append(img_save_path)
            else:
                print(f"Layout PDF not found at: {layout_pdf_path}")
                
        except Exception as e:
            st.error(f"Error processing page {i+1}: {e}")
        
        progress_bar.progress((i + 1) / total_pages)

    status_text.empty()
    progress_bar.empty()
    return layout_results

# --- MAIN APP ---

st.title("🏗️ DAX AI Structural Steel Estimator")

# 1. UPLOAD SECTION
if not st.session_state.processed:
    st.markdown("### 1. Upload Project PDF")
    uploaded_file = st.file_uploader("Upload Construction Drawings (PDF)", type="pdf")

    if uploaded_file:
        if st.button("🚀 Analyze Layout & Estimate"):
            with st.status("Processing Document...", expanded=True) as status:
                
                st.write("📥 Splitting PDF & Running MinerU...")
                # CALL THE REAL PIPELINE
                layout_images = process_pdf_pipeline(uploaded_file)
                st.session_state.layout_images = layout_images
                
                st.write("🔍 Layout Analysis Complete.")
                time.sleep(1)
                
                st.write("🤖 Running Estimation Logic (Simulated)...")
                time.sleep(1)
                
                status.update(label="Analysis Complete!", state="complete", expanded=False)
            
            st.session_state.processed = True
            st.rerun()

# 2. RESULTS SECTION
else:
    # --- LAYOUT VISUALIZATION ---
    if st.session_state.layout_images:
        with st.expander("📄 View Layout Analysis (MinerU Output)", expanded=True):
            st.info("These are the regions detected by the Layout Model (Tables, Figures, Text).")
            
            # Display images in a grid
            cols = st.columns(3)
            for i, img_path in enumerate(st.session_state.layout_images):
                with cols[i % 3]:
                    st.image(img_path, caption=f"Page {i+1} Layout", use_container_width=True)
    else:
        st.warning("No layout images were generated. Check MinerU logs.")

    st.divider()

    # --- DASHBOARD (Hardcoded Data) ---
    data = get_hardcoded_data()
    df = pd.DataFrame(data["final_bill_of_materials"])

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        price_per_lb = st.number_input("Steel Price ($/lb)", value=1.50, step=0.10)
        fab_cost_factor = st.slider("Fabrication Markup %", 0, 100, 20)
        
        st.divider()
        selected_categories = st.multiselect(
            "Filter Category", df["category"].unique(), default=df["category"].unique()
        )
        
        if st.button("🔄 Start Over"):
            st.session_state.processed = False
            st.session_state.layout_images = []
            st.rerun()

    # Filter
    filtered_df = df[df["category"].isin(selected_categories)]

    # Metrics
    total_weight = filtered_df["total_weight_lbs"].sum()
    total_cost = total_weight * price_per_lb * (1 + fab_cost_factor/100)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Weight", f"{total_weight:,.0f} lbs")
    c2.metric("Total Items", f"{filtered_df['total_qty'].sum()}")
    c3.metric("Est. Cost", f"${total_cost:,.2f}")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Analytics", "📋 Bill of Materials", "🧠 Logic Trace"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(filtered_df, values='total_weight_lbs', names='category', title="Weight by Category")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.bar(filtered_df, x='category', y='total_linear_feet', title="Linear Feet by Category")
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.dataframe(
            filtered_df[["category", "description", "total_qty", "total_linear_feet", "total_weight_lbs"]],
            use_container_width=True
        )

    with tab3:
        for _, row in filtered_df.iterrows():
            with st.expander(f"{row['description']}"):
                st.info(row['logic_trace'])

# --- FOOTER ---
st.markdown("---")
st.caption("Powered by MinerU & Gemini 1.5 Pro")