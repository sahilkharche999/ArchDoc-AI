import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.express as px
import time

from src.service import stream_estimation

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Steel Estimation Dashboard",
    page_icon="🏗️",
    layout="wide"
)

# --- INITIALIZE SESSION STATE ---
if "estimation_done" not in st.session_state:
    st.session_state.estimation_done = False
if "ai_data" not in st.session_state:
    st.session_state.ai_data = None

# --- AI EXECUTION FUNCTION ---

def run_ai_estimation(uploaded_file):

    progress_bar = st.progress(0)
    status_text = st.empty()

    os.makedirs("assets", exist_ok=True)

    file_path = f"assets/{uploaded_file.name}"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    agent_result = None
    step_count = 0
    estimated_total_steps = 4  # adjust to match your real nodes

    try:
        for thread_id, event in stream_estimation(file_path, "output_temp"):

            st.session_state.thread_id = thread_id

            for node_name, state_update in event.items():

                step_count += 1
                percent = int((step_count / estimated_total_steps) * 100)

                status_text.text(f"Finished Agent {step_count}: {node_name}")
                progress_bar.progress(min(percent, 95))

                # Capture final result dynamically
                if "final_bill_of_materials" in state_update:
                    agent_result = state_update

        progress_bar.progress(100)
        status_text.text("Estimation Complete")

        if agent_result:
            st.session_state.ai_data = agent_result
        else:
            st.error("No final result returned from workflow.")
            st.stop()

    except Exception as e:
        st.error(f"Error during estimation: {e}")
        st.stop()

    st.session_state.estimation_done = True
    st.rerun()

# --- HEADER ---
st.title("🏗️ DAX AI Structural Steel Estimator")
st.divider()

# --- IF ESTIMATION IS NOT DONE YET (SHOW UPLOAD IN THE MIDDLE) ---
if not st.session_state.estimation_done:
    col_spacer1, col_center, col_spacer2 = st.columns([1, 2, 1])
    with col_center:
        st.markdown("### 📂 Upload Blueprint")
        st.info("Please upload a PDF blueprint below to begin the extraction process.")
        uploaded_file = st.file_uploader("Upload PDF File", type=["pdf"])
        
        if uploaded_file is not None:
            st.write("") 
            if st.button("🚀 Run AI Estimation", use_container_width=True, type="primary"):
                run_ai_estimation(uploaded_file)


else:
    # --- 1. LOAD THE DATA GENERATED FROM AGENT 4 ---
    data = st.session_state.ai_data or {}
    
    if isinstance(data.get("final_bill_of_materials"), dict) and "final_bill_of_materials" in data["final_bill_of_materials"]:
        materials_list = data["final_bill_of_materials"]["final_bill_of_materials"]
    else:
        materials_list = data.get("final_bill_of_materials", [])

    df = pd.DataFrame(materials_list)

    # --- 2. DATA NORMALIZATION (THIS FIXES THE KEYERROR) ---
    if not df.empty:
        if 'category' not in df.columns:
            if 'material_size' in df.columns:
                df['category'] = df['material_size'].astype(str).str.extract(r'([A-Za-z]+)', expand=False).fillna('MISC')
            else:
                df['category'] = 'MISC'
                
        # Rename 'quantity' to 'total_qty'
        if 'total_qty' not in df.columns and 'quantity' in df.columns:
            df.rename(columns={'quantity': 'total_qty'}, inplace=True)
            
        # Add fallback columns if they are missing so charts don't crash
        if 'total_weight_lbs' not in df.columns:
            df['total_weight_lbs'] = 0.0
        if 'total_linear_feet' not in df.columns:
            df['total_linear_feet'] = 0.0

    # Ensure empty state fallback
    if df.empty:
        df = pd.DataFrame({"category": ["NONE"], "description": ["No materials found"], "total_qty": [0], "total_linear_feet": [0.0], "total_weight_lbs": [0.0]})

    # --- SIDEBAR CONTROLS ---
    st.sidebar.success("✅ Estimation Complete")
    st.sidebar.divider()
    
    st.sidebar.header("⚙️ Estimation Settings")
    price_per_lb = st.sidebar.number_input("Steel Price ($/lb)", value=1.50, step=0.10)
    fab_cost_factor = st.sidebar.slider("Fabrication Markup %", 0, 100, 20)

    # Filters
    st.sidebar.subheader("Filters")
    selected_categories = st.sidebar.multiselect(
        "Filter by Category",
        options=df["category"].unique(),
        default=df["category"].unique()
    )

    filtered_df = df[df["category"].isin(selected_categories)]

    # --- METRICS ROW ---
    total_weight = filtered_df["total_weight_lbs"].sum()
    total_pieces = filtered_df["total_qty"].sum()
    total_lf = filtered_df["total_linear_feet"].sum()

    material_cost = total_weight * price_per_lb
    total_cost = material_cost * (1 + fab_cost_factor/100)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Weight", f"{total_weight:,.0f} lbs")
    col2.metric("Total Linear Feet", f"{total_lf:,.1f} ft")
    col3.metric("Total Pieces", f"{total_pieces}")
    col4.metric("Est. Cost", f"${total_cost:,.2f}", delta=f"@ ${price_per_lb}/lb")

    # --- TABS FOR VIEWING DATA ---
    tab1, tab2, tab3 = st.tabs(["📊 Analytics", "📋 Bill of Materials", "🧠 AI Logic Trace"])

    # --- TAB 1: ANALYTICS ---
    with tab1:
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Weight Distribution by Category")
            if total_weight > 0:
                fig_weight = px.pie(filtered_df, values='total_weight_lbs', names='category', hole=0.4)
                st.plotly_chart(fig_weight, use_container_width=True)
            else:
                st.info("No weight calculations provided by the AI for this drawing.")
            
        with c2:
            st.subheader("Linear Feet by Category")
            if total_lf > 0:
                cat_group = filtered_df.groupby("category")["total_linear_feet"].sum().reset_index()
                fig_lf = px.bar(cat_group, x='category', y='total_linear_feet', color='category')
                st.plotly_chart(fig_lf, use_container_width=True)

        # Fabrication Metrics
        st.subheader("Fabrication Totals")
        fab_metrics = {
            "Total Bolts": filtered_df["total_bolts"].sum() if "total_bolts" in filtered_df else 0,
            "Total Holes": filtered_df["total_holes"].sum() if "total_holes" in filtered_df else 0,
            "Weld Inches": filtered_df["total_weld_inches"].sum() if "total_weld_inches" in filtered_df else 0
        }
        st.dataframe(pd.DataFrame([fab_metrics]), hide_index=True)

    # --- TAB 2: BILL OF MATERIALS ---
    with tab2:
        st.subheader("Detailed Takeoff")
        
        cols_to_show = [c for c in ["category", "material_size", "description", "total_qty", "total_linear_feet", "total_weight_lbs", "total_bolts", "total_weld_inches"] if c in filtered_df.columns]
        
        st.dataframe(
            filtered_df[cols_to_show],
            use_container_width=True,
            column_config={
                "total_weight_lbs": st.column_config.NumberColumn("Weight (lbs)", format="%.1f"),
                "total_linear_feet": st.column_config.NumberColumn("Length (ft)", format="%.2f"),
            }
        )
        
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download BOM as CSV",
            data=csv,
            file_name='steel_takeoff.csv',
            mime='text/csv',
        )

    # --- TAB 3: LOGIC TRACE ---
    with tab3:
        st.subheader("AI Reasoning & Traceability")
        
        if "logic_trace" in filtered_df.columns:
            for index, row in filtered_df.iterrows():
                # Uses material_size if it exists, otherwise falls back to category
                title_text = f"{row.get('material_size', row.get('category', 'Item'))} - {row.get('description', '')}"
                
                with st.expander(title_text):
                    st.markdown(f"**Logic Trace:**")
                    st.info(row.get('logic_trace', 'No logic trace available.'))
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Qty", row.get('total_qty', 0))
                    c2.metric("Length", f"{row.get('total_linear_feet', 0)} ft")
                    c3.metric("Weight", f"{row.get('total_weight_lbs', 0)} lbs")

    # --- FOOTER ---
    st.markdown("---")
    st.caption("Generated by LangGraph Multi-Agent System | Powered by Gemini 1.5 Pro")
    
    if st.sidebar.button("🔄 Start Over", use_container_width=True):
        st.session_state.estimation_done = False
        st.session_state.ai_data = None
        st.rerun()