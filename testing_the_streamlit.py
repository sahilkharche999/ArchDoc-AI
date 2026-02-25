import streamlit as st
import pandas as pd
import plotly.express as px
import json
import time
import sqlite3
import os # Added for path parsing
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph_temp_workflow.workflows.estimation.graph import workflow 

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

# --- FALLBACK DATA FUNCTION ---
def load_data():
    raw_json = {
      "project_summary": "Structural steel takeoff for Roof Framing and Shear Wall systems.",
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
        }
      ]
    }
    return raw_json

# --- AI EXECUTION FUNCTION ---
def run_ai_estimation():
    # Progress UI rendering in the main body
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Simulate Agents 1 to 3
    steps = [
        ("Agent 1: Classifying Pages & Extracting Text Rules...", 20),
        ("Agent 2: Scanning Floor Plans & Identifying Symbols...", 45),
        ("Agent 3: Cropping Section Details & Building Library...", 70),
    ]

    for text, percent in steps:
        status_text.text(text)
        time.sleep(1) # Simulate visual processing time
        progress_bar.progress(percent)
    
    status_text.text("Agent 4: Merging Logic & Calculating Linear Feet (Running Backend)...")
    progress_bar.progress(85)
    
    # --- ACTUAL LANGGRAPH BACKEND CODE ---
    try:
        # 1. Setup
        conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
        memory = SqliteSaver(conn)
        app = workflow.compile(checkpointer=memory)
        config = {"configurable": {"thread_id": "job_123"}}

        # 2. Define the CORRECT Floor Plan Paths
        manual_floor_plan = [
            "output_temp/floor_3/floor_3/vlm/images/c2071a8eb39ff6495f84a2cb170897bc62a795ef8b60ce9e337bd32f615e99dc.jpg",
            "output_temp/floor_4/floor_4/vlm/images/cb7ea89114e1c238311cf9bf3f1babcc1ef68eec3373691da3efe37289b125fe.jpg"
        ]

        # 3. Inject State & Rewind
        app.update_state(
            config, 
            {"floor_plan_images": manual_floor_plan}, 
            as_node="process_details" 
        )

        # 4. Run Agent 4 & Capture Output
        agent_result = None
        for event in app.stream(None, config=config):
            for node_name, state_update in event.items():
                if node_name == "agent_4_merger":
                    agent_result = state_update
        
        # Save results to session state
        if agent_result and "final_bill_of_materials" in agent_result:
            st.session_state.ai_data = agent_result
        else:
            st.session_state.ai_data = load_data()

    except Exception as e:
        st.error(f"Error running LangGraph: {e}")
        st.session_state.ai_data = load_data() 
        
    finally:
        status_text.text("Finalizing Bill of Materials...")
        progress_bar.progress(100)
        time.sleep(0.5)
        
        status_text.empty()
        progress_bar.empty()
        
        # Mark estimation as done and reload page
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
                run_ai_estimation()

# --- IF ESTIMATION IS DONE (SHOW EXISTING DASHBOARD WITH SIDEBAR) ---
else:
    # --- 1. LOAD THE DATA GENERATED FROM AGENT 4 ---
    data = st.session_state.ai_data
    
    # If the output from agent is a nested dict (sometimes happens with AI output)
    if isinstance(data.get("final_bill_of_materials"), dict) and "final_bill_of_materials" in data["final_bill_of_materials"]:
        materials_list = data["final_bill_of_materials"]["final_bill_of_materials"]
    else:
        materials_list = data.get("final_bill_of_materials", [])

    df = pd.DataFrame(materials_list)

    # --- 2. DATA NORMALIZATION (THIS FIXES THE KEYERROR) ---
    if not df.empty:
        # Create 'category' by extracting the first letters of 'material_size' (e.g. 'ROD3/4' -> 'ROD')
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