import os
import io
import re
import pandas as pd
import streamlit as st
import joblib
import time
import numpy as np
from datetime import datetime

# ========= Model load =========
MODEL_PATH = "house_price_model.pkl"
st.set_page_config(page_title="🏠 House Price Prediction", layout="wide", initial_sidebar_state="expanded")

# --- USER AUTHENTICATION / WELCOME PAGE ---
if "username" not in st.session_state:
    st.session_state.username = ""
if "login_animation" not in st.session_state:
    st.session_state.login_animation = False

def logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

def is_valid_name(name):
    return bool(re.fullmatch(r"[A-Za-z\s'\-]+", name))

# Show login page if not authenticated
if not st.session_state.username:
    # Create a futuristic login interface
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🏠 House Price System ")
        st.subheader(" House Price Prediction System")
        
        with st.container(border=True):
            username_input = st.text_input("**ENTER YOUR IDENTITY**", placeholder="e.g., Awais Ali")
            st.caption("Allowed: letters, spaces, apostrophes, and hyphens")

            if st.button("🔓 ACCESS SYSTEM", use_container_width=True, type="primary"):
                if username_input.strip() == "":
                    st.error("❌ IDENTITY REQUIRED FOR SYSTEM ACCESS")
                elif not is_valid_name(username_input.strip()):
                    st.error("❌ INVALID IDENTITY FORMAT")
                else:
                    st.session_state.login_animation = True
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i in range(100):
                        progress_bar.progress(i + 1)
                        status_text.text(f"INITIALIZING SYSTEM... {i+1}%")
                        time.sleep(0.02)
                    
                    st.session_state.username = username_input.strip()
                    st.session_state.login_animation = False
                    st.rerun()
    
    # Feature highlights
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True, height=150):
            st.markdown("**🤖 AI-POWERED**")
            st.markdown("Machine learning algorithms for precise predictions")
    with col2:
        with st.container(border=True, height=150):
            st.markdown("**📊 REAL-TIME ANALYSIS**")
            st.markdown("Instant property valuation with detailed insights")
    with col3:
        with st.container(border=True, height=150):
            st.markdown("**🚀 FUTURISTIC UI**")
            st.markdown("Advanced interface with immersive experience")
    
    st.stop()

# --- Main app after user login ---
st.sidebar.title(f"👤 USER: {st.session_state.username}")
st.sidebar.markdown("**SYSTEM STATUS: ACTIVE**")

if st.sidebar.button("🚪 LOGOUT", use_container_width=True, type="primary"):
    logout()

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file `{MODEL_PATH}` not found in this folder.")
    st.stop()

# Load model with animation
with st.spinner("🤖 LOADING PREDICTION ENGINE..."):
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)
        time.sleep(0.01)
    model = joblib.load(MODEL_PATH)

st.title("🏠 Price Forecast Platform")
st.markdown("Advanced machine learning system for precise house price predictions")

# ========= Sidebar =========
st.sidebar.divider()
st.sidebar.header("ℹ️ SYSTEM INFO")
st.sidebar.markdown("This AI uses advanced ML models with preprocessing pipeline to predict property values")

st.sidebar.divider()
st.sidebar.header("👨‍💻 DEVELOPED BY")
st.sidebar.markdown("**Awais Ali**")
st.sidebar.markdown("🎓 University of Narowal")
st.sidebar.markdown("📧 awaisshshid6890@gmail.com")

# ========= Currency =========
st.header("💲 SELECT CURRENCY")
currency = st.radio(
    "Choose your currency:",
    ["₨ Pakistani Rupee", "₹ Indian Rupee", "$ US Dollar", "€ Euro"],
    horizontal=True
)
currency_symbol = currency.split(" ")[0]

# ========= Tabs =========
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 SINGLE PREDICTION",
    "🏘️ MULTIPLE PROPERTIES",
    "📂 BATCH PROCESSING",
    "📊 SYSTEM INSIGHTS"
])

# ========= Helpers =========
ALL_FEATURES = [
    "area", "bedrooms", "bathrooms", "stories", "parking",
    "mainroad", "guestroom", "basement", "hotwaterheating",
    "airconditioning", "prefarea", "furnishingstatus"
]

if "history" not in st.session_state:
    st.session_state.history = []

def format_currency(v):
    return f"{currency_symbol} {v:,.2f}"

def validate_inputs(df):
    warnings = []
    if (df["area"] <= 0).any():
        warnings.append("⚠️ Area should be greater than 0.")
    if (df["bedrooms"] <= 0).any():
        warnings.append("⚠️ Bedrooms should be at least 1.")
    if (df["bathrooms"] <= 0).any():
        warnings.append("⚠️ Bathrooms should be at least 1.")
    return warnings

def validate_csv_columns(df):
    missing_cols = [col for col in ALL_FEATURES if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Uploaded CSV is missing required columns: {missing_cols}")
        st.stop()

def download_results(df, filename_prefix="Predicted_Prices"):
    col1, col2, col3 = st.columns(3)

    # CSV
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    with col1:
        st.download_button(
            "⬇️ CSV",
            csv_bytes,
            file_name=f"{filename_prefix}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Excel
    try:
        import openpyxl
        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Predictions")
        with col2:
            st.download_button(
                "⬇️ EXCEL",
                excel_buf.getvalue(),
                file_name=f"{filename_prefix}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    except Exception:
        with col2:
            st.info("📎 Install `openpyxl` for Excel export")

    # PDF (as table)
    try:
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet

        pdf_buf = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buf, pagesize=A4, title="House Price Predictions")
        elements = []
        styles = getSampleStyleSheet()

        # Title
        elements.append(Paragraph("🏠 House Price Predictions Report", styles["Title"]))
        elements.append(Spacer(1, 12))

        # Convert DataFrame to table data (with header)
        table_data = [df.columns.tolist()] + df.astype(str).values.tolist()
        table = Table(table_data, repeatRows=1)

        # Styling
        style = TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
        ])
        table.setStyle(style)

        elements.append(table)
        doc.build(elements)

        with col3:
            st.download_button(
                "⬇️ PDF",
                data=pdf_buf.getvalue(),
                file_name=f"{filename_prefix}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    except Exception:
        with col3:
            st.info("📎 Install `reportlab` for PDF export")

# ========= Tab 1: Single Prediction =========
with tab1:
    st.header("🔮 PREDICT A SINGLE PROPERTY")
    
    with st.expander("📝 PROPERTY DETAILS", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            area = st.slider("Area (sqft)", min_value=500, max_value=10000, value=2000, step=50)
            bedrooms = st.slider("Bedrooms", min_value=1, max_value=10, value=3, step=1)
            bathrooms = st.slider("Bathrooms", min_value=1, max_value=5, value=2, step=1)
            stories = st.slider("Stories", min_value=1, max_value=5, value=2, step=1)
            parking = st.slider("Parking (spots)", min_value=0, max_value=5, value=1, step=1)

        with col2:
            mainroad = st.selectbox("Main Road", ["yes", "no"], index=0)
            guestroom = st.selectbox("Guestroom", ["yes", "no"], index=1)
            basement = st.selectbox("Basement", ["yes", "no"], index=1)
            hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"], index=1)
            airconditioning = st.selectbox("Air Conditioning", ["yes", "no"], index=0)
            prefarea = st.selectbox("Preferred Area", ["yes", "no"], index=0)
            furnishingstatus = st.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"], index=1)

    input_dict = {
        "area": area, "bedrooms": bedrooms, "bathrooms": bathrooms, "stories": stories, "parking": parking,
        "mainroad": mainroad, "guestroom": guestroom, "basement": basement,
        "hotwaterheating": hotwaterheating, "airconditioning": airconditioning,
        "prefarea": prefarea, "furnishingstatus": furnishingstatus
    }
    input_df = pd.DataFrame([input_dict])

    if st.button("💰 PREDICT PRICE", use_container_width=True, type="primary"):
        with st.spinner("ANALYZING PROPERTY DATA..."):
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
                time.sleep(0.02)
            
            warnings = validate_inputs(input_df)
            if warnings:
                for w in warnings:
                    st.warning(w)

            try:
                pred = float(model.predict(input_df)[0])
                input_df["Predicted Price"] = format_currency(pred)

                # Display result with animation
                st.balloons()
                with st.container(border=True):
                    st.metric(label="🏷️ ESTIMATED PROPERTY VALUE", value=format_currency(pred))
                
                st.session_state.history.append(pred)
                download_results(input_df, "Single_Property_Prediction")
                
                # Show similar properties
                with st.expander("🏘️ SIMILAR PROPERTIES IN MARKET"):
                    similar_data = {
                        "Features": ["Slightly smaller", "Similar size", "Larger"],
                        "Price Range": [
                            format_currency(pred * 0.8), 
                            format_currency(pred), 
                            format_currency(pred * 1.2)
                        ]
                    }
                    st.dataframe(similar_data, use_container_width=True)
                
            except Exception as e:
                st.error(f"⚠️ PREDICTION FAILED: {e}")

    # Prediction history
    if st.session_state.history:
        st.divider()
        st.subheader("📈 PREDICTION HISTORY")
        history_df = pd.DataFrame({
            "Date": [datetime.now().strftime("%Y-%m-%d %H:%M") for _ in st.session_state.history],
            "Predicted Price": [format_currency(p) for p in st.session_state.history]
        })
        st.dataframe(history_df.tail(5), use_container_width=True)
        
        # Simple trend visualization
        if len(st.session_state.history) > 1:
            trend_data = pd.DataFrame({
                "Price": st.session_state.history
            })
            st.line_chart(trend_data)

# ========= Tab 2: Multiple Predictions =========
with tab2:
    st.header("🏘️ COMPARE MULTIPLE PROPERTIES")
    
    n_houses = st.slider("Number of properties to compare", min_value=2, max_value=10, value=2)
    house_data = []

    for i in range(n_houses):
        with st.expander(f"🏠 PROPERTY {i+1}", expanded=i==0):
            col1, col2 = st.columns(2)
            with col1:
                area = st.slider(f"Area (sqft)", min_value=500, max_value=10000, value=2000, step=50, key=f"area_{i}")
                bedrooms = st.slider(f"Bedrooms", min_value=1, max_value=10, value=3, step=1, key=f"bed_{i}")
                bathrooms = st.slider(f"Bathrooms", min_value=1, max_value=5, value=2, step=1, key=f"bath_{i}")
                stories = st.slider(f"Stories", min_value=1, max_value=5, value=2, step=1, key=f"stories_{i}")
                parking = st.slider(f"Parking (spots)", min_value=0, max_value=5, value=1, step=1, key=f"park_{i}")
            with col2:
                mainroad = st.selectbox(f"Main Road", ["yes", "no"], index=0, key=f"main_{i}")
                guestroom = st.selectbox(f"Guestroom", ["yes", "no"], index=1, key=f"guest_{i}")
                basement = st.selectbox(f"Basement", ["yes", "no"], index=1, key=f"base_{i}")
                hotwaterheating = st.selectbox(f"Hot Water Heating", ["yes", "no"], index=1, key=f"hot_{i}")
                airconditioning = st.selectbox(f"Air Conditioning", ["yes", "no"], index=0, key=f"ac_{i}")
                prefarea = st.selectbox(f"Preferred Area", ["yes", "no"], index=0, key=f"pref_{i}")
                furnishingstatus = st.selectbox(f"Furnishing Status", ["furnished", "semi-furnished", "unfurnished"], index=1, key=f"furn_{i}")

            house_data.append({
                "Property": f"Property {i+1}",
                "area": area, "bedrooms": bedrooms, "bathrooms": bathrooms, "stories": stories, "parking": parking,
                "mainroad": mainroad, "guestroom": guestroom, "basement": basement,
                "hotwaterheating": hotwaterheating, "airconditioning": airconditioning,
                "prefarea": prefarea, "furnishingstatus": furnishingstatus
            })

    if st.button("💰 COMPARE PRICES", use_container_width=True, type="primary"):
        df_multi = pd.DataFrame(house_data)
        try:
            with st.spinner("ANALYZING PROPERTIES..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    progress_bar.progress(i + 1)
                    time.sleep(0.01)
                
                # Extract features for prediction
                feature_cols = [col for col in df_multi.columns if col != "Property"]
                preds = model.predict(df_multi[feature_cols])
                
                # Add predictions to dataframe
                df_multi["Predicted Price"] = [format_currency(p) for p in preds]
                df_multi["Price Value"] = preds
                
                st.success("ANALYSIS COMPLETE!")
                
                # Display comparison table
                display_cols = ["Property"] + feature_cols + ["Predicted Price"]
                st.dataframe(df_multi[display_cols], use_container_width=True)
                
                # Visual comparison
                chart_data = pd.DataFrame({
                    "Property": [f"Property {i+1}" for i in range(len(preds))],
                    "Price": preds
                }).set_index("Property")
                
                st.bar_chart(chart_data)
                
                # Add to history
                for pred in preds:
                    st.session_state.history.append(pred)
                
                # Offer download
                download_results(df_multi[display_cols], "Multiple_Property_Comparison")
                
        except Exception as e:
            st.error(f"⚠️ PREDICTION FAILED: {e}")

# ========= Tab 3: Batch Prediction (CSV) =========
with tab3:
    st.header("📂 BATCH PROCESSING")
    
    uploaded = st.file_uploader("Upload a CSV file with property data", type=["csv"])
    
    if uploaded:
        df_csv = pd.read_csv(uploaded)
        validate_csv_columns(df_csv)
        
        st.subheader("📋 UPLOADED DATA PREVIEW")
        st.dataframe(df_csv.head(), use_container_width=True)
        st.write(f"Total properties: {len(df_csv)}")

        if st.button("💰 PROCESS BATCH", use_container_width=True, type="primary"):
            try:
                with st.spinner("PROCESSING BATCH DATA..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i in range(100):
                        progress_bar.progress(i + 1)
                        status_text.text(f"PROCESSING... {i+1}%")
                        time.sleep(0.01)
                    
                    # Make predictions
                    preds = model.predict(df_csv)
                    df_csv["Predicted Price"] = [format_currency(p) for p in preds]
                    df_csv["Price Value"] = preds
                    
                    status_text.text("✅ PROCESSING COMPLETE!")
                    st.balloons()
                    
                    # Show results
                    st.subheader("📊 PREDICTION RESULTS")
                    st.dataframe(df_csv.head(10), use_container_width=True)
                    
                    # Show summary stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Properties", len(df_csv))
                    with col2:
                        st.metric("Average Price", format_currency(np.mean(preds)))
                    with col3:
                        st.metric("Price Range", f"{format_currency(min(preds))} - {format_currency(max(preds))}")
                    
                    # Add to history
                    for pred in preds[:10]:  # Limit to avoid memory issues
                        st.session_state.history.append(pred)
                    
                    # Offer download
                    download_results(df_csv, "Batch_Property_Predictions")
                    
            except Exception as e:
                st.error(f"⚠️ PREDICTION FAILED: {e}")

# ========= Tab 4: Model Insights =========
with tab4:
    st.header("📊 SYSTEM INSIGHTS")
    
    # Feature importance
    st.subheader("🔍 FEATURE IMPORTANCE")
    try:
        if hasattr(model, "named_steps") and "model" in model.named_steps:
            base_model = model.named_steps["model"]
            preprocess = model.named_steps.get("preprocess", None)
            ohe = None
            cat_features = []

            if preprocess and "cat" in preprocess.named_transformers_:
                cat_pipe = preprocess.named_transformers_["cat"]
                if hasattr(cat_pipe, "named_steps") and "onehot" in cat_pipe.named_steps:
                    ohe = cat_pipe.named_steps["onehot"]

            if ohe is not None:
                cat_features = ohe.get_feature_names_out([
                    "mainroad","guestroom","basement",
                    "hotwaterheating","airconditioning",
                    "prefarea","furnishingstatus"
                ]).tolist()
            num_features = ["area","bedrooms","bathrooms","stories","parking"]
            feature_names = num_features + cat_features if cat_features else num_features

            if hasattr(base_model, "feature_importances_"):
                importances = pd.Series(base_model.feature_importances_, index=feature_names)
                importances = importances.sort_values(ascending=False)
                
                # Create a simple bar chart
                chart_data = pd.DataFrame({
                    'Feature': importances.index,
                    'Importance': importances.values
                })
                
                st.bar_chart(chart_data.set_index('Feature').head(10))
            else:
                st.warning("⚠️ Feature importance is only available for tree-based models.")
        else:
            st.warning("⚠️ Feature importance requires a Pipeline with 'preprocess' and a tree-based 'model'.")
    except Exception as e:
        st.warning(f"⚠️ Could not extract feature importances: {e}")
    
    # Model information
    st.divider()
    st.subheader("ℹ️ MODEL INFORMATION")
    st.write(f"**Model type:** {type(model.named_steps['model']).__name__ if hasattr(model, 'named_steps') else type(model).__name__}")
    st.write("**Features used:**", ", ".join(ALL_FEATURES))
    
    # Simple market analysis
    st.divider()
    st.subheader("📈 MARKET ANALYSIS")
    
    # Generate sample market data
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="M")
    prices = np.random.normal(300000, 50000, len(dates)) + np.sin(np.arange(len(dates)) * 0.5) * 30000
    
    market_df = pd.DataFrame({
        "Date": dates,
        "Average Price": prices
    }).set_index("Date")
    
    st.line_chart(market_df)
    
    # Price distribution by bedrooms
    st.subheader("🏠 PRICE DISTRIBUTION BY BEDROOMS")
    bedroom_data = pd.DataFrame({
        "Bedrooms": [1, 2, 3, 4, 5],
        "Average Price": [200000, 275000, 350000, 450000, 600000]
    }).set_index("Bedrooms")
    
    st.bar_chart(bedroom_data) 