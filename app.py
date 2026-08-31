import streamlit as st
import pandas as pd
from google import genai


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="DataMate AI",
    page_icon="📊",
    layout="wide"
)


# ==========================================
# Load Gemini API Key from secrets.toml
# ==========================================

try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = None


# ==========================================
# App Header
# ==========================================

st.title("📊 DataMate AI")

st.subheader(
    "AI-powered data analysis and visualizations"
)


# ==========================================
# Sidebar API Status
# ==========================================

if api_key:
    st.sidebar.success("✅ Gemini API Connected")
else:
    st.sidebar.error("❌ Gemini API Key not found")


# ==========================================
# CSV File Upload
# ==========================================

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)


# ==========================================
# Main Application
# ==========================================

if uploaded_file is not None:

    try:

        # Read CSV
        df = pd.read_csv(uploaded_file)

        # Upload Success
        st.success("Data uploaded successfully!")


        # ==========================================
        # Gemini AI Analysis
        # ==========================================

        st.header("🤖 Gemini AI Data Analyst")

        if api_key:

            if st.button(
                "🚀 Analyze Data with Gemini AI",
                use_container_width=True
            ):

                try:

                    # Create Gemini Client
                    client = genai.Client(
                        api_key=api_key
                    )


                    # Take first 20 rows
                    data_sample = df.head(20).to_csv(
                        index=False
                    )


                    # AI Prompt
                    prompt = f"""
You are an expert data analyst.

Analyze the following CSV dataset and provide a professional analysis.

Use Markdown formatting with clear headings.

Please provide:

## 1. Important Insights
Analyze the most important patterns and findings.

## 2. Data Quality Issues
Identify possible duplicate data, inconsistent values, or other quality problems.

## 3. Missing Values Analysis
Explain missing values and their possible impact.

## 4. Important Trends
Identify important trends and patterns in the dataset.

## 5. Business Recommendations
Provide practical and actionable business recommendations.

Dataset Information:

- Total Rows: {df.shape[0]}
- Total Columns: {df.shape[1]}

Column Names:

{list(df.columns)}

Dataset Sample:

{data_sample}
"""


                    # Gemini Analysis
                    with st.spinner(
                        "🤖 Gemini is analyzing your data..."
                    ):

                        response = client.models.generate_content(
                            model="gemini-3.6-flash",
                            contents=prompt
                        )


                    # Success Message
                    st.success(
                        "🎉 Analysis completed successfully!"
                    )


                    # Divider
                    st.divider()


                    # AI Insights Title
                    st.subheader(
                        "🧠 AI Insights"
                    )


                    # Markdown Output
                    st.markdown(
                        response.text
                    )


                    # Bottom Divider
                    st.divider()


                except Exception as e:

                    st.error(
                        f"❌ Gemini API Error: {e}"
                    )


        else:

            st.warning(
                "⚠️ Gemini API Key not found. "
                "Please add it to .streamlit/secrets.toml"
            )


        # ==========================================
        # Dataset Preview
        # ==========================================

        st.header("📋 Dataset Preview")

        st.dataframe(
            df,
            use_container_width=True
        )


        # ==========================================
        # Dataset Information
        # ==========================================

        st.header("📊 Dataset Information")


        col1, col2, col3 = st.columns(3)


        col1.metric(
            "Rows",
            df.shape[0]
        )


        col2.metric(
            "Columns",
            df.shape[1]
        )


        col3.metric(
            "Missing Values",
            df.isnull().sum().sum()
        )


        # ==========================================
        # Data Summary
        # ==========================================

        st.header("📈 Data Summary")


        st.dataframe(
            df.describe(
                include="all"
            ),
            use_container_width=True
        )


        # ==========================================
        # Data Visualizations
        # ==========================================

        st.header("📉 Data Visualizations")


        # Find Numeric Columns
        numeric_columns = df.select_dtypes(
            include=[
                "int64",
                "float64",
                "int32",
                "float32"
            ]
        ).columns.tolist()


        # Find Categorical Columns
        categorical_columns = df.select_dtypes(
            include=[
                "object",
                "category"
            ]
        ).columns.tolist()


        # ==========================================
        # Numeric Chart
        # ==========================================

        if len(numeric_columns) > 0:

            st.subheader(
                "📊 Numeric Data"
            )


            selected_numeric = st.selectbox(
                "Select numeric column",
                numeric_columns
            )


            st.bar_chart(
                df[selected_numeric]
            )


        # ==========================================
        # Categorical Chart
        # ==========================================

        if len(categorical_columns) > 0:

            st.subheader(
                "📊 Categorical Data"
            )


            selected_category = st.selectbox(
                "Select category",
                categorical_columns
            )


            category_count = df[
                selected_category
            ].value_counts()


            st.bar_chart(
                category_count
            )


        # ==========================================
        # Download CSV
        # ==========================================

        st.header("⬇️ Download Data")


        csv = df.to_csv(
            index=False
        ).encode("utf-8")


        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name="analyzed_data.csv",
            mime="text/csv",
            use_container_width=True
        )


    except Exception as e:

        st.error(
            f"❌ Error: {e}"
        )


# ==========================================
# No File Uploaded
# ==========================================

else:

    st.info(
        "📁 Please upload a CSV file to start analyzing your data."
    )