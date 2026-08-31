import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="DataMate AI",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 DataMate AI")
st.subheader("Transform raw data into insights and visualizations")

# File uploader
uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)

        st.success("Data uploaded successfully!")

        # Dataset Preview
        st.header("📋 Dataset Preview")
        st.dataframe(df, use_container_width=True)

        # Dataset Information
        st.header("📊 Dataset Information")

        col1, col2, col3 = st.columns(3)

        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())

        # Data Summary
        st.header("📈 Data Summary")
        st.dataframe(df.describe(include="all"), use_container_width=True)

        # Charts section
        st.header("📉 Data Visualizations")

        numeric_columns = df.select_dtypes(
            include=["int64", "float64", "int32", "float32"]
        ).columns.tolist()

        categorical_columns = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # Numeric charts
        if len(numeric_columns) > 0:

            st.subheader("📊 Numeric Data Analysis")

            selected_numeric = st.selectbox(
                "Select a numeric column",
                numeric_columns
            )

            st.bar_chart(df[selected_numeric])

            st.subheader("Distribution")
            st.line_chart(df[selected_numeric])

        # Category analysis
        if len(categorical_columns) > 0:

            st.subheader("🏢 Category Analysis")

            selected_category = st.selectbox(
                "Select a category",
                categorical_columns
            )

            category_count = df[selected_category].value_counts()

            st.bar_chart(category_count)

        # Correlation
        if len(numeric_columns) > 1:

            st.subheader("🔥 Correlation Matrix")

            correlation = df[numeric_columns].corr()

            st.dataframe(
                correlation.style.background_gradient(cmap="Blues"),
                use_container_width=True
            )

        # Download cleaned data
        st.header("⬇️ Download Data")

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="analyzed_data.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error reading file: {e}")

else:
    st.info("Please upload a CSV file to start analyzing your data.")