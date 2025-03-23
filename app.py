import streamlit as st
import pandas as pd

# ✅ Set page config FIRST
st.set_page_config(page_title="Clark Admissions Assistant", layout="wide")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("Admissions Data of Clark.csv")

df = load_data()

# App content
st.title("🎓 Clark University Admissions Assistant")
st.markdown("Easily explore all the important undergraduate admissions information you need!")

# Sidebar filters
st.sidebar.header("🔍 Filter Options")
supercat = st.sidebar.selectbox("Supercategory", ["All"] + sorted(df['Supercategory'].dropna().unique()))
category = st.sidebar.selectbox("Category", ["All"] + sorted(df['Category'].dropna().unique()))
audience = st.sidebar.selectbox("Audience", ["All"] + sorted(df['Audience'].dropna().unique()))

# Filter logic
filtered = df.copy()
if supercat != "All":
    filtered = filtered[filtered['Supercategory'] == supercat]
if category != "All":
    filtered = filtered[filtered['Category'] == category]
if audience != "All":
    filtered = filtered[filtered['Audience'] == audience]

# Display results
st.markdown("### 📋 Filtered Results")
st.dataframe(filtered[["Category", "Subcategory", "Label", "Value", "Audience", "Required", "Details"]], use_container_width=True)

# Show source links
if st.checkbox("Show Source URLs"):
    for idx, row in filtered.iterrows():
        st.markdown(f"🔗 [{row['Label']}]({row['Source URL']})")
