
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

tab1, tab2 = st.tabs(["Try Out Method", "Other Tab"])

with tab1:
    st.subheader("Methods used")
    method = st.radio(
        "Select Method",
        ["Mean Variance Optimization", "Risk Parity"]
    )
    if method == "Mean Variance Optimization":
        weights_df = pd.read_csv("Optimal_Weights(MVO).csv")  
    elif method == "Risk Parity":
        weights_df = pd.read_csv("Optimal_Weights(RP).csv") 


    ### bar chart
    weights = weights_df.set_index("Ticker")
    weights_nonzero = weights[weights["Optimized_Weights"] > 0]
    st.title(method)

    col1, col2 = st.columns([1,2])  
    with col1:
        st.subheader("Weights Table")
        st.dataframe(weights, height=300)

    with col2:
        st.subheader("Bar Chart")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(weights_nonzero.index.astype(str), 
            weights_nonzero["Optimized_Weights"], 
            color='skyblue', width=0.8)
        ax.set_xlabel("Ticker")
        ax.set_ylabel("Weight")
        ax.set_title("Optimal Weights")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)


    ### metric
    df = pd.read_csv("Top30_Log_Returns_Corrected.csv", index_col = 0)
    w = weights_df["Optimized_Weights"]
    mean = df.mean().values * 252
    cov_matrix = df.cov().values * 252
    port_return = np.dot(w, mean)*100
    port_risk = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))*100

    col1, col2 = st.columns([1,1]) 
    with col1:
        st.metric(label="Returns", value=f"{port_return:.2f}%")

    with col2:
        st.metric(label="Risks", value=f"{port_risk:.2f}%")



with tab2:
    st.metric(label="Capital", value="1,000,000")

    stocks = pd.read_csv("Top30_Most_Stable_Stocks_Corrected.csv")  
    industry = stocks[["Ticker", "Industry"]]
    data = pd.merge(weights_df, industry, on="Ticker")
    data = data[data["Optimized_Weights"] > 0]
    num_industries = data["Industry"].nunique()
    st.metric(label="Number of Industries", value=num_industries)