#import packages

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objs as go 

##title of the app
st.title("Balance Your Budget")
st.markdown("*Where Preparation Meets Opportunity*")

##My expenses
st.markdown(
    """
    <div style="
        border: 2px solid pink;
        padding: 15px;
        border-radius: 10px;
        background-color: #ffe6f0;
        ">
        💖 My Expenses:
    </div>
    """,
    unsafe_allow_html=True
)


