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

## make a function that has a header "My Expenses" in a box and all my listed expenses below it in rows 

def show_expense_box(expenses, title="💖 My Expenses:"):
  
    # Start the main container div
    html = f"""
    <div style="
        border: 2px solid pink;
        padding: 15px;
        border-radius: 10px;
        background-color: #fff5f7;
        font-size:18px;
    ">
        <div style="font-weight:bold; margin-bottom:10px;">{title}</div>
    """

    # Add each expense as a flex row
    for category, amount in expenses:
        html += f"""
        <div style="display: flex; justify-content: space-between; margin-bottom:5px;">
            <span>{category}</span>
            <span style="font-weight:bold;">${amount}</span>
        </div>
        """

    # Close the main div
    html += "</div>"

    st.markdown(html, unsafe_allow_html=True)

## Now create my expenses so we can use them in the function 

my_expenses = [
    ("Nails", 70),
    ("Toes", 50),
    ("Student Loans", 200),
    ("Phone Bill", 150),
    ("Car Insurance", 150),
    ("Gas", 60),
    ("Gym", 15),
    ("Music", 15),
    ("Wax", 15),
]

show_expense_box(my_expenses)
