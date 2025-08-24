#import packages

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objs as go 

## background 

st.markdown(
    """
    <style>
    /* Background for the main page */
    section[data-testid="stApp"] {
        background-color: #ffe6f0;
    }
    </style>
    """,
    unsafe_allow_html=True
)


##title of the app
st.markdown(
    """
    <h1 style='color:#ffb6c1;'>Balance Your Budget</h1>
    """,
    unsafe_allow_html=True
)

st.markdown("*Where Preparation Meets Opportunity*")

##My expenses
st.markdown(
    """
    <div style="
        border: 2px solid pink;
        padding: 15px;
        border-radius: 10px;
        background-color: #ffe6f0;
        font-size: 20px;   
        font-weight: bold; /* make text bold */
        ">
        💖 My Expenses:
    </div>
    """,
    unsafe_allow_html=True
)

## make a function to list each of the expenses and their amounts 

def list_my_expenses(my_expenses):
    for category, amount in my_expenses:
        st.markdown(
            f"""
            <div style="
                border: 2px solid pink;
                padding: 5px;
                border-radius: 10px;
                background-color: #fff5f7;
                font-size:18px;
                display: flex;
                justify-content: space-between;
                ">
               <span>{category}</span>
               <span>${amount}</span>
            </div>
            """,
            unsafe_allow_html=True
        )


## Now create my expenses so we can use them in the function

my_expenses = [
               ("Student Loans", 200), 
               ("Phone Bill", 150), 
               ("Car Insurance", 150), 
               ("Gas", 60), 
    
               ("Nails", 70), 
               ("Toes", 50), 
               ("Music", 11), 
               ("Gym", 15), 
               ("Wax", 75), ] 

list_my_expenses(my_expenses)

# Sum all amounts
total_expenses = sum(amount for category, amount in my_expenses)

st.markdown(
    f"""
    <div style="
        border: 2px solid pink;
        padding: 5px;
        border-radius: 10px;
        background-color: #fff5f7;
        font-size:18px;
        font-weight:bold;
        display: flex;
        justify-content: space-between;
        ">
        <span>Total Expenses</span>
        <span>${total_expenses}</span>
    </div>
    """,
    unsafe_allow_html=True
)
