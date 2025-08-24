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

# Add space between sections
st.markdown("<br><br>", unsafe_allow_html=True)

############# INCOME

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
        💖 My Income:
    </div>
    """,
    unsafe_allow_html=True
)

## make a function to list income

my_income = [
               ("Income", 1600) ]


def list_my_income(my_income):
    for income, income_amount in my_income:
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
               <span>{inccome}</span>
               <span>${income_amount}</span>
            </div>
            """,
            unsafe_allow_html=True
        )











## Break it down

# Define categories
bills = ["Student Loans", "Phone Bill", "Car Insurance", "Gas"]
maintenance = ["Nails", "Toes", "Wax", "Gym", "Music"]

# Calculate totals
total_bills = sum(amount for category, amount in my_expenses if category in bills)
total_maintenance = sum(amount for category, amount in my_expenses if category in maintenance)
total_fun_money = sum(income - total_bills - total_maintenance)

# Display table
st.markdown(
    f"""
    <table style="border: 2px solid pink; border-collapse: collapse; width: 60%; text-align:center;">
        <tr style="background-color:#ffe6f0;">
            <th style="border: 1px solid pink; padding: 8px;">Bills</th>
            <th style="border: 1px solid pink; padding: 8px;">Maintenance</th>
            <th style="border: 1px solid pink; padding: 8px;">Fun Money</th>
        </tr>
        <tr>
            <td style="border: 1px solid pink; padding: 8px;">${total_bills}</td>
            <td style="border: 1px solid pink; padding: 8px;">${total_maintenance}</td>
            <td style="border: 1px solid pink; padding: 8px;">${total_fun_money}</td>
        </tr>
    </table>
    """,
    unsafe_allow_html=True
)
