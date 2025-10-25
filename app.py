import streamlit as st
import pandas as pd
import numpy as np
from apputil import GroupEstimate


st.write(
'''
# GroupEstimate: Building a Basic Model

This app demonstrates the `GroupEstimate` class that predicts values based on group statistics.

## What does it do?

- Takes categorical data and corresponding continuous values
- Determines which group a new observation falls into
- Predicts an estimate value (mean or median) based on the training data
''')

# Create tabs for different exercises
tab1, tab2, tab3 = st.tabs(["Exercise 1: Basic Usage", "Exercise 1: Example", "Bonus Exercise 2: Default Category"])

with tab1:
    st.header("Exercise 1: Basic GroupEstimate")
    
    st.write("""
    ### Create Your Own Data
    Upload data or use the default coffee example below.
    """)
    
    # Sample data
    default_data = pd.DataFrame({
        'loc_country': ['Guatemala', 'Guatemala', 'Mexico', 'Mexico', 'Brazil', 'Brazil', 'Guatemala', 'Mexico'],
        'roast': ['Light', 'Light', 'Medium', 'Medium', 'Dark', 'Medium', 'Medium', 'Light'],
        'rating': [88.0, 88.8, 91.0, 91.0, 85.0, 89.0, 90.0, 92.5]
    })
    
    st.write("#### Training Data")
    st.dataframe(default_data)
    
    # Select estimate type
    estimate_type = st.radio("Select Estimate Type:", ["mean", "median"], horizontal=True)
    
    # Select columns for grouping
    group_cols = st.multiselect(
        "Select columns to group by:",
        options=default_data.columns.tolist(),
        default=['loc_country', 'roast']
    )
    
    target_col = st.selectbox(
        "Select target column:",
        options=[col for col in default_data.columns if col not in group_cols],
        index=0
    )
    
    if st.button("Fit Model", key="fit1"):
        if len(group_cols) > 0:
            X = default_data[group_cols]
            y = default_data[target_col]
            
            gm = GroupEstimate(estimate=estimate_type)
            gm.fit(X, y)
            
            # Store in session state
            st.session_state['model1'] = gm
            st.session_state['group_cols1'] = group_cols
            
            st.success(f"Model fitted with {estimate_type} estimate!")
            
            # Show group statistics
            st.write("#### Group Statistics")
            stats_df = pd.DataFrame({
                'Group': [str(idx) for idx in gm.group_estimates_.index],
                f'{estimate_type.capitalize()}': gm.group_estimates_.values
            })
            st.dataframe(stats_df)
        else:
            st.error("Please select at least one grouping column!")
    
    # Prediction section
    if 'model1' in st.session_state:
        st.write("---")
        st.write("#### Make Predictions")
        
        # Create input fields dynamically
        pred_inputs = {}
        cols = st.columns(len(st.session_state['group_cols1']))
        
        for i, col_name in enumerate(st.session_state['group_cols1']):
            with cols[i]:
                unique_vals = default_data[col_name].unique().tolist()
                pred_inputs[col_name] = st.selectbox(
                    f"{col_name}:",
                    options=unique_vals + ["Other (not in training)"],
                    key=f"pred_{col_name}"
                )
        
        if st.button("Predict", key="predict1"):
            # Prepare prediction input
            X_pred = [[pred_inputs[col] for col in st.session_state['group_cols1']]]
            
            try:
                prediction = st.session_state['model1'].predict(X_pred)
                st.write(f"**Predicted {target_col}:** {prediction[0]:.2f}" if not np.isnan(prediction[0]) else f"**Predicted {target_col}:** NaN (group not found)")
            except Exception as e:
                st.error(f"Error: {e}")

with tab2:
    st.header("Coffee Review Example")
    
    st.write("""
    This example matches the one from the exercise notebook.
    
    We have coffee reviews with:
    - **Country of origin** (categorical)
    - **Roast type** (categorical)  
    - **Rating** (continuous)
    
    We'll predict ratings for new coffee combinations.
    """)
    
    # Create the example data
    df_raw = pd.DataFrame({
        'loc_country': ['Guatemala', 'Guatemala', 'Mexico', 'Mexico', 'Brazil', 'Brazil', 'Guatemala'],
        'roast': ['Light', 'Light', 'Medium', 'Medium', 'Dark', 'Medium', 'Medium'],
        'rating': [88.0, 88.8, 91.0, 91.0, 85.0, 89.0, 90.0]
    })
    
    st.write("#### Training Data")
    st.dataframe(df_raw)
    
    # Fit the model
    X = df_raw[["loc_country", "roast"]]
    y = df_raw["rating"]
    
    gm = GroupEstimate(estimate='mean')
    gm.fit(X, y)
    
    st.write("#### Model Fitted (using mean)")
    
    # Show predictions for example data
    st.write("#### Example Predictions")
    
    X_test = pd.DataFrame([
        ["Guatemala", "Light"],
        ["Mexico", "Medium"],
        ["Canada", "Dark"]
    ], columns=["loc_country", "roast"])
    
    st.dataframe(X_test)
    
    predictions = gm.predict(X_test)
    
    result_df = X_test.copy()
    result_df['Predicted Rating'] = predictions
    
    st.write("**Results:**")
    st.dataframe(result_df)
    
    st.info("""
    ℹ️ **Note:** The Canadian Dark roast returns NaN because this combination 
    doesn't exist in the training data.
    """)

with tab3:
    st.header("Bonus Exercise: Default Category Fallback")
    
    st.write("""
    When a specific combination is missing, we can fall back to estimates based on 
    a single category (like just the country, ignoring roast type).
    """)
    
    # Use same data
    df_raw = pd.DataFrame({
        'loc_country': ['Guatemala', 'Guatemala', 'Mexico', 'Mexico', 'Brazil', 'Brazil', 'Guatemala'],
        'roast': ['Light', 'Light', 'Medium', 'Medium', 'Dark', 'Medium', 'Medium'],
        'rating': [88.0, 88.8, 91.0, 91.0, 85.0, 89.0, 90.0]
    })
    
    st.write("#### Training Data")
    st.dataframe(df_raw)
    
    # Select default category
    default_cat = st.selectbox(
        "Select default category for fallback:",
        options=['loc_country', 'roast'],
        key="default_cat"
    )
    
    estimate_type2 = st.radio("Select Estimate Type:", ["mean", "median"], horizontal=True, key="est2")
    
    if st.button("Fit Model with Default Category", key="fit2"):
        X = df_raw[["loc_country", "roast"]]
        y = df_raw["rating"]
        
        gm = GroupEstimate(estimate=estimate_type2)
        gm.fit(X, y, default_category=default_cat)
        
        st.session_state['model2'] = gm
        st.success(f"Model fitted with {estimate_type2} estimate and default_category='{default_cat}'!")
        
        # Show default category statistics
        st.write(f"#### Default Category ({default_cat}) Statistics")
        default_stats = pd.DataFrame({
            default_cat: [str(idx) for idx in gm.default_estimates_.index],
            f'{estimate_type2.capitalize()}': gm.default_estimates_.values
        })
        st.dataframe(default_stats)
    
    if 'model2' in st.session_state:
        st.write("---")
        st.write("#### Test Predictions")
        
        test_cases = pd.DataFrame([
            ["Guatemala", "Light"],      # Exists: exact match
            ["Mexico", "Medium"],         # Exists: exact match
            ["Brazil", "Light"],          # Missing combo: will use Brazil average
            ["Canada", "Dark"]            # Missing country: will return NaN
        ], columns=["loc_country", "roast"])
        
        st.dataframe(test_cases)
        
        if st.button("Predict All Test Cases", key="predict2"):
            predictions = st.session_state['model2'].predict(test_cases)
            
            result_df = test_cases.copy()
            result_df['Predicted Rating'] = predictions
            result_df['Status'] = ['Exact match', 'Exact match', 'Used default category', 'Not found']
            
            st.write("**Results:**")
            st.dataframe(result_df)
            
            st.success("""
            ✅ **Explanation:**
            - Guatemala/Light & Mexico/Medium: Found exact combinations
            - Brazil/Light: Combination missing, used Brazil's average rating
            - Canada/Dark: Country not in data at all, returns NaN
            """)

# Footer
st.write("---")
st.write("""
### About the GroupEstimate Class

The `GroupEstimate` class provides:
- **`.fit(X, y, default_category=None)`**: Train on categorical features and target values
- **`.predict(X_)`**: Predict estimates for new observations
- **Handles missing groups**: Returns NaN or uses default category fallback
- **Flexible estimation**: Supports both mean and median
""")