# ============================================
# STREAMLIT APP FOR ICE CREAM DECISION TREE REGRESSOR
# File name: app.py
# Save this file as app.py in your project folder
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Ice Cream Predictor",
    page_icon="🍦",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        color: #ff6b6b;
    }
    .sub-header {
        font-size: 24px;
        text-align: center;
        color: #666;
        margin-bottom: 30px;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 20px 0;
    }
    .metric-value {
        font-size: 52px;
        font-weight: bold;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🍦 Ice Cream Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by Decision Tree Regressor</p>', unsafe_allow_html=True)

# Load model and metadata
@st.cache_resource
def load_model():
    try:
        with open('decision_tree_regressor_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            features = pickle.load(f)
        with open('target_name.pkl', 'rb') as f:
            target = pickle.load(f)
        return model, features, target
    except Exception as e:
        st.error(f"⚠️ Error loading model files: {e}")
        st.info("Please make sure these files are in the same folder as app.py:")
        st.code("""
- decision_tree_regressor_model.pkl
- feature_names.pkl
- target_name.pkl
        """)
        return None, None, None

model, feature_names, target_name = load_model()

if model is not None:
    # Create two columns for layout
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.markdown("### 🔧 Input Features")
        st.markdown("Adjust the controls to set your values:")
        
        # Dynamic input creation based on feature names
        input_values = {}
        
        for feature in feature_names:
            # Customize ranges based on feature name
            if 'temperature' in feature.lower() or 'temp' in feature.lower():
                input_values[feature] = st.slider(
                    f"🌡️ {feature}",
                    min_value=15.0,
                    max_value=40.0,
                    value=25.0,
                    step=0.5,
                    help="Temperature in degrees Celsius"
                )
            elif 'price' in feature.lower():
                input_values[feature] = st.slider(
                    f"💰 {feature}",
                    min_value=2.0,
                    max_value=10.0,
                    value=5.0,
                    step=0.25,
                    help="Price in dollars"
                )
            elif 'customer' in feature.lower():
                input_values[feature] = st.slider(
                    f"👥 {feature}",
                    min_value=50,
                    max_value=500,
                    value=200,
                    step=10,
                    help="Number of customers"
                )
            elif 'day' in feature.lower() or 'week' in feature.lower():
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                day_idx = st.selectbox(f"📅 {feature}", range(7), format_func=lambda x: day_names[x])
                input_values[feature] = day_idx
            elif 'rating' in feature.lower():
                input_values[feature] = st.slider(
                    f"⭐ {feature}",
                    min_value=1.0,
                    max_value=5.0,
                    value=4.0,
                    step=0.1,
                    help="Rating score"
                )
            else:
                # Generic numeric input
                input_values[feature] = st.slider(
                    f"📊 {feature}",
                    min_value=0.0,
                    max_value=100.0,
                    value=50.0,
                    step=1.0
                )
        
        # Predict button
        predict_button = st.button("🔮 Predict", type="primary", use_container_width=True)
    
    with col_right:
        # Create feature dataframe
        input_data = pd.DataFrame([input_values])
        
        if predict_button or 'initial_prediction' not in st.session_state:
            st.session_state['initial_prediction'] = True
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Display prediction
            st.markdown("### 🎯 Prediction Result")
            st.markdown(f"""
            <div class="prediction-box">
                <h3 style="margin: 0; font-weight: normal;">Predicted {target_name}</h3>
                <div class="metric-value">{prediction:.2f}</div>
                <p style="margin: 5px 0; opacity: 0.9;">Based on Decision Tree Regression Model</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature importance visualization
            st.markdown("### 📊 Feature Contribution")
            
            feature_importance = model.feature_importances_
            
            fig_importance = go.Figure(data=[
                go.Bar(
                    x=feature_names,
                    y=feature_importance * 100,
                    text=[f'{imp*100:.1f}%' for imp in feature_importance],
                    textposition='auto',
                    marker=dict(
                        color=feature_importance,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Importance")
                    )
                )
            ])
            
            fig_importance.update_layout(
                title="How Each Feature Affects Prediction",
                xaxis_title="Feature",
                yaxis_title="Importance (%)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # Input summary
            st.markdown("### 📋 Your Input Summary")
            
            # Create columns for metrics
            num_cols = len(feature_names)
            cols = st.columns(min(num_cols, 4))
            
            for i, (feature, value) in enumerate(input_values.items()):
                with cols[i % 4]:
                    st.metric(feature, f"{value:.2f}" if isinstance(value, float) else str(value))
            
            # Model information
            st.markdown("### 🤖 Model Information")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info(f"""
                **Max Depth**  
                {model.max_depth if model.max_depth else 'Unlimited'}
                """)
            
            with col2:
                st.info(f"""
                **Min Samples Split**  
                {model.min_samples_split}
                """)
            
            with col3:
                st.info(f"""
                **Min Samples Leaf**  
                {model.min_samples_leaf}
                """)

# Sidebar information
with st.sidebar:
    st.markdown("## ℹ️ About This App")
    st.markdown("""
    This application uses a **Decision Tree Regressor** to make predictions 
    based on multiple input features.
    
    ### 🎯 How It Works
    1. Adjust the input controls on the left
    2. Click "Predict" to get estimation
    3. View feature importance and analysis
    
    ### 📊 Decision Tree Regressor
    A Decision Tree learns patterns from data by creating a tree-like model 
    of decisions. Each branch represents a choice based on a feature value.
    
    ### 🔍 Features
    The model considers multiple factors to make accurate predictions. 
    The feature importance chart shows which factors matter most.
    """)
    
    st.markdown("---")
    st.markdown("### 🛠️ Technical Stack")
    st.markdown("""
    - **Framework**: Streamlit
    - **ML Library**: Scikit-learn
    - **Visualization**: Plotly
    - **Language**: Python 3.8+
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray; font-size: 12px;">
        Built with Streamlit<br>
        Decision Tree Regressor<br>
        Lab 09 Task 3
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>💡 <strong>Tip:</strong> Try adjusting different features to see how they affect the prediction!</p>
    <p style="font-size: 12px;">Machine Learning Project - Decision Tree Regression</p>
</div>
""", unsafe_allow_html=True)