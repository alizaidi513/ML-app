import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (mean_squared_error, r2_score, 
                            accuracy_score, confusion_matrix, 
                            silhouette_score)
from sklearn.impute import SimpleImputer
import base64
import time

# Set page config
st.set_page_config(
    page_title="Financial Data Analyzer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with professional background image and updated font styling
st.markdown("""
<style>
    /* Main background with professional finance image */
    .stApp {
        background-image: url("https://www.shutterstock.com/shutterstock/videos/28544560/thumb/1.jpg?ip=x480");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-blend-mode: overlay;
        background-color: rgba(245, 247, 250, 0.7);
    }

    /* Main content area with semi-transparent background */
    .main .block-container {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        margin-top: 2rem;
        margin-bottom: 2rem;
    }

    /* Sidebar with gradient */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50 0%, #1a252f 100%) !important;
        color: white !important;
    }

    /* Updated font styling for all text elements */
    body {
        font-size: 16px !important;
        color: #333333 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }

    /* Headers with modern font, larger size and darker color */
    h1, h2, h3, h4, h5, h6 {
        color: #222222 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    h1 {
        font-size: 2.2rem !important;
    }
    
    h2 {
        font-size: 1.8rem !important;
    }
    
    h3 {
        font-size: 1.5rem !important;
    }
    
    h4 {
        font-size: 1.3rem !important;
    }

    /* Regular text with larger size and darker color */
    .stMarkdown, .stText, .stAlert, .stDataFrame, .stSelectbox label, .stSlider label {
        font-size: 16px !important;
        color: #333333 !important;
    }

    /* Buttons with professional styling */
    .stButton>button {
        background: linear-gradient(45deg, #3498db 0%, #2980b9 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.75rem 1.5rem;
        margin: 0.5rem 0;
        transition: all 0.3s;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        font-size: 26px !important;
    }

    .stButton>button:hover {
        background: linear-gradient(45deg, #2980b9 0%, #3498db 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }

    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 1px solid #e1e4e8;
        background-color: white;
        font-size: 15px !important;
    }

    /* Metrics cards */
    .stMetric {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #3498db;
        transition: all 0.3s;
        font-size: 16px !important;
    }

    .stMetric:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }

    /* Alerts */
    .stAlert.st-success {
        background-color: #d4edda;
        color: #155724;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        font-size: 16px !important;
    }

    .stAlert.st-info {
        background-color: #d1ecf1;
        color: #0c5460;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        font-size: 16px !important;
    }

    .stAlert.st-warning {
        background-color: #fff3cd;
        color: #856404;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        font-size: 16px !important;
    }

    .stAlert.st-error {
        background-color: #f8d7da;
        color: #721c24;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        font-size: 16px !important;
    }

    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #3498db 0%, #2980b9 100%);
        border-radius: 4px;
    }

    /* Select boxes */
    .stSelectbox:first-child > div > div {
        border-radius: 8px !important;
        border: 1px solid #dfe6e9 !important;
        background-color: white !important;
        font-size: 16px !important;
    }

    /* File uploader */
    .stFileUploader > div > div {
        border-radius: 8px !important;
        border: 2px dashed #bdc3c7 !important;
        background: rgba(255, 255, 255, 0.7) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        padding: 10px 20px !important;
        transition: all 0.3s !important;
        background-color: #f1f3f5 !important;
        font-size: 16px !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #3498db !important;
        color: white !important;
    }
    
    /* Sidebar text */
    .css-1d391kg {
        font-size: 16px !important;
    }
    
    /* Input labels */
    .stTextInput label, .stNumberInput label, .stSelectbox label, .stSlider label {
        font-size: 16px !important;
        color: #222222 !important;
        font-weight: 500 !important;
    }
    
    /* Multiselect items */
    .stMultiSelect [role="button"] span {
        font-size: 16px !important;
    }
    
    /* Expander headers */
    .streamlit-expanderHeader {
        font-size: 18px !important;
        color: #222222 !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# Data loading function
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        st.success("Data loaded successfully!")
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# Preprocessing function
def preprocess_data(df):
    # Display missing values before processing
    st.subheader("Missing Values Before Processing")
    missing_data = df.isnull().sum()
    missing_count = missing_data[missing_data > 0]
    
    if len(missing_count) > 0:
        st.warning(f"Found {len(missing_count)} columns with missing values")
        st.dataframe(missing_count)
        
        # Show progress bar for missing value treatment
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(1, 4):
            progress_bar.progress(i * 33)
            status_text.text(f"Processing step {i}/3...")
            time.sleep(0.5)
    else:
        st.success("No missing values found in the dataset!")
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    
    if len(numeric_cols) > 0:
        imputer = SimpleImputer(strategy='mean')
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    if len(categorical_cols) > 0:
        imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = imputer.fit_transform(df[categorical_cols])
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Complete progress
    if len(missing_count) > 0:
        progress_bar.progress(100)
        status_text.text("Data preprocessing completed!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
    
    st.success("All missing values have been handled!")
    
    return df

# Feature engineering function
def perform_feature_engineering(df, target_col=None):
    st.subheader("Feature Engineering")
    
    # Display original features in an expander
    with st.expander("View Original Features"):
        st.write(df.columns.tolist())
    
    # If target column is specified, show correlation with numeric features
    if target_col and target_col in df.columns:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        if len(numeric_cols) > 0:
            st.subheader("Feature Correlation with Target")
            
            # Calculate correlation with animation
            with st.spinner("Calculating correlations..."):
                corr = df[numeric_cols + [target_col]].corr()[target_col].sort_values(ascending=False)
                time.sleep(1)  # Simulate calculation time
            
            # Display correlation in a nice visualization
            fig = px.bar(corr, x=corr.index, y=target_col, 
                         title=f"Correlation with {target_col}",
                         color=corr.values,
                         color_continuous_scale='blues')
            st.plotly_chart(fig, use_container_width=True)
    
    # Allow user to select features to keep with a nice multi-select
    all_features = df.columns.tolist()
    if target_col and target_col in all_features:
        all_features.remove(target_col)
    
    st.markdown("### Select Features to Keep")
    selected_features = st.multiselect(
        "Choose features for modeling:", 
        all_features, 
        default=all_features,
        help="Select the features you want to include in your model"
    )
    
    if target_col:
        selected_features.append(target_col)
    
    if len(selected_features) > 0:
        df = df[selected_features]
        st.success(f"✅ Selected {len(selected_features)} features for modeling.")
    else:
        st.warning("⚠️ No features selected. Keeping all features.")
    
    return df

# Model training function with enhanced UI
def train_model(model_type, X_train, y_train):
    # Identify numeric and categorical columns
    numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()
    
    # Show preprocessing steps
    with st.expander("View Preprocessing Steps"):
        st.markdown("**Numeric Columns:**")
        st.write(numeric_cols if numeric_cols else "No numeric columns found")
        
        st.markdown("**Categorical Columns:**")
        st.write(categorical_cols if categorical_cols else "No categorical columns found")
    
    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)])
    
    # Create model pipeline
    if model_type == "Linear Regression":
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())])
    elif model_type == "Logistic Regression":
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000))])
    elif model_type == "K-Means Clustering":
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('cluster', KMeans(n_clusters=3, random_state=42))])
    
    # Enhanced training progress
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    for percent_complete in range(100):
        time.sleep(0.02)  # Simulate training time
        progress_bar.progress(percent_complete + 1)
        progress_text.text(f"Training {model_type} model... {percent_complete + 1}%")
    
    if model_type == "K-Means Clustering":
        model.fit(X_train)
    else:
        model.fit(X_train, y_train)
    
    progress_text.empty()
    progress_bar.empty()
    
    st.success(f"🎉 {model_type} model trained successfully!")
    
    return model

# Enhanced evaluation function with better visuals
def evaluate_model(model, model_type, X_test, y_test=None):
    st.subheader("📊 Model Evaluation")
    
    if model_type in ["Linear Regression", "Logistic Regression"]:
        y_pred = model.predict(X_test)
        
        if model_type == "Linear Regression":
            # Create two columns for metrics
            col1, col2 = st.columns(2)
            
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            with col1:
                st.metric("Mean Squared Error", f"{mse:.4f}", 
                          help="Lower values indicate better performance")
            
            with col2:
                st.metric("R-squared Score", f"{r2:.4f}", 
                          delta=f"{r2*100:.1f}%",
                          help="Percentage of variance explained by the model")
            
            # Plot actual vs predicted with enhanced styling
            fig = px.scatter(x=y_test, y=y_pred, 
                            labels={'x': 'Actual', 'y': 'Predicted'},
                            title="Actual vs Predicted Values",
                            color_discrete_sequence=['#3498db'])
            fig.add_shape(type="line", x0=min(y_test), y0=min(y_test),
                          x1=max(y_test), y1=max(y_test),
                          line=dict(color="#e74c3c", width=3, dash="dot"))
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif model_type == "Logistic Regression":
            accuracy = accuracy_score(y_test, y_pred)
            
            # Display accuracy with gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = accuracy,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Accuracy Score"},
                gauge = {
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "#3498db"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "#e74c3c"},
                        {'range': [0.5, 0.8], 'color': "#f39c12"},
                        {'range': [0.8, 1], 'color': "#2ecc71"}]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            # Confusion matrix with better styling
            st.markdown("### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(cm, text_auto=True,
                            labels=dict(x="Predicted", y="Actual"),
                            title="Confusion Matrix",
                            color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
    
    elif model_type == "K-Means Clustering":
        # Get preprocessed data for visualization
        preprocessed_data = model.named_steps['preprocessor'].transform(X_test)
        
        if isinstance(preprocessed_data, np.ndarray):
            if preprocessed_data.shape[1] >= 2:
                clusters = model.named_steps['cluster'].labels_
                
                # Enhanced cluster visualization
                fig = px.scatter(x=preprocessed_data[:, 0], y=preprocessed_data[:, 1],
                                color=clusters, title="Cluster Visualization",
                                color_discrete_sequence=px.colors.qualitative.Pastel)
                fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Silhouette score with visual indicator
                score = silhouette_score(preprocessed_data, clusters)
                
                if score > 0.7:
                    st.success(f"Silhouette Score: {score:.4f} (Good separation)")
                elif score > 0.5:
                    st.info(f"Silhouette Score: {score:.4f} (Reasonable separation)")
                else:
                    st.warning(f"Silhouette Score: {score:.4f} (Weak separation)")
            else:
                st.warning("Need at least 2 features to visualize clusters.")
        else:
            st.warning("Could not visualize clusters with this data format.")

# Main app function with updated header
def main():
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'model_type' not in st.session_state:
        st.session_state.model_type = None
    if 'target_col' not in st.session_state:
        st.session_state.target_col = None
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center;">
            <h1 style="color: white; margin-bottom: 0;">💰</h1>
            <h2 style="color: white; margin-top: 0;">Financial Data Analyzer</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Data loading options
        st.header("📤 Data Upload")
        uploaded_file = st.file_uploader("Upload Dataset", type=['csv', 'xlsx'], 
                                        help="Upload your financial dataset in CSV or Excel format")
        
        if uploaded_file is not None:
            if st.button("Load Data", key="load_data", help="Click to process the uploaded file"):
                with st.spinner("Loading data..."):
                    st.session_state.df = load_data(uploaded_file)
                    if st.session_state.df is not None:
                        st.session_state.model = None
                        st.session_state.model_type = None
                        st.session_state.target_col = None
        
        # Model selection
        st.header("🤖 Model Selection")
        model_type = st.selectbox("Choose a model:", 
                                ["Linear Regression", "Logistic Regression", "K-Means Clustering"],
                                help="Select the type of model you want to train")
        
        # Display current status in a nice card
        st.header("📊 Current Status")
        status_card = st.container()
        
        if st.session_state.df is not None:
            status_card.success("✅ Data loaded")
            status_card.write(f"📐 Shape: {st.session_state.df.shape}")
            status_card.write(f"🔢 Numeric columns: {len(st.session_state.df.select_dtypes(include=np.number).columns)}")
            status_card.write(f"🔤 Categorical columns: {len(st.session_state.df.select_dtypes(exclude=np.number).columns)}")
        else:
            status_card.warning("⚠️ No data loaded")
        
        if st.session_state.model is not None:
            status_card.success(f"✅ {st.session_state.model_type} model trained")
    
    # Main content with professional header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%); 
                padding: 2rem; 
                border-radius: 10px;
                color: white;
                box-shadow: 0 4px 20px rgba(0,0,0,0.15);
                margin-bottom: 2rem;">
        <h1 style="color: white; text-align: center; margin: 0;">Financial Data Analysis Pipeline</h1>
        <p style="text-align: center; margin: 0.5rem 0 0 0; font-size: 1.1rem;">
            Advanced analytics for your financial data
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.info("""
        👋 Welcome to the Financial Data Analyzer!
        
        Please upload a dataset to begin analysis. Supported formats:
        - CSV files
        - Excel files
        """)
    else:
        # Step 1: Data Preview with tabs
        st.header("🔍 Step 1: Data Preview")
        
        tab1, tab2, tab3 = st.tabs(["Data Head", "Data Summary", "Data Types"])
        
        with tab1:
            st.dataframe(st.session_state.df.head())
        
        with tab2:
            st.write(st.session_state.df.describe())
        
        with tab3:
            st.write(pd.DataFrame({
                'Column': st.session_state.df.columns,
                'Type': st.session_state.df.dtypes,
                'Missing Values': st.session_state.df.isnull().sum()
            }))
        
        st.write(f"Dataset shape: {st.session_state.df.shape}")
        
        # Step 2: Preprocessing
        st.header("🧹 Step 2: Data Preprocessing")
        if st.button("Preprocess Data", key="preprocess", 
                    help="Handle missing values and clean the data"):
            with st.spinner("Preprocessing data..."):
                st.session_state.df = preprocess_data(st.session_state.df)
                st.success("Data preprocessing completed!")
        
        # Step 3: Feature Engineering
        st.header("⚙️ Step 3: Feature Engineering")
        
        # Select target column if regression model
        if model_type in ["Linear Regression", "Logistic Regression"]:
            numeric_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols) > 0:
                st.session_state.target_col = st.selectbox("Select target variable:", 
                                                         numeric_cols,
                                                         help="Choose the column you want to predict")
            else:
                st.error("No numeric columns available for target variable!")
        
        if st.button("Perform Feature Engineering", key="feature_eng",
                    help="Select features and analyze correlations"):
            with st.spinner("Performing feature engineering..."):
                st.session_state.df = perform_feature_engineering(
                    st.session_state.df, 
                    st.session_state.target_col if model_type in ["Linear Regression", "Logistic Regression"] else None
                )
                st.success("Feature engineering completed!")
        
        # Step 4: Train/Test Split
        st.header("✂️ Step 4: Train/Test Split")
        if model_type in ["Linear Regression", "Logistic Regression"] and st.session_state.target_col:
            test_size = st.slider("Test set size:", 0.1, 0.5, 0.2, 0.05,
                                help="Select the percentage of data to use for testing")
            
            if st.button("Split Data", key="split_data",
                        help="Divide data into training and testing sets"):
                X = st.session_state.df.drop(st.session_state.target_col, axis=1)
                y = st.session_state.df[st.session_state.target_col]
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                # Store in session state
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                
                # Enhanced split visualization
                fig = px.pie(values=[len(X_train), len(X_test)],
                             names=["Training Set", "Test Set"],
                             title="Train/Test Split",
                             color_discrete_sequence=['#3498db', '#2ecc71'])
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
                st.success("Data split completed!")
        
        elif model_type == "K-Means Clustering":
            if st.button("Prepare Data for Clustering", key="prepare_cluster",
                        help="Select numeric features for clustering"):
                X = st.session_state.df.select_dtypes(include=np.number)
                
                if X.shape[1] == 0:
                    st.error("No numeric features available for clustering.")
                    return
                
                st.session_state.X_train = X
                st.success("Data prepared for clustering!")
        
        # Step 5: Model Training
        st.header("🤖 Step 5: Model Training")
        if model_type in ["Linear Regression", "Logistic Regression"]:
            if 'X_train' in st.session_state and 'y_train' in st.session_state:
                if st.button(f"Train {model_type} Model", key="train_model",
                            help="Train the selected model on your data"):
                    st.session_state.model = train_model(
                        model_type,
                        st.session_state.X_train,
                        st.session_state.y_train
                    )
                    st.session_state.model_type = model_type
        elif model_type == "K-Means Clustering":
            if 'X_train' in st.session_state:
                if st.button("Train K-Means Model", key="train_cluster",
                            help="Perform clustering on your data"):
                    st.session_state.model = train_model(
                        model_type,
                        st.session_state.X_train,
                        None
                    )
                    st.session_state.model_type = model_type
        
        # Step 6: Model Evaluation
        st.header("📊 Step 6: Model Evaluation")
        if st.session_state.model is not None:
            if model_type in ["Linear Regression", "Logistic Regression"]:
                evaluate_model(
                    st.session_state.model,
                    model_type,
                    st.session_state.X_test,
                    st.session_state.y_test
                )
            elif model_type == "K-Means Clustering":
                evaluate_model(
                    st.session_state.model,
                    model_type,
                    st.session_state.X_train
                )
            
            # Download results with enhanced button
            st.header("💾 Download Results")
            if st.button("Export Model Predictions", key="export_results",
                        help="Download the model predictions as a CSV file"):
                if model_type in ["Linear Regression", "Logistic Regression"]:
                    y_pred = st.session_state.model.predict(st.session_state.X_test)
                    results = pd.DataFrame({
                        'Actual': st.session_state.y_test,
                        'Predicted': y_pred
                    })
                elif model_type == "K-Means Clustering":
                    clusters = st.session_state.model.predict(st.session_state.X_train)
                    results = pd.DataFrame({
                        'Cluster': clusters
                    })
                    results = pd.concat([st.session_state.X_train, results], axis=1)
                
                csv = results.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'''
                <a href="data:file/csv;base64,{b64}" download="model_results.csv" style="
                    display: inline-block;
                    padding: 0.75rem 1.5rem;
                    background: linear-gradient(45deg, #2ecc71 0%, #27ae60 100%);
                    color: white;
                    text-decoration: none;
                    border-radius: 8px;
                    font-weight: bold;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    transition: all 0.3s;
                ">
                    Download CSV File
                </a>
                '''
                st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()