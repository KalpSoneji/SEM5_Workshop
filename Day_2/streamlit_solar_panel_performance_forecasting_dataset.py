import streamlit as st
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score, confusion_matrix
import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Solar Panel Performance Forecasting",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background-color: #FF6B35;
        color: white;
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">☀️ Solar Panel Performance Forecasting</h1>', unsafe_allow_html=True)

# Data Generation Functions
@st.cache_data
def generate_solar_data():
    """Generate synthetic solar panel data for all seasons"""
    
    # Feature ranges for different seasons
    feature_ranges = {
        'summer': {
            'irradiance': (600, 1000),
            'humidity': (10, 50),
            'wind_speed': (0, 5),
            'ambient_temperature': (30, 45),
            'tilt_angle': (10, 40),
        },
        'winter': {
            'irradiance': (300, 700),
            'humidity': (30, 70),
            'wind_speed': (1, 6),
            'ambient_temperature': (5, 20),
            'tilt_angle': (10, 40),
        },
        'monsoon': {
            'irradiance': (100, 600),
            'humidity': (70, 100),
            'wind_speed': (2, 8),
            'ambient_temperature': (20, 35),
            'tilt_angle': (10, 40),
        }
    }
    
    # Months for each season
    seasons_months = {
        'summer': {'March': 31, 'April': 30, 'May': 31, 'June': 30},
        'winter': {'November': 30, 'December': 31, 'January': 31, 'February': 28},
        'monsoon': {'July': 31, 'August': 31, 'September': 30, 'October': 31}
    }
    
    # kWh calculation functions
    def calc_kwh_summer(irradiance, humidity, wind_speed, ambient_temp, tilt_angle):
        return (0.25 * irradiance - 0.05 * humidity + 0.02 * wind_speed + 
                0.1 * ambient_temp - 0.03 * abs(tilt_angle - 30))
    
    def calc_kwh_winter(irradiance, humidity, wind_speed, ambient_temp, tilt_angle):
        return (0.18 * irradiance - 0.03 * humidity + 0.015 * wind_speed + 
                0.08 * ambient_temp - 0.02 * abs(tilt_angle - 30))
    
    def calc_kwh_monsoon(irradiance, humidity, wind_speed, ambient_temp, tilt_angle):
        return (0.15 * irradiance - 0.1 * humidity + 0.01 * wind_speed + 
                0.05 * ambient_temp - 0.04 * abs(tilt_angle - 30))
    
    calc_functions = {
        'summer': calc_kwh_summer,
        'winter': calc_kwh_winter,
        'monsoon': calc_kwh_monsoon
    }
    
    # Generate data
    all_data = []
    
    for season in ['summer', 'winter', 'monsoon']:
        for month, days in seasons_months[season].items():
            for _ in range(days):
                irr = np.random.uniform(*feature_ranges[season]['irradiance'])
                hum = np.random.uniform(*feature_ranges[season]['humidity'])
                wind = np.random.uniform(*feature_ranges[season]['wind_speed'])
                temp = np.random.uniform(*feature_ranges[season]['ambient_temperature'])
                tilt = np.random.uniform(*feature_ranges[season]['tilt_angle'])
                kwh = calc_functions[season](irr, hum, wind, temp, tilt)
                
                all_data.append({
                    'irradiance': round(irr, 2),
                    'humidity': round(hum, 2),
                    'wind_speed': round(wind, 2),
                    'ambient_temperature': round(temp, 2),
                    'tilt_angle': round(tilt, 2),
                    'kwh': round(kwh, 2),
                    'season': season,
                    'month': month
                })
    
    return pd.DataFrame(all_data)

# Load data
df = generate_solar_data()

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a page", 
                           ["📊 Dashboard", "🔮 Predictions", "🤖 Model Training", "📈 Analytics"])

if page == "📊 Dashboard":
    st.header("📊 Solar Panel Performance Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Data Points", len(df))
    with col2:
        st.metric("Average Daily Output", f"{df['kwh'].mean():.2f} kWh")
    with col3:
        st.metric("Peak Output", f"{df['kwh'].max():.2f} kWh")
    with col4:
        st.metric("Seasons Covered", df['season'].nunique())
    
    # Interactive plots
    st.subheader("📈 Interactive Visualizations")
    
    # Energy output by season
    fig_violin = px.violin(df, x='season', y='kwh', 
                          title="Energy Output Distribution by Season",
                          color='season')
    st.plotly_chart(fig_violin, use_container_width=True)
    
    # Monthly trends
    monthly_stats = df.groupby('month')['kwh'].agg(['mean', 'std']).reset_index()
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly_stats['month'] = pd.Categorical(monthly_stats['month'], 
                                          categories=month_order, ordered=True)
    monthly_stats = monthly_stats.sort_values('month')
    
    fig_monthly = px.bar(monthly_stats, x='month', y='mean', 
                        error_y='std', title="Monthly Average Energy Output",
                        color='mean', color_continuous_scale='viridis')
    st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Feature correlation
    st.subheader("🔗 Feature Correlations")
    corr_data = df[['irradiance', 'humidity', 'wind_speed', 
                   'ambient_temperature', 'tilt_angle', 'kwh']].corr()
    
    fig_heatmap = px.imshow(corr_data, text_auto=True, aspect="auto",
                           title="Feature Correlation Matrix")
    st.plotly_chart(fig_heatmap, use_container_width=True)

elif page == "🔮 Predictions":
    st.header("🔮 Solar Panel Performance Predictions")
    
    # Train model for predictions
    X = df[['irradiance', 'humidity', 'wind_speed', 'ambient_temperature', 'tilt_angle']]
    y = df['kwh']
    
    model = LinearRegression()
    model.fit(X, y)
    
    st.subheader("🎛️ Input Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        irradiance = st.slider("Solar Irradiance (W/m²)", 
                              min_value=100, max_value=1000, value=500)
        humidity = st.slider("Humidity (%)", 
                           min_value=10, max_value=100, value=50)
        wind_speed = st.slider("Wind Speed (m/s)", 
                             min_value=0, max_value=10, value=3)
    
    with col2:
        ambient_temp = st.slider("Ambient Temperature (°C)", 
                               min_value=0, max_value=50, value=25)
        tilt_angle = st.slider("Panel Tilt Angle (degrees)", 
                             min_value=0, max_value=90, value=30)
    
    # Make prediction
    input_data = np.array([[irradiance, humidity, wind_speed, ambient_temp, tilt_angle]])
    prediction = model.predict(input_data)[0]
    
    st.subheader("📊 Prediction Results")
    
    # Display prediction with nice formatting
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Predicted Energy Output", f"{prediction:.2f} kWh")
    
    with col2:
        # Efficiency calculation
        max_theoretical = irradiance * 0.25  # Simplified efficiency
        efficiency = (prediction / max_theoretical) * 100 if max_theoretical > 0 else 0
        st.metric("Estimated Efficiency", f"{efficiency:.1f}%")
    
    with col3:
        # Daily revenue estimate (assuming rate of ₹5 per kWh)
        revenue = prediction * 5
        st.metric("Estimated Daily Revenue", f"₹{revenue:.2f}")
    
    # Show input parameters effect
    st.subheader("📈 Parameter Impact Analysis")
    
    # Feature importance
    feature_names = ['Irradiance', 'Humidity', 'Wind Speed', 'Temperature', 'Tilt Angle']
    coefficients = model.coef_
    
    fig_importance = px.bar(x=feature_names, y=coefficients,
                           title="Feature Importance (Model Coefficients)",
                           color=coefficients,
                           color_continuous_scale='RdYlBu')
    st.plotly_chart(fig_importance, use_container_width=True)

elif page == "🤖 Model Training":
    st.header("🤖 Model Training & Evaluation")
    
    # Model selection
    model_type = st.selectbox("Select Model Type", 
                             ["Linear Regression", "Logistic Regression (Season Classification)"])
    
    if model_type == "Linear Regression":
        st.subheader("📊 Linear Regression for Energy Prediction")
        
        # Prepare data
        X = df[['irradiance', 'humidity', 'wind_speed', 'ambient_temperature', 'tilt_angle']]
        y = df['kwh']
        
        # Split data
        test_size = st.slider("Test Size (%)", min_value=10, max_value=40, value=20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", f"{r2:.4f}")
        with col2:
            st.metric("RMSE", f"{rmse:.4f}")
        with col3:
            st.metric("MSE", f"{mse:.4f}")
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Actual vs Predicted
            fig_scatter = px.scatter(x=y_test, y=y_pred, 
                                   title="Actual vs Predicted Energy Output",
                                   labels={'x': 'Actual kWh', 'y': 'Predicted kWh'})
            
            # Add perfect prediction line
            min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
            fig_scatter.add_scatter(x=[min_val, max_val], y=[min_val, max_val], 
                                  mode='lines', name='Perfect Prediction', 
                                  line=dict(dash='dash', color='red'))
            
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Residuals plot
            residuals = y_test - y_pred
            fig_residuals = px.scatter(x=y_pred, y=residuals,
                                     title="Residuals Plot",
                                     labels={'x': 'Predicted kWh', 'y': 'Residuals'})
            fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_residuals, use_container_width=True)
    
    else:  # Logistic Regression
        st.subheader("🎯 Logistic Regression for Season Classification")
        
        # Prepare data
        X = df[['irradiance', 'humidity', 'wind_speed', 'ambient_temperature', 'tilt_angle', 'kwh']]
        y = df['season']
        
        # Encode target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Split data
        test_size = st.slider("Test Size (%)", min_value=10, max_value=40, value=20, key="log_test_size") / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42)
        
        # Train model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        st.metric("Classification Accuracy", f"{accuracy:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                          title="Confusion Matrix",
                          labels=dict(x="Predicted Season", y="Actual Season"),
                          x=le.classes_, y=le.classes_)
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Classification report
        st.subheader("📋 Classification Report")
        report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        
        # Convert to DataFrame for better display
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(3))

elif page == "📈 Analytics":
    st.header("📈 Advanced Analytics")
    
    # Seasonal analysis
    st.subheader("🌍 Seasonal Performance Analysis")
    
    seasonal_stats = df.groupby('season').agg({
        'kwh': ['mean', 'std', 'min', 'max'],
        'irradiance': 'mean',
        'humidity': 'mean',
        'ambient_temperature': 'mean'
    }).round(2)
    
    # Flatten column names
    seasonal_stats.columns = ['_'.join(col).strip() for col in seasonal_stats.columns]
    st.dataframe(seasonal_stats)
    
    # Time series analysis
    st.subheader("📅 Time Series Analysis")
    
    # Add day number for time series
    df_time = df.copy()
    df_time['day_number'] = range(len(df_time))
    
    # Rolling average
    window_size = st.slider("Rolling Average Window (days)", min_value=3, max_value=30, value=7)
    df_time['rolling_avg'] = df_time['kwh'].rolling(window=window_size, center=True).mean()
    
    fig_time = px.line(df_time, x='day_number', y=['kwh', 'rolling_avg'],
                      title=f"Daily Energy Output with {window_size}-day Rolling Average",
                      labels={'value': 'Energy Output (kWh)', 'day_number': 'Day'})
    st.plotly_chart(fig_time, use_container_width=True)
    
    # Feature distribution analysis
    st.subheader("📊 Feature Distribution Analysis")
    
    feature_to_analyze = st.selectbox("Select Feature to Analyze", 
                                     ['irradiance', 'humidity', 'wind_speed', 
                                      'ambient_temperature', 'tilt_angle'])
    
    fig_dist = px.box(df, x='season', y=feature_to_analyze,
                     title=f"{feature_to_analyze.replace('_', ' ').title()} Distribution by Season")
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Correlation with energy output
    st.subheader("🔗 Feature vs Energy Output Correlation")
    
    fig_corr = px.scatter(df, x=feature_to_analyze, y='kwh', color='season',
                         title=f"{feature_to_analyze.replace('_', ' ').title()} vs Energy Output",
                         trendline="ols")
    st.plotly_chart(fig_corr, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("### 🌟 About This App")
st.markdown("""
This Solar Panel Performance Forecasting app uses machine learning to predict energy output based on environmental conditions.
The app includes:
- **Dashboard**: Overview of solar panel performance data
- **Predictions**: Interactive tool to predict energy output
- **Model Training**: Train and evaluate different ML models
- **Analytics**: Advanced data analysis and insights

Built with Streamlit, scikit-learn, and Plotly for an interactive experience.
""")

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Solar Energy Analytics")