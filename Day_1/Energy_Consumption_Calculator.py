import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

# Page configuration
st.set_page_config(
    page_title="🏠 Energy Consumption Tracker",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .energy-breakdown {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .stSelectbox > div > div {
        background-color: #f0f2f6;
        border-radius: 8px;
    }
    
    .stSlider > div > div {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for tracking consumption history
if 'consumption_history' not in st.session_state:
    st.session_state.consumption_history = []

# Main header
st.markdown("""
<div class="main-header">
    <h1>🏠 Smart Energy Consumption Tracker</h1>
    <p>Monitor and optimize your home's daily energy usage</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for input parameters
st.sidebar.header("🔧 Configuration")

# House type selection
house_type = st.sidebar.selectbox(
    "🏠 House Type:",
    ["Apartment", "Villa", "Row House", "Independent House", "Studio"],
    help="Select your house type"
)

# BHK input
bhk = st.sidebar.slider(
    "🛏️ Number of BHK:",
    min_value=1,
    max_value=5,
    value=2,
    help="Bedroom + Hall + Kitchen count"
)

# Appliance energy consumption (kWh)
st.sidebar.subheader("⚡ Appliance Energy Ratings (kWh)")
ac_energy = st.sidebar.number_input("AC Energy (kWh)", value=1.2, min_value=0.5, max_value=3.0, step=0.1)
fan_energy = st.sidebar.number_input("Fan Energy (kWh)", value=0.08, min_value=0.05, max_value=0.2, step=0.01)
light_energy = st.sidebar.number_input("Light Energy (kWh)", value=0.06, min_value=0.02, max_value=0.15, step=0.01)
wm_energy = st.sidebar.number_input("Washing Machine Energy (kWh)", value=1.0, min_value=0.5, max_value=2.0, step=0.1)

# Appliance count per room
st.sidebar.subheader("🔢 Appliances per Room")
ac_count = st.sidebar.slider("ACs per room", 0, 2, 1)
fan_count = st.sidebar.slider("Fans per room", 0, 3, 2)
light_count = st.sidebar.slider("Lights per room", 0, 5, 3)

# Usage tracking mode
st.sidebar.subheader("📅 Usage Tracking Mode")
tracking_mode = st.sidebar.radio(
    "Select tracking mode:",
    ["Daily Average", "Weekly Pattern"],
    help="Choose between daily average or weekly pattern tracking"
)

if tracking_mode == "Daily Average":
    # Usage hours
    st.sidebar.subheader("⏰ Daily Usage Hours")
    ac_hours = st.sidebar.slider("AC Usage (hours/day)", 0, 24, 8)
    fan_hours = st.sidebar.slider("Fan Usage (hours/day)", 0, 24, 12)
    light_hours = st.sidebar.slider("Light Usage (hours/day)", 0, 24, 6)
    wm_hours = st.sidebar.slider("Washing Machine Usage (hours/day)", 0, 8, 1)
    
    # Create weekly pattern from daily averages
    weekly_usage = {
        'Monday': {'ac': ac_hours, 'fan': fan_hours, 'light': light_hours, 'wm': wm_hours},
        'Tuesday': {'ac': ac_hours, 'fan': fan_hours, 'light': light_hours, 'wm': wm_hours},
        'Wednesday': {'ac': ac_hours, 'fan': fan_hours, 'light': light_hours, 'wm': wm_hours},
        'Thursday': {'ac': ac_hours, 'fan': fan_hours, 'light': light_hours, 'wm': wm_hours},
        'Friday': {'ac': ac_hours, 'fan': fan_hours, 'light': light_hours, 'wm': wm_hours},
        'Saturday': {'ac': ac_hours, 'fan': fan_hours, 'light': light_hours, 'wm': wm_hours},
        'Sunday': {'ac': ac_hours, 'fan': fan_hours, 'light': light_hours, 'wm': wm_hours}
    }
    
else:  # Weekly Pattern
    st.sidebar.subheader("📅 Weekly Usage Pattern")
    
    # Initialize weekly usage dictionary
    weekly_usage = {}
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Day selection for easy configuration
    selected_day = st.sidebar.selectbox("🗓️ Configure Day:", days)
    
    with st.sidebar.expander(f"⚙️ {selected_day} Settings", expanded=True):
        # Store usage for selected day
        if 'weekly_config' not in st.session_state:
            st.session_state.weekly_config = {
                day: {'ac': 8, 'fan': 12, 'light': 6, 'wm': 1} for day in days
            }
        
        # Special presets for different days
        if st.button(f"📋 Load {selected_day} Preset"):
            if selected_day in ['Saturday', 'Sunday']:
                # Weekend preset - more usage
                st.session_state.weekly_config[selected_day] = {'ac': 12, 'fan': 16, 'light': 8, 'wm': 2}
            elif selected_day == 'Monday':
                # Monday preset - less washing machine usage
                st.session_state.weekly_config[selected_day] = {'ac': 8, 'fan': 12, 'light': 6, 'wm': 0}
            else:
                # Weekday preset
                st.session_state.weekly_config[selected_day] = {'ac': 8, 'fan': 12, 'light': 6, 'wm': 1}
        
        # Individual appliance settings for selected day
        st.session_state.weekly_config[selected_day]['ac'] = st.slider(
            f"AC Usage", 0, 24, st.session_state.weekly_config[selected_day]['ac'], 
            key=f"ac_{selected_day}"
        )
        st.session_state.weekly_config[selected_day]['fan'] = st.slider(
            f"Fan Usage", 0, 24, st.session_state.weekly_config[selected_day]['fan'], 
            key=f"fan_{selected_day}"
        )
        st.session_state.weekly_config[selected_day]['light'] = st.slider(
            f"Light Usage", 0, 24, st.session_state.weekly_config[selected_day]['light'], 
            key=f"light_{selected_day}"
        )
        st.session_state.weekly_config[selected_day]['wm'] = st.slider(
            f"Washing Machine", 0, 8, st.session_state.weekly_config[selected_day]['wm'], 
            key=f"wm_{selected_day}"
        )
    
    # Copy the configuration to weekly_usage
    weekly_usage = st.session_state.weekly_config
    
    # Show quick overview of all days
    st.sidebar.subheader("📊 Week Overview")
    for day in days:
        wm_status = "🚫" if weekly_usage[day]['wm'] == 0 else f"✅({weekly_usage[day]['wm']}h)"
        st.sidebar.text(f"{day[:3]}: AC:{weekly_usage[day]['ac']}h, WM:{wm_status}")
    
    # Calculate average for current day calculation
    current_day = datetime.now().strftime('%A')
    if current_day in weekly_usage:
        ac_hours = weekly_usage[current_day]['ac']
        fan_hours = weekly_usage[current_day]['fan']
        light_hours = weekly_usage[current_day]['light']
        wm_hours = weekly_usage[current_day]['wm']
    else:
        ac_hours = fan_hours = light_hours = wm_hours = 0

# Calculate total rooms (BHK + 3 additional rooms: bathroom, kitchen, living room)
total_rooms = bhk + 3

# Calculate daily energy consumption
ac_consumption = total_rooms * ac_energy * ac_count * ac_hours
fan_consumption = total_rooms * fan_energy * fan_count * fan_hours
light_consumption = total_rooms * light_energy * light_count * light_hours
wm_consumption = wm_energy * wm_hours  # Washing machine is typically one per house

total_daily_energy = ac_consumption + fan_consumption + light_consumption + wm_consumption

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Today's Energy Consumption")
    
    # Display main metrics
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.metric(
            label="Total Energy",
            value=f"{total_daily_energy:.2f} kWh",
            delta=f"₹{total_daily_energy * 6:.2f}",
            help="Total daily energy consumption and estimated cost"
        )
    
    with col_b:
        st.metric(
            label="Total Rooms",
            value=f"{total_rooms}",
            delta=f"{bhk} BHK + 3",
            help="Total rooms in your house"
        )
    
    with col_c:
        st.metric(
            label="House Type",
            value=house_type,
            help="Your selected house type"
        )

with col2:
    st.subheader("💰 Cost Breakdown")
    
    # Assuming electricity rate of ₹6 per kWh
    electricity_rate = 6
    
    # Calculate weekly and monthly costs based on weekly pattern
    if tracking_mode == "Weekly Pattern":
        weekly_energy = sum([
            (total_rooms * ac_energy * ac_count * weekly_usage[day]['ac']) +
            (total_rooms * fan_energy * fan_count * weekly_usage[day]['fan']) +
            (total_rooms * light_energy * light_count * weekly_usage[day]['light']) +
            (wm_energy * weekly_usage[day]['wm'])
            for day in weekly_usage.keys()
        ])
        monthly_cost = weekly_energy * electricity_rate * 4.33  # Average weeks per month
        yearly_cost = weekly_energy * electricity_rate * 52
        
        st.info(f"**Today's Cost:** ₹{total_daily_energy * electricity_rate:.2f}")
        st.info(f"**Weekly Cost:** ₹{weekly_energy * electricity_rate:.2f}")
        st.info(f"**Monthly Cost:** ₹{monthly_cost:.2f}")
        st.info(f"**Yearly Cost:** ₹{yearly_cost:.2f}")
    else:
        monthly_cost = total_daily_energy * electricity_rate * 30
        yearly_cost = monthly_cost * 12
        
        st.info(f"**Daily Cost:** ₹{total_daily_energy * electricity_rate:.2f}")
        st.info(f"**Monthly Cost:** ₹{monthly_cost:.2f}")
        st.info(f"**Yearly Cost:** ₹{yearly_cost:.2f}")

# Weekly Pattern Visualization
if tracking_mode == "Weekly Pattern":
    st.subheader("📅 Weekly Energy Pattern")
    
    # Calculate daily consumption for each day of the week
    weekly_data = []
    for day in weekly_usage.keys():
        day_ac = total_rooms * ac_energy * ac_count * weekly_usage[day]['ac']
        day_fan = total_rooms * fan_energy * fan_count * weekly_usage[day]['fan']
        day_light = total_rooms * light_energy * light_count * weekly_usage[day]['light']
        day_wm = wm_energy * weekly_usage[day]['wm']
        day_total = day_ac + day_fan + day_light + day_wm
        
        weekly_data.append({
            'Day': day,
            'AC': day_ac,
            'Fan': day_fan,
            'Light': day_light,
            'Washing Machine': day_wm,
            'Total': day_total,
            'Cost': day_total * electricity_rate
        })
    
    df_weekly = pd.DataFrame(weekly_data)
    
    # Create stacked bar chart for weekly pattern
    fig_weekly = go.Figure()
    
    fig_weekly.add_trace(go.Bar(
        name='AC',
        x=df_weekly['Day'],
        y=df_weekly['AC'],
        marker_color='#ff6b6b'
    ))
    
    fig_weekly.add_trace(go.Bar(
        name='Fan',
        x=df_weekly['Day'],
        y=df_weekly['Fan'],
        marker_color='#4ecdc4'
    ))
    
    fig_weekly.add_trace(go.Bar(
        name='Light',
        x=df_weekly['Day'],
        y=df_weekly['Light'],
        marker_color='#45b7d1'
    ))
    
    fig_weekly.add_trace(go.Bar(
        name='Washing Machine',
        x=df_weekly['Day'],
        y=df_weekly['Washing Machine'],
        marker_color='#f9ca24'
    ))
    
    fig_weekly.update_layout(
        title='Weekly Energy Consumption Pattern',
        xaxis_title='Day of Week',
        yaxis_title='Energy (kWh)',
        barmode='stack',
        height=500
    )
    
    st.plotly_chart(fig_weekly, use_container_width=True)
    
    # Weekly usage table
    st.subheader("📋 Weekly Usage Table")
    
    # Create a more detailed weekly table
    weekly_display = df_weekly.copy()
    weekly_display['AC (h)'] = [weekly_usage[day]['ac'] for day in weekly_usage.keys()]
    weekly_display['Fan (h)'] = [weekly_usage[day]['fan'] for day in weekly_usage.keys()]
    weekly_display['Light (h)'] = [weekly_usage[day]['light'] for day in weekly_usage.keys()]
    weekly_display['WM (h)'] = [weekly_usage[day]['wm'] for day in weekly_usage.keys()]
    
    # Reorder columns for better display
    display_cols = ['Day', 'AC (h)', 'Fan (h)', 'Light (h)', 'WM (h)', 'Total', 'Cost']
    st.dataframe(
        weekly_display[display_cols].round(2),
        use_container_width=True
    )
    
    # Weekly insights
    st.subheader("🔍 Weekly Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        highest_day = df_weekly.loc[df_weekly['Total'].idxmax()]
        st.metric(
            "Highest Usage Day",
            highest_day['Day'],
            f"{highest_day['Total']:.2f} kWh"
        )
    
    with col2:
        lowest_day = df_weekly.loc[df_weekly['Total'].idxmin()]
        st.metric(
            "Lowest Usage Day",
            lowest_day['Day'],
            f"{lowest_day['Total']:.2f} kWh"
        )
    
    with col3:
        no_wm_days = len([day for day in weekly_usage.keys() if weekly_usage[day]['wm'] == 0])
        st.metric(
            "Days without WM",
            f"{no_wm_days} days",
            f"Save ₹{no_wm_days * wm_energy * electricity_rate:.2f}/week"
        )
    
    # Energy saving suggestions based on weekly pattern
    st.subheader("💡 Weekly Optimization Suggestions")
    
    suggestions = []
    
    # Find days with high AC usage
    high_ac_days = [day for day in weekly_usage.keys() if weekly_usage[day]['ac'] > 10]
    if high_ac_days:
        suggestions.append(f"🌡️ High AC usage detected on {', '.join(high_ac_days)}. Consider reducing by 1-2 hours.")
    
    # Find days with no washing machine usage
    no_wm_days = [day for day in weekly_usage.keys() if weekly_usage[day]['wm'] == 0]
    if no_wm_days:
        suggestions.append(f"✅ Great! No washing machine usage on {', '.join(no_wm_days)}. This saves energy!")
    
    # Find opportunities to consolidate washing machine usage
    wm_days = [day for day in weekly_usage.keys() if weekly_usage[day]['wm'] > 0]
    if len(wm_days) > 3:
        suggestions.append(f"🧺 Consider consolidating washing machine usage to fewer days to save energy.")
    
    # Weekend vs weekday analysis
    weekend_avg = (df_weekly[df_weekly['Day'].isin(['Saturday', 'Sunday'])]['Total'].mean())
    weekday_avg = (df_weekly[~df_weekly['Day'].isin(['Saturday', 'Sunday'])]['Total'].mean())
    
    if weekend_avg > weekday_avg * 1.2:
        suggestions.append(f"🏠 Weekend usage is {((weekend_avg/weekday_avg - 1) * 100):.1f}% higher than weekdays. Consider energy-saving activities.")
    
    for suggestion in suggestions:
        st.info(suggestion)

appliance_data = {
    'Appliance': ['Air Conditioner', 'Fans', 'Lights', 'Washing Machine'],
    'Energy (kWh)': [ac_consumption, fan_consumption, light_consumption, wm_consumption],
    'Cost (₹)': [ac_consumption * 6, fan_consumption * 6, light_consumption * 6, wm_consumption * 6]
}

df_breakdown = pd.DataFrame(appliance_data)

# Create pie chart
fig_pie = px.pie(
    df_breakdown, 
    values='Energy (kWh)', 
    names='Appliance',
    title="Energy Distribution by Appliance",
    color_discrete_sequence=['#667eea', '#764ba2', '#f093fb', '#f5576c']
)
fig_pie.update_layout(showlegend=True, height=400)

# Create bar chart
fig_bar = px.bar(
    df_breakdown, 
    x='Appliance', 
    y='Energy (kWh)',
    title="Energy Consumption by Appliance",
    color='Energy (kWh)',
    color_continuous_scale='Viridis'
)
fig_bar.update_layout(showlegend=False, height=400)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_pie, use_container_width=True)
with col2:
    st.plotly_chart(fig_bar, use_container_width=True)

# Weekly Pattern Visualization
if tracking_mode == "Weekly Pattern":
    st.subheader("📅 Weekly Energy Pattern")
    
    # Calculate daily consumption for each day of the week
    weekly_data = []
    for day in weekly_usage.keys():
        day_ac = total_rooms * ac_energy * ac_count * weekly_usage[day]['ac']
        day_fan = total_rooms * fan_energy * fan_count * weekly_usage[day]['fan']
        day_light = total_rooms * light_energy * light_count * weekly_usage[day]['light']
        day_wm = wm_energy * weekly_usage[day]['wm']
        day_total = day_ac + day_fan + day_light + day_wm
        
        weekly_data.append({
            'Day': day,
            'AC': day_ac,
            'Fan': day_fan,
            'Light': day_light,
            'Washing Machine': day_wm,
            'Total': day_total,
            'Cost': day_total * electricity_rate
        })
    
    df_weekly = pd.DataFrame(weekly_data)
    
    # Create stacked bar chart for weekly pattern
    fig_weekly = go.Figure()
    
    fig_weekly.add_trace(go.Bar(
        name='AC',
        x=df_weekly['Day'],
        y=df_weekly['AC'],
        marker_color='#ff6b6b'
    ))
    
    fig_weekly.add_trace(go.Bar(
        name='Fan',
        x=df_weekly['Day'],
        y=df_weekly['Fan'],
        marker_color='#4ecdc4'
    ))
    
    fig_weekly.add_trace(go.Bar(
        name='Light',
        x=df_weekly['Day'],
        y=df_weekly['Light'],
        marker_color='#45b7d1'
    ))
    
    fig_weekly.add_trace(go.Bar(
        name='Washing Machine',
        x=df_weekly['Day'],
        y=df_weekly['Washing Machine'],
        marker_color='#f9ca24'
    ))
    
    fig_weekly.update_layout(
        title='Weekly Energy Consumption Pattern',
        xaxis_title='Day of Week',
        yaxis_title='Energy (kWh)',
        barmode='stack',
        height=500
    )
    
    st.plotly_chart(fig_weekly, use_container_width=True)
    
    # Weekly usage table
    st.subheader("📋 Weekly Usage Table")
    
    # Create a more detailed weekly table
    weekly_display = df_weekly.copy()
    weekly_display['AC (h)'] = [weekly_usage[day]['ac'] for day in weekly_usage.keys()]
    weekly_display['Fan (h)'] = [weekly_usage[day]['fan'] for day in weekly_usage.keys()]
    weekly_display['Light (h)'] = [weekly_usage[day]['light'] for day in weekly_usage.keys()]
    weekly_display['WM (h)'] = [weekly_usage[day]['wm'] for day in weekly_usage.keys()]
    
    # Reorder columns for better display
    display_cols = ['Day', 'AC (h)', 'Fan (h)', 'Light (h)', 'WM (h)', 'Total', 'Cost']
    st.dataframe(
        weekly_display[display_cols].round(2),
        use_container_width=True
    )
    
    # Weekly insights
    st.subheader("🔍 Weekly Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        highest_day = df_weekly.loc[df_weekly['Total'].idxmax()]
        st.metric(
            "Highest Usage Day",
            highest_day['Day'],
            f"{highest_day['Total']:.2f} kWh"
        )
    
    with col2:
        lowest_day = df_weekly.loc[df_weekly['Total'].idxmin()]
        st.metric(
            "Lowest Usage Day",
            lowest_day['Day'],
            f"{lowest_day['Total']:.2f} kWh"
        )
    
    with col3:
        no_wm_days = len([day for day in weekly_usage.keys() if weekly_usage[day]['wm'] == 0])
        st.metric(
            "Days without WM",
            f"{no_wm_days} days",
            f"Save ₹{no_wm_days * wm_energy * electricity_rate:.2f}/week"
        )
    
    # Energy saving suggestions based on weekly pattern
    st.subheader("💡 Weekly Optimization Suggestions")
    
    suggestions = []
    
    # Find days with high AC usage
    high_ac_days = [day for day in weekly_usage.keys() if weekly_usage[day]['ac'] > 10]
    if high_ac_days:
        suggestions.append(f"🌡️ High AC usage detected on {', '.join(high_ac_days)}. Consider reducing by 1-2 hours.")
    
    # Find days with no washing machine usage
    no_wm_days = [day for day in weekly_usage.keys() if weekly_usage[day]['wm'] == 0]
    if no_wm_days:
        suggestions.append(f"✅ Great! No washing machine usage on {', '.join(no_wm_days)}. This saves energy!")
    
    # Find opportunities to consolidate washing machine usage
    wm_days = [day for day in weekly_usage.keys() if weekly_usage[day]['wm'] > 0]
    if len(wm_days) > 3:
        suggestions.append(f"🧺 Consider consolidating washing machine usage to fewer days to save energy.")
    
    # Weekend vs weekday analysis
    weekend_avg = (df_weekly[df_weekly['Day'].isin(['Saturday', 'Sunday'])]['Total'].mean())
    weekday_avg = (df_weekly[~df_weekly['Day'].isin(['Saturday', 'Sunday'])]['Total'].mean())
    
    if weekend_avg > weekday_avg * 1.2:
        suggestions.append(f"🏠 Weekend usage is {((weekend_avg/weekday_avg - 1) * 100):.1f}% higher than weekdays. Consider energy-saving activities.")
    
    for suggestion in suggestions:
        st.info(suggestion)

# Save daily consumption button
st.subheader("📅 Track Daily Consumption")

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("💾 Save Today's Consumption", type="primary"):
        consumption_record = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'day_of_week': datetime.now().strftime('%A'),
            'house_type': house_type,
            'bhk': bhk,
            'total_energy': total_daily_energy,
            'ac_energy': ac_consumption,
            'fan_energy': fan_consumption,
            'light_energy': light_consumption,
            'wm_energy': wm_consumption,
            'cost': total_daily_energy * electricity_rate,
            'tracking_mode': tracking_mode,
            'ac_hours': ac_hours,
            'fan_hours': fan_hours,
            'light_hours': light_hours,
            'wm_hours': wm_hours,
            'cost': total_daily_energy * electricity_rate
        }
        
        # Check if today's record already exists
        existing_record = False
        for i, record in enumerate(st.session_state.consumption_history):
            if record['date'] == consumption_record['date']:
                st.session_state.consumption_history[i] = consumption_record
                existing_record = True
                break
        
        if not existing_record:
            st.session_state.consumption_history.append(consumption_record)
        
        st.success("✅ Today's consumption saved successfully!")

with col2:
    if st.button("🗑️ Clear History"):
        st.session_state.consumption_history = []
        st.success("🧹 History cleared!")

# Display consumption history
if st.session_state.consumption_history:
    st.subheader("📈 Consumption History")
    
    # Convert to DataFrame for better display
    df_history = pd.DataFrame(st.session_state.consumption_history)
    df_history = df_history.sort_values('date', ascending=False)
    
    # Create trend chart
    fig_trend = px.line(
        df_history.sort_values('date'), 
        x='date', 
        y='total_energy',
        title="Energy Consumption Trend",
        markers=True
    )
    fig_trend.update_layout(height=400)
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Display history table
    st.dataframe(
        df_history[['date', 'day_of_week', 'house_type', 'bhk', 'total_energy', 'cost', 'tracking_mode', 'wm_hours']].round(2),
        use_container_width=True
    )
    
    # Day-wise analysis for weekly pattern users
    if any(record.get('tracking_mode') == 'Weekly Pattern' for record in st.session_state.consumption_history):
        st.subheader("📊 Day-wise Analysis")
        
        # Group by day of week
        df_history['day_of_week'] = pd.Categorical(
            df_history['day_of_week'], 
            categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            ordered=True
        )
        
        day_wise_avg = df_history.groupby('day_of_week').agg({
            'total_energy': 'mean',
            'cost': 'mean',
            'wm_hours': 'mean'
        }).round(2)
        
        # Create day-wise consumption chart
        fig_daywise = px.bar(
            day_wise_avg.reset_index(),
            x='day_of_week',
            y='total_energy',
            title='Average Energy Consumption by Day of Week',
            color='total_energy',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_daywise, use_container_width=True)
        
        # Show day-wise statistics
        st.subheader("📈 Day-wise Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Average Energy by Day:**")
            st.dataframe(day_wise_avg, use_container_width=True)
        
        with col2:
            # Find patterns
            st.write("**Weekly Patterns:**")
            
            # Days with no washing machine usage
            no_wm_days = df_history[df_history['wm_hours'] == 0]['day_of_week'].value_counts()
            if not no_wm_days.empty:
                st.write("🚫 **Days without Washing Machine:**")
                for day, count in no_wm_days.items():
                    st.write(f"   • {day}: {count} times")
            
            # Highest and lowest consumption days
            highest_day = day_wise_avg['total_energy'].idxmax()
            lowest_day = day_wise_avg['total_energy'].idxmin()
            
            st.write(f"📈 **Highest avg consumption:** {highest_day}")
            st.write(f"📉 **Lowest avg consumption:** {lowest_day}")
            
            # Weekend vs weekday pattern
            weekend_days = ['Saturday', 'Sunday']
            weekend_avg = df_history[df_history['day_of_week'].isin(weekend_days)]['total_energy'].mean()
            weekday_avg = df_history[~df_history['day_of_week'].isin(weekend_days)]['total_energy'].mean()
            
            if not pd.isna(weekend_avg) and not pd.isna(weekday_avg):
                diff_pct = ((weekend_avg - weekday_avg) / weekday_avg) * 100
                st.write(f"🏠 **Weekend vs Weekday:** {diff_pct:.1f}% {'higher' if diff_pct > 0 else 'lower'}")
    
    # Enhanced summary statistics
    st.subheader("📊 Enhanced Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average Daily", f"{df_history['total_energy'].mean():.2f} kWh")
    with col2:
        st.metric("Highest Day", f"{df_history['total_energy'].max():.2f} kWh")
    with col3:
        st.metric("Lowest Day", f"{df_history['total_energy'].min():.2f} kWh")
    with col4:
        st.metric("Total Cost", f"₹{df_history['cost'].sum():.2f}")
    
    # Additional insights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        wm_usage_days = len(df_history[df_history['wm_hours'] > 0])
        st.metric("Days with WM Usage", f"{wm_usage_days}")
    
    with col2:
        avg_wm_hours = df_history[df_history['wm_hours'] > 0]['wm_hours'].mean()
        if not pd.isna(avg_wm_hours):
            st.metric("Avg WM Hours", f"{avg_wm_hours:.1f}h")
        else:
            st.metric("Avg WM Hours", "0h")
    
    with col3:
        total_wm_savings = len(df_history[df_history['wm_hours'] == 0]) * wm_energy * electricity_rate
        st.metric("WM Savings", f"₹{total_wm_savings:.2f}")
    
    # Weekly pattern insights
    if len(df_history) >= 7:
        st.subheader("🔍 Weekly Pattern Insights")
        
        # Calculate weekly totals
        df_history['week'] = pd.to_datetime(df_history['date']).dt.isocalendar().week
        weekly_totals = df_history.groupby('week').agg({
            'total_energy': 'sum',
            'cost': 'sum'
        }).round(2)
        
        if len(weekly_totals) > 1:
            # Weekly trend
            fig_weekly_trend = px.line(
                weekly_totals.reset_index(),
                x='week',
                y='total_energy',
                title='Weekly Energy Consumption Trend',
                markers=True
            )
            st.plotly_chart(fig_weekly_trend, use_container_width=True)
            
            # Weekly statistics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Weekly", f"{weekly_totals['total_energy'].mean():.2f} kWh")
            with col2:
                st.metric("Weekly Cost Avg", f"₹{weekly_totals['cost'].mean():.2f}")
    
    # Monthly projection
    if len(df_history) >= 3:
        st.subheader("📅 Monthly Projection")
        
        days_tracked = len(df_history)
        avg_daily = df_history['total_energy'].mean()
        monthly_projection = avg_daily * 30
        monthly_cost_projection = monthly_projection * electricity_rate
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Days Tracked", f"{days_tracked}")
        with col2:
            st.metric("Monthly Projection", f"{monthly_projection:.2f} kWh")
        with col3:
            st.metric("Projected Monthly Cost", f"₹{monthly_cost_projection:.2f}")
    
    # Energy efficiency score
    st.subheader("⭐ Energy Efficiency Score")
    
    # Calculate efficiency based on various factors
    efficiency_score = 100
    
    # Deduct points for high consumption
    avg_consumption = df_history['total_energy'].mean()
    if avg_consumption > 20:
        efficiency_score -= 20
    elif avg_consumption > 15:
        efficiency_score -= 10
    elif avg_consumption > 10:
        efficiency_score -= 5
    
    # Add points for consistent washing machine patterns
    wm_zero_days = len(df_history[df_history['wm_hours'] == 0])
    if wm_zero_days > 0:
        efficiency_score += min(15, wm_zero_days * 2)
    
    # Add points for consistent usage patterns
    std_dev = df_history['total_energy'].std()
    if std_dev < 2:
        efficiency_score += 10
    elif std_dev < 5:
        efficiency_score += 5
    
    # Cap at 100
    efficiency_score = min(100, efficiency_score)
    
    # Display efficiency score with color coding
    if efficiency_score >= 80:
        st.success(f"🌟 Excellent! Your efficiency score is {efficiency_score}%")
    elif efficiency_score >= 60:
        st.info(f"👍 Good! Your efficiency score is {efficiency_score}%")
    else:
        st.warning(f"⚠️ Room for improvement. Your efficiency score is {efficiency_score}%")
    
    # Efficiency recommendations
    recommendations = []
    
    if avg_consumption > 15:
        recommendations.append("🔋 Consider reducing overall energy consumption by optimizing appliance usage")
    
    if wm_zero_days == 0:
        recommendations.append("🧺 Try to have washing-machine-free days to save energy")
    
    if std_dev > 5:
        recommendations.append("📊 Try to maintain consistent daily energy consumption patterns")
    
    if len(df_history[df_history['wm_hours'] > 2]) > 0:
        recommendations.append("⏰ Consider reducing washing machine usage to 1-2 hours per day")
    
    if recommendations:
        st.subheader("💡 Personalized Recommendations")
        for rec in recommendations:
            st.info(rec)

# Energy saving tips
st.subheader("💡 Energy Saving Tips")

tips = [
    "🌡️ Set AC temperature to 24°C for optimal efficiency",
    "🔄 Use ceiling fans along with AC to circulate air better",
    "💡 Replace traditional bulbs with LED lights",
    "🕐 Use appliances during off-peak hours when possible",
    "🏠 Ensure proper insulation to reduce AC load",
    "🌞 Use natural light during daytime",
    "🔌 Unplug electronics when not in use",
    "🧺 Use washing machine with full loads only"
]

for tip in tips[:4]:  # Show first 4 tips
    st.info(tip)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 1rem;">
        🌱 Track your energy consumption and contribute to a greener future!
    </div>
    """,
    unsafe_allow_html=True
)