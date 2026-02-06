import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Nifty Gap Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1e1e4a 0%, #2a2a5a 100%);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        text-align: center;
        margin: 10px 0;
    }
    
    .metric-value {
        font-size: 72px;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .metric-label {
        font-size: 18px;
        color: #a0a0c0;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 8px;
    }
    
    .dashboard-title {
        font-size: 56px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 50%, #ff006e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        text-shadow: 0 0 40px rgba(0, 212, 255, 0.3);
    }
    
    .dashboard-subtitle {
        font-size: 22px;
        color: #a0a0c0;
        text-align: center;
        margin-bottom: 30px;
    }
    
    /* Insight boxes */
    .insight-box {
        background: linear-gradient(145deg, rgba(0, 212, 255, 0.1) 0%, rgba(124, 58, 237, 0.1) 100%);
        border-left: 4px solid #00d4ff;
        border-radius: 8px;
        padding: 16px;
        margin: 16px 0;
    }
    
    .insight-title {
        color: #00d4ff;
        font-weight: 600;
        font-size: 16px;
    }
    
    .insight-text {
        color: #e0e0e0;
        font-size: 14px;
        margin-top: 8px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a3e 0%, #0f0f23 100%);
    }
    
    /* Charts container */
    .chart-container {
        background: rgba(30, 30, 74, 0.5);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin: 10px 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 30, 74, 0.5);
        border-radius: 8px;
        padding: 10px 20px;
        color: #a0a0c0;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    # Read from source CSV
    df = pd.read_csv(r'c:\Users\rohit\OneDrive\Desktop\TradersCafe\nifty data 5 yrs.csv', encoding='utf-8')
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Calculate gap percentage
    df['Prev_Close'] = df['Close'].shift(1)
    df['Gap_%'] = ((df['Open'] - df['Prev_Close']) / df['Prev_Close']) * 100
    
    # Filter for gaps > 1%
    gap_up = df[df['Gap_%'] > 1].copy()
    gap_down = df[df['Gap_%'] < -1].copy()
    gap_up['Gap_Type'] = 'Gap Up'
    gap_down['Gap_Type'] = 'Gap Down'
    
    # Analyze gap behavior
    def analyze_gap_behavior(row):
        open_price, high, low, close = row['Open'], row['High'], row['Low'], row['Close']
        prev_close, gap_type = row['Prev_Close'], row['Gap_Type']
        close_vs_open_pct = ((close - open_price) / open_price) * 100
        
        if gap_type == 'Gap Up':
            gap_filled = 'Gap Filled' if low <= prev_close else 'Gap Held'
            if low <= prev_close and close < prev_close:
                verdict = 'Complete Gap Fill - Closed below Prev Close'
            elif low <= prev_close and close >= prev_close:
                verdict = 'Gap Tested but Recovered'
            elif close > open_price:
                verdict = 'Gap Up Extended - Strong Bulls'
            elif close < open_price and close > prev_close:
                verdict = 'Gap Up Faded but Held'
            else:
                verdict = 'Gap Up Faded'
        else:
            gap_filled = 'Gap Filled' if high >= prev_close else 'Gap Held'
            if high >= prev_close and close > prev_close:
                verdict = 'Complete Gap Fill - Closed above Prev Close'
            elif high >= prev_close and close <= prev_close:
                verdict = 'Gap Tested Prev Close but Faded'
            elif close > open_price:
                verdict = 'Gap Down Bounce - Recovery from Open'
            elif close < open_price:
                verdict = 'Gap Down Extended - Strong Bears'
            else:
                verdict = 'Gap Down Held'
        
        return pd.Series([gap_filled, verdict, round(close_vs_open_pct, 2)])
    
    gap_up[['Gap_Status', 'Verdict', 'Close_vs_Open_%']] = gap_up.apply(analyze_gap_behavior, axis=1)
    gap_down[['Gap_Status', 'Verdict', 'Close_vs_Open_%']] = gap_down.apply(analyze_gap_behavior, axis=1)
    
    result = pd.concat([gap_up, gap_down]).sort_values('Date', ascending=False)
    result['Gap_%'] = result['Gap_%'].round(2)
    result['Year'] = result['Date'].dt.year
    result['Month'] = result['Date'].dt.month_name()
    result['Day_of_Week'] = result['Date'].dt.day_name()
    
    return result

df = load_data()

# Header
st.markdown('<h1 class="dashboard-title">📊 NIFTY GAP ANALYSIS</h1>', unsafe_allow_html=True)
st.markdown('<p class="dashboard-subtitle">5-Year Analysis of Market Opening Gaps (>1%) | TradersCafe Research</p>', unsafe_allow_html=True)

# Sidebar filters
st.sidebar.markdown("### 🎯 Filters")
st.sidebar.markdown("---")

# Year filter
years = sorted(df['Year'].unique())
selected_years = st.sidebar.multiselect("Select Year(s)", years, default=years)

# Gap type filter
gap_types = df['Gap_Type'].unique()
selected_gap_types = st.sidebar.multiselect("Gap Type", gap_types, default=gap_types)

# Filter data
filtered_df = df[(df['Year'].isin(selected_years)) & (df['Gap_Type'].isin(selected_gap_types))]

# Glossary/Definitions
st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 Glossary")

with st.sidebar.expander("📊 Key Metrics Explained", expanded=False):
    st.markdown("""
    **Total Gaps**: Total number of trading days where market opened with >1% gap (up or down) from previous close.
    
    **Gap Ups**: Days when market opened MORE than 1% HIGHER than previous day's close. (Bullish signal)
    
    **Gap Downs**: Days when market opened MORE than 1% LOWER than previous day's close. (Bearish signal)
    
    **Avg Gap Size**: Average percentage difference between previous close and today's open across all gap days.
    
    **Max Gap**: Largest single-day gap recorded in the analyzed period.
    """)

with st.sidebar.expander("🎯 Verdict Terms Explained", expanded=False):
    st.markdown("""
    **Gap Filled**: Price moved back to touch or cross the previous day's close during the day.
    
    **Gap Held**: Gap was NOT filled - price stayed above (gap up) or below (gap down) previous close.
    
    **Gap Extended**: Market continued in the gap direction - very strong trend day.
    
    **Gap Faded**: Market moved opposite to gap direction but didn't fill completely.
    
    **Gap Tested but Recovered**: Gap was tested (price touched previous close) but recovered by end of day.
    """)

with st.sidebar.expander("📈 Analysis Terms Explained", expanded=False):
    st.markdown("""
    **Fill Rate**: Percentage of gaps that get filled during the trading day.
    
    **Extended Rate**: Percentage of gaps where market extends in gap direction.
    
    **Intraday Range**: (High - Low) / Low × 100 - measures day's volatility.
    
    **Close vs Open %**: How market closed relative to its open - positive means closed higher.
    
    **Correlation**: Statistical measure (-1 to +1) of relationship between two variables.
    """)

# Key Metrics Row
st.markdown("### 📈 Key Metrics")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value">{len(filtered_df)}</p>
        <p class="metric-label">Total Gaps</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    gap_up_count = len(filtered_df[filtered_df['Gap_Type'] == 'Gap Up'])
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value" style="background: linear-gradient(135deg, #10b981 0%, #22d3ee 100%); -webkit-background-clip: text;">{gap_up_count}</p>
        <p class="metric-label">Gap Ups</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    gap_down_count = len(filtered_df[filtered_df['Gap_Type'] == 'Gap Down'])
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value" style="background: linear-gradient(135deg, #ef4444 0%, #fb923c 100%); -webkit-background-clip: text;">{gap_down_count}</p>
        <p class="metric-label">Gap Downs</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    avg_gap = filtered_df['Gap_%'].abs().mean()
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value">{avg_gap:.2f}%</p>
        <p class="metric-label">Avg Gap Size</p>
    </div>
    """, unsafe_allow_html=True)

with col5:
    max_gap = filtered_df['Gap_%'].abs().max()
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value">{max_gap:.2f}%</p>
        <p class="metric-label">Max Gap</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🎯 Verdict Analysis", "📅 Time Analysis", "🔍 Deep Insights", "📱 Social Media Ready"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # Gap Up vs Gap Down Distribution
        gap_up_count_chart = len(filtered_df[filtered_df['Gap_Type'] == 'Gap Up'])
        gap_down_count_chart = len(filtered_df[filtered_df['Gap_Type'] == 'Gap Down'])
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=[f'Gap Up ({gap_up_count_chart})', f'Gap Down ({gap_down_count_chart})'],
            values=[gap_up_count_chart, gap_down_count_chart],
            hole=0.6,
            marker=dict(
                colors=['#10b981', '#ef4444'],
                line=dict(color='#0f172a', width=3)
            ),
            textinfo='label+percent',
            textfont=dict(size=16, color='white'),
            textposition='outside',
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
        )])
        
        fig_pie.update_layout(
            title=dict(text="Gap Up vs Gap Down Distribution", font=dict(size=20, color='white'), x=0.5, xanchor='center'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.1,
                xanchor="center",
                x=0.5,
                font=dict(size=12)
            ),
            annotations=[dict(
                text=f'<b>{len(filtered_df)}</b><br>Total',
                x=0.5, y=0.5,
                font_size=36,
                font_color='white',
                showarrow=False
            )],
            height=400
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Gap Size Distribution
        fig_hist = px.histogram(
            filtered_df, 
            x='Gap_%', 
            color='Gap_Type',
            nbins=20,
            color_discrete_map={'Gap Up': '#10b981', 'Gap Down': '#ef4444'},
            title="Gap Size Distribution"
        )
        
        fig_hist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=12),
            title=dict(text="Gap Size Distribution", font=dict(size=20), x=0.5, xanchor='center'),
            xaxis=dict(
                title=dict(text="Gap Percentage", font=dict(size=14)),
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.2)'
            ),
            yaxis=dict(
                title=dict(text="Frequency", font=dict(size=14)),
                gridcolor='rgba(255,255,255,0.1)'
            ),
            bargap=0.1,
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Gap Timeline
    st.markdown("### 📈 Gap Timeline (Scatter Plot)")
    
    fig_scatter = px.scatter(
        filtered_df,
        x='Date',
        y='Gap_%',
        color='Gap_Type',
        size=filtered_df['Gap_%'].abs(),
        color_discrete_map={'Gap Up': '#10b981', 'Gap Down': '#ef4444'},
        hover_data=['Verdict', 'Close_vs_Open_%'],
        title="Gap Occurrences Over Time"
    )
    
    fig_scatter.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title=dict(font=dict(size=20), x=0.5, xanchor='center'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(
            title="Gap %",
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.3)',
            zerolinewidth=2
        ),
        height=450
    )
    fig_scatter.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab2:
    st.markdown("### 🎯 Post-Gap Verdict Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gap Up Verdict
        gap_up_df = filtered_df[filtered_df['Gap_Type'] == 'Gap Up']
        if len(gap_up_df) > 0:
            verdict_up = gap_up_df['Verdict'].value_counts()
            
            # Professional muted colors
            colors_up = ['#059669', '#0284c7', '#7c3aed', '#0891b2', '#64748b']
            
            fig_up = go.Figure(data=[go.Pie(
                labels=verdict_up.index,
                values=verdict_up.values,
                hole=0.5,
                marker=dict(
                    colors=colors_up[:len(verdict_up)], 
                    line=dict(color='#0f172a', width=2)
                ),
                textinfo='label+value+percent',
                textfont=dict(size=13, color='white'),
                textposition='outside',
                outsidetextfont=dict(size=13, color='white'),
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>"
            )])
            
            fig_up.update_layout(
                title=dict(
                    text=f"<b>Gap Up Outcomes</b><br><span style='font-size:16px;color:#94a3b8'>{len(gap_up_df)} Total Instances</span>", 
                    font=dict(size=28, color='white'), 
                    x=0.5,
                    xanchor='center'
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=11),
                showlegend=False,
                height=550,
                margin=dict(t=80, b=80, l=120, r=120),
                annotations=[dict(
                    text=f'<b>{len(gap_up_df)}</b><br>Gap Ups',
                    x=0.5, y=0.5,
                    font_size=22,
                    font_color='#22c55e',
                    showarrow=False
                )]
            )
            st.plotly_chart(fig_up, use_container_width=True)
            
            extended = len(gap_up_df[gap_up_df['Verdict'].str.contains('Extended', na=False)])
            pct = (extended / len(gap_up_df)) * 100
            st.markdown(f"""
            <div class="insight-box">
                <p class="insight-title">📈 Key Insight</p>
                <p class="insight-text"><b>{pct:.0f}%</b> of Gap Ups continued higher (Extended). Momentum continuation is common after gap ups.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Gap Down Verdict
        gap_down_df = filtered_df[filtered_df['Gap_Type'] == 'Gap Down']
        if len(gap_down_df) > 0:
            verdict_down = gap_down_df['Verdict'].value_counts()
            
            # Professional muted colors for Gap Down
            colors_down = ['#dc2626', '#ea580c', '#9333ea', '#0e7490', '#475569']
            
            fig_down = go.Figure(data=[go.Pie(
                labels=verdict_down.index,
                values=verdict_down.values,
                hole=0.5,
                marker=dict(
                    colors=colors_down[:len(verdict_down)], 
                    line=dict(color='#0f172a', width=2)
                ),
                textinfo='label+value+percent',
                textfont=dict(size=13, color='white'),
                textposition='outside',
                outsidetextfont=dict(size=13, color='white'),
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>"
            )])
            
            fig_down.update_layout(
                title=dict(
                    text=f"<b>Gap Down Outcomes</b><br><span style='font-size:16px;color:#94a3b8'>{len(gap_down_df)} Total Instances</span>", 
                    font=dict(size=28, color='white'), 
                    x=0.5,
                    xanchor='center'
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=11),
                showlegend=False,
                height=550,
                margin=dict(t=80, b=80, l=120, r=120),
                annotations=[dict(
                    text=f'<b>{len(gap_down_df)}</b><br>Gap Downs',
                    x=0.5, y=0.5,
                    font_size=22,
                    font_color='#ef4444',
                    showarrow=False
                )]
            )
            st.plotly_chart(fig_down, use_container_width=True)
            
            extended = len(gap_down_df[gap_down_df['Verdict'].str.contains('Extended', na=False)])
            pct = (extended / len(gap_down_df)) * 100
            st.markdown(f"""
            <div class="insight-box">
                <p class="insight-title">📉 Key Insight</p>
                <p class="insight-text"><b>{pct:.0f}%</b> of Gap Downs continued lower (Extended). Bears maintain control on most gap down days.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Verdict Comparison Bar Chart
    st.markdown("### 📊 Verdict Comparison - All Gaps")
    
    verdict_counts = filtered_df.groupby(['Gap_Type', 'Verdict']).size().reset_index(name='Count')
    
    fig_bar = px.bar(
        verdict_counts,
        x='Verdict',
        y='Count',
        color='Gap_Type',
        barmode='group',
        color_discrete_map={'Gap Up': '#10b981', 'Gap Down': '#ef4444'},
        title="Verdict Distribution by Gap Type"
    )
    
    fig_bar.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12),
        title=dict(text="Verdict Distribution by Gap Type", font=dict(size=20), x=0.5, xanchor='center'),
        xaxis=dict(
            tickangle=-90,
            tickfont=dict(size=10),
            gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(title=dict(text="Count", font=dict(size=14)), gridcolor='rgba(255,255,255,0.1)'),
        height=500,
        margin=dict(b=150),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with tab3:
    st.markdown("### 📅 Time-Based Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gaps by Year
        year_counts = filtered_df.groupby(['Year', 'Gap_Type']).size().reset_index(name='Count')
        
        fig_year = px.bar(
            year_counts,
            x='Year',
            y='Count',
            color='Gap_Type',
            barmode='stack',
            color_discrete_map={'Gap Up': '#10b981', 'Gap Down': '#ef4444'},
            title="Gaps by Year"
        )
        
        fig_year.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=12),
            title=dict(text="Gaps by Year", font=dict(size=18), x=0.5, xanchor='center'),
            xaxis=dict(title=dict(text="Year", font=dict(size=12)), gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(title=dict(text="Count", font=dict(size=12)), gridcolor='rgba(255,255,255,0.1)'),
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_year, use_container_width=True)
    
    with col2:
        # Gaps by Day of Week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        day_counts = filtered_df.groupby(['Day_of_Week', 'Gap_Type']).size().reset_index(name='Count')
        day_counts['Day_of_Week'] = pd.Categorical(day_counts['Day_of_Week'], categories=day_order, ordered=True)
        day_counts = day_counts.sort_values('Day_of_Week')
        
        fig_day = px.bar(
            day_counts,
            x='Day_of_Week',
            y='Count',
            color='Gap_Type',
            barmode='group',
            color_discrete_map={'Gap Up': '#10b981', 'Gap Down': '#ef4444'},
            title="Gaps by Day of Week"
        )
        
        fig_day.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=12),
            title=dict(text="Gaps by Day of Week", font=dict(size=18), x=0.5, xanchor='center'),
            xaxis=dict(tickangle=-45, tickfont=dict(size=11), gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(title=dict(text="Count", font=dict(size=12)), gridcolor='rgba(255,255,255,0.1)'),
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_day, use_container_width=True)
    
    # Monthly Heatmap
    st.markdown("### 🗓️ Monthly Gap Heatmap")
    
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    heatmap_data = filtered_df.groupby(['Year', 'Month']).size().reset_index(name='Gaps')
    heatmap_pivot = heatmap_data.pivot(index='Month', columns='Year', values='Gaps').fillna(0)
    
    # Reorder months
    heatmap_pivot = heatmap_pivot.reindex([m for m in month_order if m in heatmap_pivot.index])
    
    fig_heat = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        colorscale=[
            [0, '#1e293b'],
            [0.25, '#3b82f6'],
            [0.5, '#22d3ee'],
            [0.75, '#f59e0b'],
            [1, '#ef4444']
        ],
        text=heatmap_pivot.values,
        texttemplate='%{text:.0f}',
        textfont=dict(size=12, color='white'),
        hovertemplate='Year: %{x}<br>Month: %{y}<br>Gaps: %{z}<extra></extra>'
    ))
    
    fig_heat.update_layout(
        title=dict(text="Gap Occurrences by Month & Year", font=dict(size=20, color='white'), x=0.5, xanchor='center'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12),
        xaxis=dict(title=dict(text="Year", font=dict(size=14)), tickfont=dict(size=12)),
        yaxis=dict(title=dict(text="Month", font=dict(size=14)), tickfont=dict(size=11)),
        height=550,
        margin=dict(l=100)
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with tab4:
    st.markdown("### 🔍 Deep Insights & Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    # Gap Fill Probability
    gap_up_filled = len(filtered_df[(filtered_df['Gap_Type'] == 'Gap Up') & (filtered_df['Gap_Status'] == 'Gap Filled')])
    gap_up_total = len(filtered_df[filtered_df['Gap_Type'] == 'Gap Up'])
    gap_down_filled = len(filtered_df[(filtered_df['Gap_Type'] == 'Gap Down') & (filtered_df['Gap_Status'] == 'Gap Filled')])
    gap_down_total = len(filtered_df[filtered_df['Gap_Type'] == 'Gap Down'])
    
    with col1:
        fill_pct_up = (gap_up_filled / gap_up_total * 100) if gap_up_total > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value" style="font-size: 48px;">{fill_pct_up:.0f}%</p>
            <p class="metric-label">Gap Up Fill Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        fill_pct_down = (gap_down_filled / gap_down_total * 100) if gap_down_total > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value" style="font-size: 48px;">{fill_pct_down:.0f}%</p>
            <p class="metric-label">Gap Down Fill Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_intraday_range = ((filtered_df['High'] - filtered_df['Low']) / filtered_df['Low'] * 100).mean()
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value" style="font-size: 48px;">{avg_intraday_range:.2f}%</p>
            <p class="metric-label">Avg Intraday Range</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gap Size vs Close Performance
        st.markdown("#### 📈 Gap Size vs Close Performance")
        
        fig_scatter2 = px.scatter(
            filtered_df,
            x='Gap_%',
            y='Close_vs_Open_%',
            color='Gap_Type',
            color_discrete_map={'Gap Up': '#059669', 'Gap Down': '#dc2626'},
            trendline='ols',
            title="Does Gap Size Predict Closing Direction?"
        )
        
        fig_scatter2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title=dict(font=dict(size=18), x=0.5, xanchor='center'),
            xaxis=dict(title="Gap %", gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(title="Close vs Open %", gridcolor='rgba(255,255,255,0.1)'),
            height=400
        )
        st.plotly_chart(fig_scatter2, use_container_width=True)
        
        # Correlation insight
        corr = filtered_df['Gap_%'].corr(filtered_df['Close_vs_Open_%'])
        st.markdown(f"""
        <div class="insight-box">
            <p class="insight-title">📊 Correlation Analysis</p>
            <p class="insight-text">Correlation between Gap Size and Close Performance: <b>{corr:.3f}</b><br>
            {'Weak correlation - Gap size has limited predictive power for closing direction.' if abs(corr) < 0.3 else 'Moderate correlation exists between gap size and close direction.'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Extended vs Filled by Gap Size
        st.markdown("#### 🎯 Outcome by Gap Size Range")
        
        filtered_df['Gap_Size_Range'] = pd.cut(
            filtered_df['Gap_%'].abs(), 
            bins=[0, 1.5, 2.0, 3.0, 10],
            labels=['1.0-1.5%', '1.5-2.0%', '2.0-3.0%', '3.0%+']
        )
        
        size_outcome = filtered_df.groupby(['Gap_Size_Range', 'Gap_Status']).size().reset_index(name='Count')
        
        fig_size = px.bar(
            size_outcome,
            x='Gap_Size_Range',
            y='Count',
            color='Gap_Status',
            barmode='group',
            color_discrete_map={'Gap Filled': '#3b82f6', 'Gap Held': '#22c55e'},
            title="Gap Fill Rate by Size"
        )
        
        fig_size.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title=dict(font=dict(size=18), x=0.5, xanchor='center'),
            xaxis=dict(title="Gap Size Range", gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(title="Count", gridcolor='rgba(255,255,255,0.1)'),
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_size, use_container_width=True)
    
    # Key Statistics Table
    st.markdown("### 📋 Key Statistics Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Gap Up Statistics")
        gap_up_stats = filtered_df[filtered_df['Gap_Type'] == 'Gap Up']
        if len(gap_up_stats) > 0:
            stats_up = {
                'Metric': ['Total Count', 'Average Gap %', 'Max Gap %', 'Fill Rate', 'Extended Rate', 'Avg Close vs Open'],
                'Value': [
                    len(gap_up_stats),
                    f"{gap_up_stats['Gap_%'].mean():.2f}%",
                    f"{gap_up_stats['Gap_%'].max():.2f}%",
                    f"{(gap_up_filled/gap_up_total*100):.0f}%" if gap_up_total > 0 else "N/A",
                    f"{len(gap_up_stats[gap_up_stats['Verdict'].str.contains('Extended', na=False)])/len(gap_up_stats)*100:.0f}%",
                    f"{gap_up_stats['Close_vs_Open_%'].mean():.2f}%"
                ]
            }
            st.dataframe(pd.DataFrame(stats_up), use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### Gap Down Statistics")
        gap_down_stats = filtered_df[filtered_df['Gap_Type'] == 'Gap Down']
        if len(gap_down_stats) > 0:
            stats_down = {
                'Metric': ['Total Count', 'Average Gap %', 'Max Gap %', 'Fill Rate', 'Extended Rate', 'Avg Close vs Open'],
                'Value': [
                    len(gap_down_stats),
                    f"{gap_down_stats['Gap_%'].mean():.2f}%",
                    f"{gap_down_stats['Gap_%'].min():.2f}%",
                    f"{(gap_down_filled/gap_down_total*100):.0f}%" if gap_down_total > 0 else "N/A",
                    f"{len(gap_down_stats[gap_down_stats['Verdict'].str.contains('Extended', na=False)])/len(gap_down_stats)*100:.0f}%",
                    f"{gap_down_stats['Close_vs_Open_%'].mean():.2f}%"
                ]
            }
            st.dataframe(pd.DataFrame(stats_down), use_container_width=True, hide_index=True)

    # Definitions for the statistics
    st.markdown("---")
    st.markdown("### 📖 What These Statistics Mean")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
            <p class="insight-title">📊 Basic Metrics</p>
            <p class="insight-text">
            <b>Total Count:</b> Number of gap days in selected period<br><br>
            <b>Average Gap %:</b> Mean gap size - how much market typically gaps<br><br>
            <b>Max Gap %:</b> Largest gap recorded - shows extreme volatility
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-box">
            <p class="insight-title">🎯 Performance Metrics</p>
            <p class="insight-text">
            <b>Fill Rate:</b> % of gaps where price returns to previous close<br><br>
            <b>Extended Rate:</b> % of gaps where market continues in gap direction (strong trend)<br><br>
            <b>Avg Close vs Open:</b> How market typically closes relative to open (+ = bullish, - = bearish)
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Top 10 Gap Ups and Gap Downs
    st.markdown("---")
    st.markdown("### 🏆 Top 10 Largest Gaps")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Top 10 Gap Ups")
        gap_up_data = filtered_df[filtered_df['Gap_Type'] == 'Gap Up'].copy()
        if len(gap_up_data) > 0:
            gap_up_data = gap_up_data.nlargest(10, 'Gap_%')
            gap_up_data['Gap_Points'] = gap_up_data['Open'] - gap_up_data['Prev_Close']
            
            top_up = pd.DataFrame({
                'Date': gap_up_data['Date'].dt.strftime('%d-%b-%Y'),
                'Gap %': gap_up_data['Gap_%'].apply(lambda x: f"+{x:.2f}%"),
                'Gap Points': gap_up_data['Gap_Points'].apply(lambda x: f"+{x:.0f}"),
                'Open': gap_up_data['Open'].apply(lambda x: f"{x:,.0f}"),
                'Close': gap_up_data['Close'].apply(lambda x: f"{x:,.0f}"),
                'Verdict': gap_up_data['Verdict'].apply(lambda x: x.split(' - ')[0] if ' - ' in x else x[:20])
            })
            st.dataframe(top_up, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### 📉 Top 10 Gap Downs")
        gap_down_data = filtered_df[filtered_df['Gap_Type'] == 'Gap Down'].copy()
        if len(gap_down_data) > 0:
            gap_down_data = gap_down_data.nsmallest(10, 'Gap_%')
            gap_down_data['Gap_Points'] = gap_down_data['Open'] - gap_down_data['Prev_Close']
            
            top_down = pd.DataFrame({
                'Date': gap_down_data['Date'].dt.strftime('%d-%b-%Y'),
                'Gap %': gap_down_data['Gap_%'].apply(lambda x: f"{x:.2f}%"),
                'Gap Points': gap_down_data['Gap_Points'].apply(lambda x: f"{x:.0f}"),
                'Open': gap_down_data['Open'].apply(lambda x: f"{x:,.0f}"),
                'Close': gap_down_data['Close'].apply(lambda x: f"{x:,.0f}"),
                'Verdict': gap_down_data['Verdict'].apply(lambda x: x.split(' - ')[0] if ' - ' in x else x[:20])
            })
            st.dataframe(top_down, use_container_width=True, hide_index=True)

with tab5:
    st.markdown("### 📱 Social Media Ready Charts")
    st.markdown("*Click on any chart and use the camera icon to download high-quality images for social media*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Social Media Card 1: Key Stats
        gap_up_extended = len(filtered_df[(filtered_df['Gap_Type'] == 'Gap Up') & (filtered_df['Verdict'].str.contains('Extended', na=False))])
        gap_down_extended = len(filtered_df[(filtered_df['Gap_Type'] == 'Gap Down') & (filtered_df['Verdict'].str.contains('Extended', na=False))])
        total_up = len(filtered_df[filtered_df['Gap_Type'] == 'Gap Up'])
        total_down = len(filtered_df[filtered_df['Gap_Type'] == 'Gap Down'])
        
        fig_social1 = go.Figure()
        
        fig_social1.add_trace(go.Bar(
            x=['Gap Up', 'Gap Down'],
            y=[gap_up_extended/total_up*100 if total_up > 0 else 0, 
               gap_down_extended/total_down*100 if total_down > 0 else 0],
            marker_color=['#10b981', '#ef4444'],
            text=[f'{gap_up_extended/total_up*100:.0f}%' if total_up > 0 else '0%',
                  f'{gap_down_extended/total_down*100:.0f}%' if total_down > 0 else '0%'],
            textposition='outside',
            textfont=dict(size=24, color='white')
        ))
        
        fig_social1.update_layout(
            title=dict(
                text="<b>NIFTY GAPS:</b><br><span style='font-size:18px'>How Often Do They Continue?</span>",
                font=dict(size=24, color='white'),
                x=0.5, y=0.95
            ),
            paper_bgcolor='#1a1a3e',
            plot_bgcolor='#1a1a3e',
            font=dict(color='white'),
            xaxis=dict(
                tickfont=dict(size=18),
                gridcolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                title="% Continued in Gap Direction",
                range=[0, 70],
                gridcolor='rgba(255,255,255,0.1)'
            ),
            height=500,
            annotations=[dict(
                text="@TradersCafe | 5 Year Analysis",
                xref="paper", yref="paper",
                x=0.5, y=-0.1,
                showarrow=False,
                font=dict(size=14, color='#a0a0c0')
            )]
        )
        st.plotly_chart(fig_social1, use_container_width=True)
    
    with col2:
        # Social Media Card 2: Gap Distribution
        fig_social2 = go.Figure()
        
        fig_social2.add_trace(go.Indicator(
            mode="number+delta",
            value=len(filtered_df),
            title=dict(text="<b>Total >1% Gaps</b><br><span style='font-size:14px'>in Last 5 Years</span>", font=dict(size=20)),
            number=dict(font=dict(size=60), valueformat=".0f"),
            domain={'x': [0.1, 0.9], 'y': [0.55, 1]}
        ))
        
        fig_social2.add_trace(go.Pie(
            values=[total_up, total_down],
            labels=['Gap Up', 'Gap Down'],
            hole=0.6,
            marker=dict(colors=['#10b981', '#ef4444']),
            textinfo='label+value',
            textfont=dict(size=14),
            domain={'x': [0.15, 0.85], 'y': [0, 0.45]}
        ))
        
        fig_social2.update_layout(
            paper_bgcolor='#1a1a3e',
            plot_bgcolor='#1a1a3e',
            font=dict(color='white'),
            height=500,
            showlegend=False,
            annotations=[dict(
                text="@TradersCafe",
                xref="paper", yref="paper",
                x=0.5, y=-0.05,
                showarrow=False,
                font=dict(size=14, color='#a0a0c0')
            )]
        )
        st.plotly_chart(fig_social2, use_container_width=True)
    
    # Top Gaps Table
    st.markdown("### 🐦 Top 10 Gap Ups - Twitter Ready")
    st.markdown("*Click the camera icon on the chart to download for Twitter*")
    
    # Get top 10 gap ups for Twitter card
    gap_up_twitter = filtered_df[filtered_df['Gap_Type'] == 'Gap Up'].copy()
    if len(gap_up_twitter) > 0:
        gap_up_twitter = gap_up_twitter.nlargest(10, 'Gap_%')
        gap_up_twitter['Gap_Points'] = (gap_up_twitter['Open'] - gap_up_twitter['Prev_Close']).round(0)
        gap_up_twitter['Date_Str'] = gap_up_twitter['Date'].dt.strftime('%d-%b-%Y')
        
        # Create a table as a Plotly figure for easy download
        fig_twitter = go.Figure(data=[go.Table(
            header=dict(
                values=['<b>Rank</b>', '<b>Date</b>', '<b>Gap %</b>', '<b>Points</b>', '<b>Open</b>'],
                fill_color='#059669',
                align='center',
                font=dict(color='white', size=14),
                height=40
            ),
            cells=dict(
                values=[
                    [f"#{i+1}" for i in range(len(gap_up_twitter))],
                    gap_up_twitter['Date_Str'].tolist(),
                    [f"+{x:.2f}%" for x in gap_up_twitter['Gap_%']],
                    [f"+{x:.0f}" for x in gap_up_twitter['Gap_Points']],
                    [f"{x:,.0f}" for x in gap_up_twitter['Open']]
                ],
                fill_color=[['#1e293b', '#0f172a'] * 5],
                align='center',
                font=dict(color='white', size=13),
                height=35
            )
        )])
        
        fig_twitter.update_layout(
            title=dict(
                text="<b>🚀 TOP 10 NIFTY GAP UPS (5 Years)</b><br><span style='font-size:14px;color:#94a3b8'>Gaps >1% | @TradersCafe</span>",
                font=dict(size=22, color='#22c55e'),
                x=0.5,
                xanchor='center'
            ),
            paper_bgcolor='#0f172a',
            margin=dict(t=80, b=40, l=20, r=20),
            height=500,
            width=700
        )
        
        # Add footer annotation
        fig_twitter.add_annotation(
            text="📊 Data: NSE Historical | 5-Year Analysis | @TradersCafe",
            xref="paper", yref="paper",
            x=0.5, y=-0.05,
            showarrow=False,
            font=dict(size=12, color='#64748b'),
            xanchor='center'
        )
        
        st.plotly_chart(fig_twitter, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #a0a0c0; padding: 20px;">
    <p>📊 <b>NIFTY Gap Analysis Dashboard</b> | Built by TradersCafe</p>
    <p style="font-size: 12px;">Data Source: NSE Historical Data | Analysis Period: 5 Years</p>
</div>
""", unsafe_allow_html=True)
