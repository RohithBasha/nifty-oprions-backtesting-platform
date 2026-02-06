"""
NIFTY Options Backtesting Dashboard
====================================
Professional MT QUANT-inspired backtesting platform for NIFTY options strategies.
Uses 4 years of minute-level options and spot data from DuckDB.

Author: TradersCafe
Version: 1.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import duckdb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, time
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="NIFTY Options Backtester",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - Premium Dark Theme
# ============================================================================

st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a3e 0%, #0f0f23 100%);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stDateInput label,
    [data-testid="stSidebar"] .stSlider label {
        color: #e0e0e0 !important;
        font-weight: 500;
    }
    
    /* Headers */
    h1 {
        background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 50%, #ff006e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        text-align: center;
    }
    
    h2, h3 {
        color: #00d4ff !important;
        font-weight: 600;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, #1e1e4a 0%, #2a2a5a 100%);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 212, 255, 0.2);
    }
    
    .metric-value {
        font-size: 36px;
        font-weight: 700;
        margin: 0;
    }
    
    .metric-value.profit {
        background: linear-gradient(135deg, #10b981 0%, #22d3ee 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-value.loss {
        background: linear-gradient(135deg, #ef4444 0%, #fb923c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-value.neutral {
        background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 13px;
        color: #a0a0c0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 8px;
    }
    
    /* Strategy card */
    .strategy-card {
        background: linear-gradient(145deg, rgba(0, 212, 255, 0.1) 0%, rgba(124, 58, 237, 0.1) 100%);
        border-left: 4px solid #7c3aed;
        border-radius: 8px;
        padding: 16px;
        margin: 16px 0;
    }
    
    .strategy-title {
        color: #00d4ff;
        font-weight: 600;
        font-size: 18px;
        margin-bottom: 8px;
    }
    
    .strategy-desc {
        color: #a0a0c0;
        font-size: 14px;
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 8px;
        padding: 12px;
        color: #e0e0e0;
        font-size: 14px;
    }
    
    .warning-box {
        background: rgba(251, 146, 60, 0.1);
        border: 1px solid rgba(251, 146, 60, 0.3);
        border-radius: 8px;
        padding: 12px;
        color: #fb923c;
        font-size: 14px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 30, 74, 0.5);
        border-radius: 8px;
        padding: 10px 24px;
        color: #a0a0c0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 100%);
        color: white !important;
    }
    
    /* Data table */
    .stDataFrame {
        background: rgba(30, 30, 74, 0.5);
        border-radius: 8px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.4);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 100%);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATABASE CONNECTION
# ============================================================================

@st.cache_resource
def get_db_connection():
    """Get DuckDB connection"""
    return duckdb.connect('nifty_data.duckdb', read_only=True)

@st.cache_data(ttl=3600)
def get_data_range(_conn):
    """Get available data range"""
    result = _conn.execute("""
        SELECT 
            MIN(date) as min_date,
            MAX(date) as max_date,
            COUNT(DISTINCT date) as trading_days
        FROM spot_data
    """).fetchone()
    return result

@st.cache_data(ttl=3600)
def get_trading_days(_conn, start_date, end_date):
    """Get list of trading days in range"""
    query = f"""
        SELECT DISTINCT date 
        FROM spot_data 
        WHERE date >= '{start_date}' AND date <= '{end_date}'
        ORDER BY date
    """
    return _conn.execute(query).fetchdf()['date'].tolist()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_atm_strike(spot_price: float, interval: int = 50) -> int:
    """Get ATM strike price"""
    return int(round(spot_price / interval) * interval)

def get_spot_price(_conn, date, time_str) -> Optional[float]:
    """Get spot price at specific date and time"""
    query = f"""
        SELECT close FROM spot_data 
        WHERE date = '{date}' 
        AND CAST(datetime AS TIME) >= '{time_str}'
        ORDER BY datetime
        LIMIT 1
    """
    result = _conn.execute(query).fetchone()
    return result[0] if result else None

def get_option_price(_conn, date, time_str, strike, option_type, expiry=None):
    """Get option price at specific date, time, strike"""
    query = f"""
        SELECT close, expiry_date FROM options_data 
        WHERE date = '{date}'
        AND CAST(datetime AS TIME) >= '{time_str}'
        AND strike_price = {strike}
        AND option_type = '{option_type}'
    """
    if expiry:
        query += f" AND expiry_date = '{expiry}'"
    else:
        query += f" AND expiry_date >= '{date}'"
    
    query += " ORDER BY expiry_date, datetime LIMIT 1"
    result = _conn.execute(query).fetchone()
    return (result[0], result[1]) if result else (None, None)

def get_nearest_expiry(_conn, date):
    """Get nearest expiry for a date"""
    query = f"""
        SELECT DISTINCT expiry_date FROM options_data 
        WHERE date = '{date}' AND expiry_date >= '{date}'
        ORDER BY expiry_date
        LIMIT 1
    """
    result = _conn.execute(query).fetchone()
    return result[0] if result else None

def get_option_price_intraday(_conn, date, start_time, end_time, strike, option_type, expiry):
    """Get option prices throughout the day for SL/Target monitoring"""
    query = f"""
        SELECT datetime, close FROM options_data 
        WHERE date = '{date}'
        AND CAST(datetime AS TIME) >= '{start_time}'
        AND CAST(datetime AS TIME) <= '{end_time}'
        AND strike_price = {strike}
        AND option_type = '{option_type}'
        AND expiry_date = '{expiry}'
        ORDER BY datetime
    """
    return _conn.execute(query).fetchdf()

# ============================================================================
# STRATEGY DEFINITIONS
# ============================================================================

STRATEGIES = {
    "Short Straddle": {
        "description": "Sell ATM Call + ATM Put. Profit from low volatility.",
        "legs": [
            {"type": "CALL", "action": "SELL", "strike_offset": 0},
            {"type": "PUT", "action": "SELL", "strike_offset": 0}
        ],
        "category": "Neutral"
    },
    "Long Straddle": {
        "description": "Buy ATM Call + ATM Put. Profit from high volatility.",
        "legs": [
            {"type": "CALL", "action": "BUY", "strike_offset": 0},
            {"type": "PUT", "action": "BUY", "strike_offset": 0}
        ],
        "category": "Neutral"
    },
    "Short Strangle": {
        "description": "Sell OTM Call + OTM Put. Wider profit zone.",
        "legs": [
            {"type": "CALL", "action": "SELL", "strike_offset": 100},
            {"type": "PUT", "action": "SELL", "strike_offset": -100}
        ],
        "category": "Neutral"
    },
    "Long Strangle": {
        "description": "Buy OTM Call + OTM Put. Lower cost than straddle.",
        "legs": [
            {"type": "CALL", "action": "BUY", "strike_offset": 100},
            {"type": "PUT", "action": "BUY", "strike_offset": -100}
        ],
        "category": "Neutral"
    },
    "Iron Condor": {
        "description": "Sell OTM strangle + Buy further OTM options.",
        "legs": [
            {"type": "CALL", "action": "SELL", "strike_offset": 100},
            {"type": "CALL", "action": "BUY", "strike_offset": 200},
            {"type": "PUT", "action": "SELL", "strike_offset": -100},
            {"type": "PUT", "action": "BUY", "strike_offset": -200}
        ],
        "category": "Neutral"
    },
    "Bull Call Spread": {
        "description": "Buy ATM Call + Sell OTM Call. Bullish view.",
        "legs": [
            {"type": "CALL", "action": "BUY", "strike_offset": 0},
            {"type": "CALL", "action": "SELL", "strike_offset": 100}
        ],
        "category": "Bullish"
    },
    "Bear Put Spread": {
        "description": "Buy ATM Put + Sell OTM Put. Bearish view.",
        "legs": [
            {"type": "PUT", "action": "BUY", "strike_offset": 0},
            {"type": "PUT", "action": "SELL", "strike_offset": -100}
        ],
        "category": "Bearish"
    },
    "Naked Call": {
        "description": "Sell single ATM/OTM Call. High risk.",
        "legs": [
            {"type": "CALL", "action": "SELL", "strike_offset": 0}
        ],
        "category": "Bearish"
    },
    "Naked Put": {
        "description": "Sell single ATM/OTM Put. Bullish bias.",
        "legs": [
            {"type": "PUT", "action": "SELL", "strike_offset": 0}
        ],
        "category": "Bullish"
    }
}

# ============================================================================
# BACKTESTING ENGINE
# ============================================================================

def calculate_metrics(trades_df: pd.DataFrame, initial_capital: float = 100000) -> Dict:
    """Calculate comprehensive performance metrics"""
    if len(trades_df) == 0:
        return {}
    
    total_trades = len(trades_df)
    winners = trades_df[trades_df['pnl'] > 0]
    losers = trades_df[trades_df['pnl'] < 0]
    
    total_pnl = trades_df['pnl'].sum()
    win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0
    
    avg_winner = winners['pnl'].mean() if len(winners) > 0 else 0
    avg_loser = abs(losers['pnl'].mean()) if len(losers) > 0 else 0
    
    profit_factor = abs(winners['pnl'].sum() / losers['pnl'].sum()) if len(losers) > 0 and losers['pnl'].sum() != 0 else float('inf')
    
    # Drawdown calculation
    cumulative = trades_df['pnl'].cumsum()
    peak = cumulative.cummax()
    drawdown = cumulative - peak
    max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
    max_dd_pct = (max_drawdown / initial_capital) * 100 if initial_capital > 0 else 0
    
    # Sharpe Ratio (assuming daily returns)
    daily_returns = trades_df['pnl'] / initial_capital
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
    
    # Sortino Ratio (only downside deviation)
    downside_returns = daily_returns[daily_returns < 0]
    sortino = (daily_returns.mean() / downside_returns.std()) * np.sqrt(252) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
    
    # Calmar Ratio
    annual_return = (total_pnl / initial_capital) * (252 / total_trades) if total_trades > 0 else 0
    calmar = annual_return / (max_dd_pct / 100) if max_dd_pct > 0 else 0
    
    # Win/Loss streaks
    streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    current_streak = 0
    last_result = None
    
    for pnl in trades_df['pnl']:
        current_result = 'win' if pnl > 0 else 'loss'
        if current_result == last_result:
            current_streak += 1
        else:
            current_streak = 1
        
        if current_result == 'win':
            max_win_streak = max(max_win_streak, current_streak)
        else:
            max_loss_streak = max(max_loss_streak, current_streak)
        
        last_result = current_result
    
    # Monthly profitability
    if 'date' in trades_df.columns:
        trades_df['month'] = pd.to_datetime(trades_df['date']).dt.to_period('M')
        monthly_pnl = trades_df.groupby('month')['pnl'].sum()
        profitable_months = (monthly_pnl > 0).sum()
        total_months = len(monthly_pnl)
        monthly_win_rate = (profitable_months / total_months * 100) if total_months > 0 else 0
    else:
        monthly_win_rate = 0
    
    # Expectancy
    expectancy = (win_rate/100 * avg_winner) - ((100-win_rate)/100 * avg_loser)
    
    return {
        'total_pnl': total_pnl,
        'total_trades': total_trades,
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': win_rate,
        'avg_winner': avg_winner,
        'avg_loser': avg_loser,
        'avg_trade': total_pnl / total_trades if total_trades > 0 else 0,
        'best_trade': trades_df['pnl'].max(),
        'worst_trade': trades_df['pnl'].min(),
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'max_dd_pct': max_dd_pct,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_win_streak': max_win_streak,
        'max_loss_streak': max_loss_streak,
        'monthly_win_rate': monthly_win_rate,
        'expectancy': expectancy,
        'return_pct': (total_pnl / initial_capital) * 100
    }

def run_backtest(conn, strategy_name: str, config: Dict, progress_callback=None) -> pd.DataFrame:
    """Run backtest for a strategy"""
    
    strategy = STRATEGIES[strategy_name]
    legs = strategy['legs']
    
    trading_days = get_trading_days(conn, config['start_date'], config['end_date'])
    
    results = []
    total_days = len(trading_days)
    
    for i, trade_date in enumerate(trading_days):
        if progress_callback:
            progress_callback((i + 1) / total_days)
        
        try:
            # Get spot price at entry
            spot_at_entry = get_spot_price(conn, trade_date, config['entry_time'])
            if not spot_at_entry:
                continue
            
            # Calculate ATM strike
            atm_strike = get_atm_strike(spot_at_entry)
            
            # Get nearest expiry
            expiry = get_nearest_expiry(conn, trade_date)
            if not expiry:
                continue
            
            # Execute each leg
            leg_data = []
            skip_trade = False
            total_premium = 0
            
            for leg in legs:
                strike = atm_strike + leg['strike_offset']
                option_type = leg['type']
                action = leg['action']
                
                entry_price, _ = get_option_price(conn, trade_date, config['entry_time'], 
                                                   strike, option_type, expiry)
                exit_price, _ = get_option_price(conn, trade_date, config['exit_time'], 
                                                  strike, option_type, expiry)
                
                if entry_price is None or exit_price is None:
                    skip_trade = True
                    break
                
                # Apply slippage
                slippage = config.get('slippage_pct', 0.05) / 100
                if action == 'BUY':
                    entry_price = entry_price * (1 + slippage)
                    exit_price = exit_price * (1 - slippage)
                else:
                    entry_price = entry_price * (1 - slippage)
                    exit_price = exit_price * (1 + slippage)
                
                leg_data.append({
                    'strike': strike,
                    'type': option_type,
                    'action': action,
                    'entry': entry_price,
                    'exit': exit_price
                })
                
                # Calculate P&L for this leg
                multiplier = 1 if action == 'SELL' else -1
                total_premium += (entry_price - exit_price) * multiplier
            
            if skip_trade:
                continue
            
            # Calculate total P&L
            lot_size = config['lot_size'] * config['num_lots']
            gross_pnl = total_premium * lot_size
            
            # Subtract brokerage
            brokerage = config.get('brokerage_per_trade', 20) * len(legs) * 2  # Entry + Exit
            net_pnl = gross_pnl - brokerage
            
            result = {
                'date': trade_date,
                'expiry': expiry,
                'spot_entry': spot_at_entry,
                'atm_strike': atm_strike,
                'strategy': strategy_name,
                'legs': len(legs),
                'gross_pnl': gross_pnl,
                'brokerage': brokerage,
                'pnl': net_pnl,
                'leg_details': leg_data
            }
            results.append(result)
            
        except Exception as e:
            continue
    
    return pd.DataFrame(results)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown("# 🚀 NIFTY OPTIONS BACKTESTER")
    st.markdown("""
    <p style='text-align: center; color: #a0a0c0; font-size: 18px; margin-top: -10px;'>
        Professional Strategy Backtesting Platform | TradersCafe
    </p>
    """, unsafe_allow_html=True)
    
    # Connect to database
    try:
        conn = get_db_connection()
        data_range = get_data_range(conn)
        min_date, max_date, trading_days = data_range
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        st.info("Please ensure `nifty_data.duckdb` is in the TradersCafe folder.")
        return
    
    # ========================================================================
    # SIDEBAR - Strategy Configuration
    # ========================================================================
    
    with st.sidebar:
        st.markdown("## ⚙️ Strategy Configuration")
        st.markdown("---")
        
        # Strategy Selection
        st.markdown("### 📊 Select Strategy")
        strategy_name = st.selectbox(
            "Strategy Type",
            options=list(STRATEGIES.keys()),
            index=0,
            help="Choose a pre-built options strategy"
        )
        
        # Show strategy info
        strategy = STRATEGIES[strategy_name]
        st.markdown(f"""
        <div class="strategy-card">
            <div class="strategy-title">{strategy_name}</div>
            <div class="strategy-desc">{strategy['description']}</div>
            <div style="margin-top: 8px; color: #7c3aed; font-size: 12px;">
                Category: {strategy['category']} | Legs: {len(strategy['legs'])}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Date Range
        st.markdown("### 📅 Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime(2025, 1, 1),
                min_value=min_date,
                max_value=max_date
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=max_date,
                min_value=min_date,
                max_value=max_date
            )
        
        st.markdown("---")
        
        # Entry/Exit Times
        st.markdown("### ⏰ Entry/Exit Times")
        col1, col2 = st.columns(2)
        with col1:
            entry_time = st.time_input("Entry Time", value=time(9, 20))
        with col2:
            exit_time = st.time_input("Exit Time", value=time(15, 15))
        
        st.markdown("---")
        
        # Position Sizing
        st.markdown("### 📦 Position Sizing")
        lot_size = st.number_input("Lot Size", value=25, min_value=1)
        num_lots = st.number_input("Number of Lots", value=1, min_value=1, max_value=100)
        
        st.markdown("---")
        
        # Risk Settings
        st.markdown("### ⚠️ Costs & Slippage")
        brokerage = st.number_input("Brokerage per Trade (₹)", value=20, min_value=0)
        slippage = st.slider("Slippage (%)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
        
        st.markdown("---")
        
        # Initial Capital
        initial_capital = st.number_input("Initial Capital (₹)", value=100000, min_value=10000, step=10000)
        
        st.markdown("---")
        
        # Run Backtest Button
        run_backtest_btn = st.button("🚀 RUN BACKTEST", use_container_width=True)
    
    # ========================================================================
    # MAIN CONTENT AREA
    # ========================================================================
    
    # Data info bar
    st.markdown(f"""
    <div class="info-box">
        📊 <strong>Data Available:</strong> {min_date.strftime('%b %d, %Y')} to {max_date.strftime('%b %d, %Y')} 
        | <strong>{trading_days}</strong> Trading Days
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # Run backtest if button clicked
    if run_backtest_btn:
        config = {
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'entry_time': entry_time.strftime('%H:%M:%S'),
            'exit_time': exit_time.strftime('%H:%M:%S'),
            'lot_size': lot_size,
            'num_lots': num_lots,
            'brokerage_per_trade': brokerage,
            'slippage_pct': slippage
        }
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(progress):
            progress_bar.progress(progress)
            status_text.text(f"Processing... {int(progress * 100)}%")
        
        with st.spinner("Running backtest..."):
            trades_df = run_backtest(conn, strategy_name, config, update_progress)
        
        progress_bar.empty()
        status_text.empty()
        
        if len(trades_df) == 0:
            st.warning("⚠️ No trades executed. Check your date range and strategy settings.")
            return
        
        # Store results in session state
        st.session_state['trades_df'] = trades_df
        st.session_state['metrics'] = calculate_metrics(trades_df, initial_capital)
        st.session_state['config'] = config
        st.session_state['strategy_name'] = strategy_name
    
    # Display results if available
    if 'trades_df' in st.session_state and len(st.session_state['trades_df']) > 0:
        trades_df = st.session_state['trades_df']
        metrics = st.session_state['metrics']
        
        # Calculate cumulative P&L
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        
        # ====================================================================
        # KEY METRICS ROW
        # ====================================================================
        
        st.markdown("### 📈 Performance Summary")
        
        cols = st.columns(6)
        
        with cols[0]:
            pnl_class = "profit" if metrics['total_pnl'] >= 0 else "loss"
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value {pnl_class}">₹{metrics['total_pnl']:,.0f}</p>
                <p class="metric-label">Total P&L</p>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[1]:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value neutral">{metrics['win_rate']:.1f}%</p>
                <p class="metric-label">Win Rate</p>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[2]:
            sharpe_class = "profit" if metrics['sharpe_ratio'] >= 0 else "loss"
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value {sharpe_class}">{metrics['sharpe_ratio']:.2f}</p>
                <p class="metric-label">Sharpe Ratio</p>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[3]:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value loss">{metrics['max_dd_pct']:.1f}%</p>
                <p class="metric-label">Max Drawdown</p>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[4]:
            pf_class = "profit" if metrics['profit_factor'] >= 1 else "loss"
            pf_display = f"{metrics['profit_factor']:.2f}" if metrics['profit_factor'] < 100 else "∞"
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value {pf_class}">{pf_display}</p>
                <p class="metric-label">Profit Factor</p>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[5]:
            avg_class = "profit" if metrics['avg_trade'] >= 0 else "loss"
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value {avg_class}">₹{metrics['avg_trade']:,.0f}</p>
                <p class="metric-label">Avg Trade</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # ====================================================================
        # TABS FOR DIFFERENT VIEWS
        # ====================================================================
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Equity Curve", 
            "📊 Analytics", 
            "📅 Calendar View", 
            "📋 Trade Log",
            "📉 Risk Metrics"
        ])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Equity Curve with Drawdown
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.7, 0.3],
                    subplot_titles=("Cumulative P&L", "Drawdown")
                )
                
                # Equity curve
                fig.add_trace(
                    go.Scatter(
                        x=trades_df['date'],
                        y=trades_df['cumulative_pnl'],
                        mode='lines',
                        name='Cumulative P&L',
                        line=dict(color='#00d4ff', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(0, 212, 255, 0.1)'
                    ),
                    row=1, col=1
                )
                
                # Calculate drawdown
                peak = trades_df['cumulative_pnl'].cummax()
                drawdown = trades_df['cumulative_pnl'] - peak
                
                # Drawdown
                fig.add_trace(
                    go.Scatter(
                        x=trades_df['date'],
                        y=drawdown,
                        mode='lines',
                        name='Drawdown',
                        line=dict(color='#ef4444', width=1),
                        fill='tozeroy',
                        fillcolor='rgba(239, 68, 68, 0.3)'
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    showlegend=False,
                    height=500,
                    margin=dict(l=50, r=20, t=40, b=40)
                )
                
                fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
                fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # P&L Distribution
                fig_hist = px.histogram(
                    trades_df,
                    x='pnl',
                    nbins=30,
                    title="P&L Distribution"
                )
                
                fig_hist.update_traces(marker_color='#7c3aed')
                
                fig_hist.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis_title="P&L (₹)",
                    yaxis_title="Frequency",
                    height=500,
                    showlegend=False
                )
                
                fig_hist.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
                fig_hist.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
                
                # Add mean line
                mean_pnl = trades_df['pnl'].mean()
                fig_hist.add_vline(x=mean_pnl, line_dash="dash", line_color="#00d4ff",
                                   annotation_text=f"Mean: ₹{mean_pnl:,.0f}")
                
                st.plotly_chart(fig_hist, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Day of Week Analysis
                trades_df['day_of_week'] = pd.to_datetime(trades_df['date']).dt.day_name()
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
                day_stats = trades_df.groupby('day_of_week').agg({
                    'pnl': ['sum', 'mean', 'count']
                }).reset_index()
                day_stats.columns = ['day', 'total_pnl', 'avg_pnl', 'trades']
                day_stats['day'] = pd.Categorical(day_stats['day'], categories=day_order, ordered=True)
                day_stats = day_stats.sort_values('day')
                
                fig_day = px.bar(
                    day_stats,
                    x='day',
                    y='total_pnl',
                    title="P&L by Day of Week",
                    color='total_pnl',
                    color_continuous_scale=['#ef4444', '#22c55e']
                )
                
                fig_day.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    height=400,
                    showlegend=False,
                    coloraxis_showscale=False
                )
                
                fig_day.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
                fig_day.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
                
                st.plotly_chart(fig_day, use_container_width=True)
            
            with col2:
                # Win/Loss pie
                fig_pie = go.Figure(data=[go.Pie(
                    labels=['Winners', 'Losers'],
                    values=[metrics['winners'], metrics['losers']],
                    hole=0.6,
                    marker=dict(colors=['#10b981', '#ef4444']),
                    textinfo='label+value+percent',
                    textfont=dict(size=14, color='white')
                )])
                
                fig_pie.update_layout(
                    title="Win/Loss Distribution",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    height=400,
                    showlegend=False,
                    annotations=[dict(
                        text=f"{metrics['win_rate']:.0f}%<br>Win",
                        x=0.5, y=0.5,
                        font_size=24,
                        font_color='white',
                        showarrow=False
                    )]
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Trade Scatter Plot
            st.markdown("### 📊 Trade Outcomes Over Time")
            
            fig_scatter = px.scatter(
                trades_df,
                x='date',
                y='pnl',
                color=trades_df['pnl'].apply(lambda x: 'Profit' if x > 0 else 'Loss'),
                color_discrete_map={'Profit': '#10b981', 'Loss': '#ef4444'},
                size=trades_df['pnl'].abs(),
                hover_data=['spot_entry', 'atm_strike', 'expiry'],
                title="Individual Trade P&L"
            )
            
            fig_scatter.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400
            )
            
            fig_scatter.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
            fig_scatter.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
            fig_scatter.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
            
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with tab3:
            # Monthly Heatmap
            st.markdown("### 🗓️ Monthly Returns Heatmap")
            
            trades_df['year'] = pd.to_datetime(trades_df['date']).dt.year
            trades_df['month'] = pd.to_datetime(trades_df['date']).dt.month_name()
            
            monthly_pnl = trades_df.groupby(['year', 'month'])['pnl'].sum().reset_index()
            
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            
            pivot = monthly_pnl.pivot(index='month', columns='year', values='pnl').fillna(0)
            pivot = pivot.reindex([m for m in month_order if m in pivot.index])
            
            fig_heat = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale=[
                    [0, '#ef4444'],
                    [0.5, '#1e1e4a'],
                    [1, '#10b981']
                ],
                zmid=0,
                text=[[f"₹{v:,.0f}" for v in row] for row in pivot.values],
                texttemplate='%{text}',
                textfont=dict(size=12, color='white'),
                hovertemplate='Year: %{x}<br>Month: %{y}<br>P&L: ₹%{z:,.0f}<extra></extra>'
            ))
            
            fig_heat.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=500,
                xaxis_title="Year",
                yaxis_title="Month"
            )
            
            st.plotly_chart(fig_heat, use_container_width=True)
        
        with tab4:
            st.markdown("### 📋 Detailed Trade Log")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_outcome = st.selectbox("Filter by Outcome", ["All", "Winners", "Losers"])
            with col2:
                sort_by = st.selectbox("Sort by", ["Date", "P&L", "Spot Price"])
            with col3:
                sort_order = st.selectbox("Order", ["Descending", "Ascending"])
            
            display_df = trades_df.copy()
            
            if filter_outcome == "Winners":
                display_df = display_df[display_df['pnl'] > 0]
            elif filter_outcome == "Losers":
                display_df = display_df[display_df['pnl'] < 0]
            
            sort_col = {'Date': 'date', 'P&L': 'pnl', 'Spot Price': 'spot_entry'}[sort_by]
            display_df = display_df.sort_values(sort_col, ascending=(sort_order == "Ascending"))
            
            # Format for display
            display_cols = ['date', 'spot_entry', 'atm_strike', 'expiry', 'gross_pnl', 'brokerage', 'pnl', 'cumulative_pnl']
            display_df_show = display_df[display_cols].copy()
            display_df_show.columns = ['Date', 'Spot', 'ATM Strike', 'Expiry', 'Gross P&L', 'Brokerage', 'Net P&L', 'Cumulative']
            
            st.dataframe(
                display_df_show.style.format({
                    'Spot': '{:,.2f}',
                    'Gross P&L': '₹{:,.2f}',
                    'Brokerage': '₹{:,.2f}',
                    'Net P&L': '₹{:,.2f}',
                    'Cumulative': '₹{:,.2f}'
                }).applymap(
                    lambda x: 'color: #10b981' if isinstance(x, (int, float)) and x > 0 else 'color: #ef4444' if isinstance(x, (int, float)) and x < 0 else '',
                    subset=['Net P&L']
                ),
                use_container_width=True,
                height=500
            )
            
            # Export button
            csv = display_df_show.to_csv(index=False)
            st.download_button(
                label="📥 Download Trade Log (CSV)",
                data=csv,
                file_name=f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with tab5:
            st.markdown("### 📉 Detailed Risk Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Return Metrics")
                return_metrics = {
                    'Total P&L': f"₹{metrics['total_pnl']:,.2f}",
                    'Return on Capital': f"{metrics['return_pct']:.2f}%",
                    'Average Trade': f"₹{metrics['avg_trade']:,.2f}",
                    'Best Trade': f"₹{metrics['best_trade']:,.2f}",
                    'Worst Trade': f"₹{metrics['worst_trade']:,.2f}",
                    'Expectancy': f"₹{metrics['expectancy']:,.2f}"
                }
                st.table(pd.DataFrame(list(return_metrics.items()), columns=['Metric', 'Value']))
            
            with col2:
                st.markdown("#### Risk Metrics")
                risk_metrics = {
                    'Max Drawdown': f"₹{metrics['max_drawdown']:,.2f}",
                    'Max DD %': f"{metrics['max_dd_pct']:.2f}%",
                    'Sharpe Ratio': f"{metrics['sharpe_ratio']:.3f}",
                    'Sortino Ratio': f"{metrics['sortino_ratio']:.3f}",
                    'Calmar Ratio': f"{metrics['calmar_ratio']:.3f}",
                    'Profit Factor': f"{metrics['profit_factor']:.2f}" if metrics['profit_factor'] < 100 else "∞"
                }
                st.table(pd.DataFrame(list(risk_metrics.items()), columns=['Metric', 'Value']))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Trade Statistics")
                trade_stats = {
                    'Total Trades': metrics['total_trades'],
                    'Winners': metrics['winners'],
                    'Losers': metrics['losers'],
                    'Win Rate': f"{metrics['win_rate']:.1f}%",
                    'Avg Winner': f"₹{metrics['avg_winner']:,.2f}",
                    'Avg Loser': f"₹{metrics['avg_loser']:,.2f}"
                }
                st.table(pd.DataFrame(list(trade_stats.items()), columns=['Metric', 'Value']))
            
            with col2:
                st.markdown("#### Consistency Metrics")
                consistency = {
                    'Max Win Streak': metrics['max_win_streak'],
                    'Max Loss Streak': metrics['max_loss_streak'],
                    'Profitable Months': f"{metrics['monthly_win_rate']:.1f}%"
                }
                st.table(pd.DataFrame(list(consistency.items()), columns=['Metric', 'Value']))
    
    else:
        # No results yet - show instructions
        st.markdown("""
        <div style="text-align: center; padding: 80px 40px; background: rgba(30,30,74,0.3); border-radius: 16px; margin: 40px 0;">
            <h2 style="color: #00d4ff;">👋 Welcome to NIFTY Options Backtester</h2>
            <p style="color: #a0a0c0; font-size: 18px; margin: 20px 0;">
                Configure your strategy in the sidebar and click <strong>RUN BACKTEST</strong> to get started.
            </p>
            <div style="display: flex; justify-content: center; gap: 40px; margin-top: 40px; flex-wrap: wrap;">
                <div style="background: rgba(0,212,255,0.1); padding: 20px; border-radius: 12px; width: 200px;">
                    <div style="font-size: 36px;">📊</div>
                    <div style="color: white; font-weight: 600; margin: 10px 0;">9 Strategies</div>
                    <div style="color: #a0a0c0; font-size: 14px;">Pre-built templates</div>
                </div>
                <div style="background: rgba(124,58,237,0.1); padding: 20px; border-radius: 12px; width: 200px;">
                    <div style="font-size: 36px;">📈</div>
                    <div style="color: white; font-weight: 600; margin: 10px 0;">20+ Metrics</div>
                    <div style="color: #a0a0c0; font-size: 14px;">Comprehensive analytics</div>
                </div>
                <div style="background: rgba(16,185,129,0.1); padding: 20px; border-radius: 12px; width: 200px;">
                    <div style="font-size: 36px;">🗓️</div>
                    <div style="color: white; font-weight: 600; margin: 10px 0;">4 Years Data</div>
                    <div style="color: #a0a0c0; font-size: 14px;">Minute-level precision</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
