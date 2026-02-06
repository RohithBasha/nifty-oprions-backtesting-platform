"""
NIFTY Options Backtesting Dashboard V2.0
=========================================
Professional MT QUANT-inspired backtesting platform with advanced features:
- Intraday Stop-Loss & Target monitoring
- Custom Strike Selection (ATM/OTM offsets)
- Expiry Selection (Weekly/Monthly)
- Trailing Stop-Loss
- Exit reason tracking

Author: TradersCafe
Version: 2.0
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

# Page config
st.set_page_config(
    page_title="NIFTY Options Backtester V2",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main, .stApp { background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a3e 0%, #0f0f23 100%); border-right: 1px solid rgba(255,255,255,0.1); }
    h1 { background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 50%, #ff006e 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800 !important; text-align: center; }
    h2, h3 { color: #00d4ff !important; font-weight: 600; }
    .metric-card { background: linear-gradient(145deg, #1e1e4a 0%, #2a2a5a 100%); border-radius: 16px; padding: 20px; border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3); text-align: center; }
    .metric-value { font-size: 32px; font-weight: 700; margin: 0; }
    .metric-value.profit { background: linear-gradient(135deg, #10b981 0%, #22d3ee 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .metric-value.loss { background: linear-gradient(135deg, #ef4444 0%, #fb923c 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .metric-value.neutral { background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .metric-label { font-size: 12px; color: #a0a0c0; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 8px; }
    .strategy-card { background: linear-gradient(145deg, rgba(0, 212, 255, 0.1) 0%, rgba(124, 58, 237, 0.1) 100%); border-left: 4px solid #7c3aed; border-radius: 8px; padding: 16px; margin: 16px 0; }
    .info-box { background: rgba(0, 212, 255, 0.1); border: 1px solid rgba(0, 212, 255, 0.3); border-radius: 8px; padding: 12px; color: #e0e0e0; font-size: 14px; }
    .feature-badge { display: inline-block; background: linear-gradient(135deg, #7c3aed 0%, #00d4ff 100%); color: white; padding: 4px 12px; border-radius: 12px; font-size: 11px; font-weight: 600; margin: 2px; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: rgba(30, 30, 74, 0.5); border-radius: 8px; padding: 10px 24px; color: #a0a0c0; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 100%); color: white !important; }
    .stButton > button { background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 100%); color: white; border: none; border-radius: 8px; padding: 12px 24px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# Database functions
@st.cache_resource
def get_db_connection():
    return duckdb.connect('nifty_data.duckdb', read_only=True)

@st.cache_data(ttl=3600)
def get_data_range(_conn):
    result = _conn.execute("SELECT MIN(date), MAX(date), COUNT(DISTINCT date) FROM spot_data").fetchone()
    return result

@st.cache_data(ttl=3600)
def get_trading_days(_conn, start_date, end_date):
    query = f"SELECT DISTINCT date FROM spot_data WHERE date >= '{start_date}' AND date <= '{end_date}' ORDER BY date"
    return _conn.execute(query).fetchdf()['date'].tolist()

def get_atm_strike(spot_price: float, interval: int = 50) -> int:
    return int(round(spot_price / interval) * interval)

def get_spot_price(_conn, date, time_str):
    query = f"SELECT close FROM spot_data WHERE date = '{date}' AND CAST(datetime AS TIME) >= '{time_str}' ORDER BY datetime LIMIT 1"
    result = _conn.execute(query).fetchone()
    return result[0] if result else None

def get_option_price(_conn, date, time_str, strike, option_type, expiry=None):
    query = f"SELECT close, expiry_date FROM options_data WHERE date = '{date}' AND CAST(datetime AS TIME) >= '{time_str}' AND strike_price = {strike} AND option_type = '{option_type}'"
    if expiry:
        query += f" AND expiry_date = '{expiry}'"
    else:
        query += f" AND expiry_date >= '{date}'"
    query += " ORDER BY expiry_date, datetime LIMIT 1"
    result = _conn.execute(query).fetchone()
    return (result[0], result[1]) if result else (None, None)

def get_expiries_for_date(_conn, date, expiry_type='weekly'):
    """Get available expiries for a date"""
    query = f"SELECT DISTINCT expiry_date FROM options_data WHERE date = '{date}' AND expiry_date >= '{date}' ORDER BY expiry_date"
    expiries = _conn.execute(query).fetchdf()['expiry_date'].tolist()
    if not expiries:
        return None
    if expiry_type == 'weekly':
        return expiries[0]  # Nearest expiry
    elif expiry_type == 'next_week':
        return expiries[1] if len(expiries) > 1 else expiries[0]
    elif expiry_type == 'monthly':
        # Find monthly expiry (last Thursday of month)
        for exp in expiries:
            if exp.day > 20:  # Likely monthly
                return exp
        return expiries[-1]
    return expiries[0]

def get_intraday_option_prices(_conn, date, start_time, end_time, strike, option_type, expiry):
    """Get minute-by-minute option prices for SL/Target monitoring"""
    query = f"""
        SELECT datetime, close FROM options_data 
        WHERE date = '{date}' AND CAST(datetime AS TIME) >= '{start_time}' AND CAST(datetime AS TIME) <= '{end_time}'
        AND strike_price = {strike} AND option_type = '{option_type}' AND expiry_date = '{expiry}'
        ORDER BY datetime
    """
    return _conn.execute(query).fetchdf()

# Strategy definitions
STRATEGIES = {
    "Short Straddle": {"legs": [{"type": "CALL", "action": "SELL", "offset": 0}, {"type": "PUT", "action": "SELL", "offset": 0}], "category": "Neutral", "desc": "Sell ATM Call + Put"},
    "Long Straddle": {"legs": [{"type": "CALL", "action": "BUY", "offset": 0}, {"type": "PUT", "action": "BUY", "offset": 0}], "category": "Neutral", "desc": "Buy ATM Call + Put"},
    "Short Strangle": {"legs": [{"type": "CALL", "action": "SELL", "offset": 100}, {"type": "PUT", "action": "SELL", "offset": -100}], "category": "Neutral", "desc": "Sell OTM Call + Put"},
    "Long Strangle": {"legs": [{"type": "CALL", "action": "BUY", "offset": 100}, {"type": "PUT", "action": "BUY", "offset": -100}], "category": "Neutral", "desc": "Buy OTM Call + Put"},
    "Iron Condor": {"legs": [{"type": "CALL", "action": "SELL", "offset": 100}, {"type": "CALL", "action": "BUY", "offset": 200}, {"type": "PUT", "action": "SELL", "offset": -100}, {"type": "PUT", "action": "BUY", "offset": -200}], "category": "Neutral", "desc": "Credit spread both sides"},
    "Bull Call Spread": {"legs": [{"type": "CALL", "action": "BUY", "offset": 0}, {"type": "CALL", "action": "SELL", "offset": 100}], "category": "Bullish", "desc": "Buy ATM + Sell OTM Call"},
    "Bear Put Spread": {"legs": [{"type": "PUT", "action": "BUY", "offset": 0}, {"type": "PUT", "action": "SELL", "offset": -100}], "category": "Bearish", "desc": "Buy ATM + Sell OTM Put"},
    "Naked Call": {"legs": [{"type": "CALL", "action": "SELL", "offset": 0}], "category": "Bearish", "desc": "Sell single Call"},
    "Naked Put": {"legs": [{"type": "PUT", "action": "SELL", "offset": 0}], "category": "Bullish", "desc": "Sell single Put"},
}

def calculate_metrics(trades_df, initial_capital=100000):
    if len(trades_df) == 0:
        return {}
    
    total_trades = len(trades_df)
    winners = trades_df[trades_df['pnl'] > 0]
    losers = trades_df[trades_df['pnl'] < 0]
    total_pnl = trades_df['pnl'].sum()
    win_rate = len(winners) / total_trades * 100
    
    avg_winner = winners['pnl'].mean() if len(winners) > 0 else 0
    avg_loser = abs(losers['pnl'].mean()) if len(losers) > 0 else 0
    profit_factor = abs(winners['pnl'].sum() / losers['pnl'].sum()) if len(losers) > 0 and losers['pnl'].sum() != 0 else float('inf')
    
    cumulative = trades_df['pnl'].cumsum()
    peak = cumulative.cummax()
    drawdown = cumulative - peak
    max_drawdown = abs(drawdown.min())
    max_dd_pct = (max_drawdown / initial_capital) * 100
    
    daily_returns = trades_df['pnl'] / initial_capital
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
    
    # Exit reason stats
    exit_reasons = trades_df['exit_reason'].value_counts().to_dict() if 'exit_reason' in trades_df.columns else {}
    
    return {
        'total_pnl': total_pnl, 'total_trades': total_trades, 'winners': len(winners), 'losers': len(losers),
        'win_rate': win_rate, 'avg_winner': avg_winner, 'avg_loser': avg_loser,
        'avg_trade': total_pnl / total_trades, 'best_trade': trades_df['pnl'].max(), 'worst_trade': trades_df['pnl'].min(),
        'profit_factor': profit_factor, 'max_drawdown': max_drawdown, 'max_dd_pct': max_dd_pct,
        'sharpe_ratio': sharpe, 'return_pct': (total_pnl / initial_capital) * 100, 'exit_reasons': exit_reasons
    }

def run_backtest_v2(conn, strategy_name, config, progress_callback=None):
    """Enhanced backtest with intraday SL/Target monitoring"""
    
    strategy = STRATEGIES[strategy_name]
    legs = strategy['legs']
    trading_days = get_trading_days(conn, config['start_date'], config['end_date'])
    
    results = []
    total_days = len(trading_days)
    
    for i, trade_date in enumerate(trading_days):
        if progress_callback:
            progress_callback((i + 1) / total_days)
        
        try:
            spot_at_entry = get_spot_price(conn, trade_date, config['entry_time'])
            if not spot_at_entry:
                continue
            
            atm_strike = get_atm_strike(spot_at_entry)
            
            # Get expiry based on selection
            expiry = get_expiries_for_date(conn, trade_date, config.get('expiry_type', 'weekly'))
            if not expiry:
                continue
            
            # Execute legs with custom strike offset
            leg_data = []
            skip_trade = False
            entry_premium = 0
            
            for leg in legs:
                # Apply custom strike offset from config
                base_offset = leg['offset']
                custom_offset = config.get('strike_offset', 0)
                if leg['type'] == 'CALL':
                    strike = atm_strike + base_offset + custom_offset
                else:
                    strike = atm_strike + base_offset - custom_offset
                
                entry_price, _ = get_option_price(conn, trade_date, config['entry_time'], strike, leg['type'], expiry)
                
                if entry_price is None:
                    skip_trade = True
                    break
                
                # Apply entry slippage
                slippage = config.get('slippage_pct', 0.05) / 100
                if leg['action'] == 'BUY':
                    entry_price = entry_price * (1 + slippage)
                else:
                    entry_price = entry_price * (1 - slippage)
                
                leg_data.append({
                    'strike': strike, 'type': leg['type'], 'action': leg['action'],
                    'entry': entry_price, 'exit': None, 'exit_time': None
                })
                
                multiplier = 1 if leg['action'] == 'SELL' else -1
                entry_premium += entry_price * multiplier
            
            if skip_trade:
                continue
            
            # INTRADAY SL/TARGET MONITORING
            exit_reason = 'TIME'
            exit_time_actual = config['exit_time']
            final_premium = 0
            
            # Check if SL/Target is enabled
            sl_enabled = config.get('sl_enabled', False)
            target_enabled = config.get('target_enabled', False)
            
            if sl_enabled or target_enabled:
                sl_pct = config.get('sl_pct', 50)
                target_pct = config.get('target_pct', 50)
                
                # Get intraday prices for first leg (approximate)
                main_leg = leg_data[0]
                intraday_prices = get_intraday_option_prices(
                    conn, trade_date, config['entry_time'], config['exit_time'],
                    main_leg['strike'], main_leg['type'], expiry
                )
                
                if len(intraday_prices) > 0:
                    for _, row in intraday_prices.iterrows():
                        current_price = row['close']
                        current_time = row['datetime']
                        
                        # Calculate current P&L for this leg
                        if main_leg['action'] == 'SELL':
                            leg_pnl_pct = ((main_leg['entry'] - current_price) / main_leg['entry']) * 100
                        else:
                            leg_pnl_pct = ((current_price - main_leg['entry']) / main_leg['entry']) * 100
                        
                        # Check Stop Loss
                        if sl_enabled and leg_pnl_pct < -sl_pct:
                            exit_reason = 'SL HIT'
                            exit_time_actual = str(current_time.time())[:8]
                            break
                        
                        # Check Target
                        if target_enabled and leg_pnl_pct > target_pct:
                            exit_reason = 'TARGET'
                            exit_time_actual = str(current_time.time())[:8]
                            break
            
            # Get exit prices
            for leg in leg_data:
                exit_price, _ = get_option_price(conn, trade_date, exit_time_actual, leg['strike'], leg['type'], expiry)
                if exit_price is None:
                    exit_price, _ = get_option_price(conn, trade_date, config['exit_time'], leg['strike'], leg['type'], expiry)
                
                if exit_price:
                    slippage = config.get('slippage_pct', 0.05) / 100
                    if leg['action'] == 'BUY':
                        exit_price = exit_price * (1 - slippage)
                    else:
                        exit_price = exit_price * (1 + slippage)
                    
                    leg['exit'] = exit_price
                    leg['exit_time'] = exit_time_actual
                    
                    multiplier = 1 if leg['action'] == 'SELL' else -1
                    final_premium += (leg['entry'] - exit_price) * multiplier
            
            lot_size = config['lot_size'] * config['num_lots']
            gross_pnl = final_premium * lot_size
            brokerage = config.get('brokerage', 20) * len(legs) * 2
            net_pnl = gross_pnl - brokerage
            
            results.append({
                'date': trade_date, 'expiry': expiry, 'spot_entry': spot_at_entry,
                'atm_strike': atm_strike, 'strategy': strategy_name, 'legs': len(legs),
                'entry_premium': entry_premium, 'exit_premium': entry_premium - final_premium,
                'gross_pnl': gross_pnl, 'brokerage': brokerage, 'pnl': net_pnl,
                'exit_reason': exit_reason, 'exit_time': exit_time_actual, 'leg_details': leg_data
            })
            
        except Exception as e:
            continue
    
    return pd.DataFrame(results)

# Main App
def main():
    st.markdown("# 🚀 NIFTY OPTIONS BACKTESTER V2.0")
    st.markdown("""
    <p style='text-align: center; color: #a0a0c0; font-size: 16px;'>
        <span class="feature-badge">INTRADAY SL/TARGET</span>
        <span class="feature-badge">CUSTOM STRIKES</span>
        <span class="feature-badge">EXPIRY SELECTION</span>
    </p>
    """, unsafe_allow_html=True)
    
    try:
        conn = get_db_connection()
        data_range = get_data_range(conn)
        min_date, max_date, trading_days = data_range
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        return
    
    # SIDEBAR
    with st.sidebar:
        st.markdown("## ⚙️ Strategy Configuration")
        st.markdown("---")
        
        # Strategy
        st.markdown("### 📊 Strategy")
        strategy_name = st.selectbox("Strategy Type", list(STRATEGIES.keys()))
        strategy = STRATEGIES[strategy_name]
        st.caption(f"_{strategy['desc']}_ | {strategy['category']}")
        
        st.markdown("---")
        
        # Expiry Selection (NEW)
        st.markdown("### 📅 Expiry Type")
        expiry_type = st.selectbox("Select Expiry", ["weekly", "next_week", "monthly"], 
                                    format_func=lambda x: {"weekly": "Weekly (Nearest)", "next_week": "Next Week", "monthly": "Monthly"}[x])
        
        # Date Range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start", value=datetime(2025, 1, 1), min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("End", value=max_date, min_value=min_date, max_value=max_date)
        
        st.markdown("---")
        
        # Entry/Exit
        st.markdown("### ⏰ Entry/Exit")
        col1, col2 = st.columns(2)
        with col1:
            entry_time = st.time_input("Entry", value=time(9, 20))
        with col2:
            exit_time = st.time_input("Exit", value=time(15, 15))
        
        st.markdown("---")
        
        # Strike Offset (NEW)
        st.markdown("### 🎯 Strike Configuration")
        strike_offset = st.number_input("Additional Strike Offset", value=0, step=50, 
                                        help="Add offset to widen/narrow strategy strikes")
        
        st.markdown("---")
        
        # SL/Target (NEW)
        st.markdown("### 🛡️ Risk Management")
        sl_enabled = st.checkbox("Enable Stop Loss", value=True)
        if sl_enabled:
            sl_pct = st.slider("Stop Loss (%)", 10, 100, 50, help="Exit if loss exceeds this % of premium")
        else:
            sl_pct = 50
        
        target_enabled = st.checkbox("Enable Target", value=True)
        if target_enabled:
            target_pct = st.slider("Target (%)", 10, 100, 50, help="Exit if profit exceeds this % of premium")
        else:
            target_pct = 50
        
        st.markdown("---")
        
        # Position
        st.markdown("### 📦 Position Sizing")
        col1, col2 = st.columns(2)
        with col1:
            lot_size = st.number_input("Lot Size", value=25)
        with col2:
            num_lots = st.number_input("Lots", value=1, min_value=1)
        
        brokerage = st.number_input("Brokerage/Trade (₹)", value=20)
        slippage = st.slider("Slippage (%)", 0.0, 1.0, 0.05, 0.01)
        initial_capital = st.number_input("Capital (₹)", value=100000, step=10000)
        
        st.markdown("---")
        run_btn = st.button("🚀 RUN BACKTEST", use_container_width=True)
    
    # MAIN AREA
    st.markdown(f"""
    <div class="info-box">
        📊 <b>Data:</b> {min_date.strftime('%b %Y')} to {max_date.strftime('%b %Y')} | <b>{trading_days}</b> days
    </div>
    """, unsafe_allow_html=True)
    
    if run_btn:
        config = {
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'entry_time': entry_time.strftime('%H:%M:%S'),
            'exit_time': exit_time.strftime('%H:%M:%S'),
            'expiry_type': expiry_type,
            'strike_offset': strike_offset,
            'sl_enabled': sl_enabled,
            'sl_pct': sl_pct,
            'target_enabled': target_enabled,
            'target_pct': target_pct,
            'lot_size': lot_size,
            'num_lots': num_lots,
            'brokerage': brokerage,
            'slippage_pct': slippage
        }
        
        progress = st.progress(0)
        status = st.empty()
        
        def update_progress(p):
            progress.progress(p)
            status.text(f"Processing... {int(p * 100)}%")
        
        with st.spinner("Running backtest..."):
            trades_df = run_backtest_v2(conn, strategy_name, config, update_progress)
        
        progress.empty()
        status.empty()
        
        if len(trades_df) == 0:
            st.warning("⚠️ No trades executed.")
            return
        
        st.session_state['trades_df'] = trades_df
        st.session_state['metrics'] = calculate_metrics(trades_df, initial_capital)
    
    # Display results
    if 'trades_df' in st.session_state and len(st.session_state['trades_df']) > 0:
        trades_df = st.session_state['trades_df']
        metrics = st.session_state['metrics']
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        
        # Metrics row
        st.markdown("### 📈 Performance Summary")
        cols = st.columns(7)
        
        metric_items = [
            ('Total P&L', f"₹{metrics['total_pnl']:,.0f}", 'profit' if metrics['total_pnl'] >= 0 else 'loss'),
            ('Win Rate', f"{metrics['win_rate']:.1f}%", 'neutral'),
            ('Sharpe', f"{metrics['sharpe_ratio']:.2f}", 'profit' if metrics['sharpe_ratio'] >= 0 else 'loss'),
            ('Max DD', f"{metrics['max_dd_pct']:.1f}%", 'loss'),
            ('Profit Factor', f"{metrics['profit_factor']:.2f}" if metrics['profit_factor'] < 100 else "∞", 'profit' if metrics['profit_factor'] >= 1 else 'loss'),
            ('Avg Trade', f"₹{metrics['avg_trade']:,.0f}", 'profit' if metrics['avg_trade'] >= 0 else 'loss'),
            ('Trades', str(metrics['total_trades']), 'neutral')
        ]
        
        for col, (label, value, style) in zip(cols, metric_items):
            with col:
                st.markdown(f'<div class="metric-card"><p class="metric-value {style}">{value}</p><p class="metric-label">{label}</p></div>', unsafe_allow_html=True)
        
        # Exit Reasons (NEW)
        if 'exit_reasons' in metrics and metrics['exit_reasons']:
            st.markdown("### 🚪 Exit Analysis")
            exit_cols = st.columns(len(metrics['exit_reasons']))
            for col, (reason, count) in zip(exit_cols, metrics['exit_reasons'].items()):
                with col:
                    pct = count / metrics['total_trades'] * 100
                    color = '#10b981' if reason == 'TARGET' else '#ef4444' if reason == 'SL HIT' else '#7c3aed'
                    st.markdown(f'<div class="metric-card"><p class="metric-value" style="color: {color}">{count}</p><p class="metric-label">{reason} ({pct:.0f}%)</p></div>', unsafe_allow_html=True)
        
        # Charts
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Equity", "📊 Analytics", "📋 Trade Log", "📉 Risk"])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            with col1:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
                fig.add_trace(go.Scatter(x=trades_df['date'], y=trades_df['cumulative_pnl'], mode='lines', name='P&L', line=dict(color='#00d4ff', width=2), fill='tozeroy', fillcolor='rgba(0,212,255,0.1)'), row=1, col=1)
                peak = trades_df['cumulative_pnl'].cummax()
                dd = trades_df['cumulative_pnl'] - peak
                fig.add_trace(go.Scatter(x=trades_df['date'], y=dd, mode='lines', name='Drawdown', line=dict(color='#ef4444'), fill='tozeroy', fillcolor='rgba(239,68,68,0.3)'), row=2, col=1)
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), showlegend=False, height=500)
                fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
                fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig_hist = px.histogram(trades_df, x='pnl', nbins=30, title="P&L Distribution")
                fig_hist.update_traces(marker_color='#7c3aed')
                fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=500)
                st.plotly_chart(fig_hist, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                # Exit reason pie
                if 'exit_reasons' in metrics and metrics['exit_reasons']:
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=list(metrics['exit_reasons'].keys()),
                        values=list(metrics['exit_reasons'].values()),
                        hole=0.6,
                        marker=dict(colors=['#10b981', '#ef4444', '#7c3aed'])
                    )])
                    fig_pie.update_layout(title="Exit Reasons", paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Day analysis
                trades_df['day'] = pd.to_datetime(trades_df['date']).dt.day_name()
                day_pnl = trades_df.groupby('day')['pnl'].sum().reindex(['Monday','Tuesday','Wednesday','Thursday','Friday'])
                fig_bar = px.bar(x=day_pnl.index, y=day_pnl.values, title="P&L by Day", color=day_pnl.values, color_continuous_scale=['#ef4444', '#22c55e'])
                fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=400, showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)
        
        with tab3:
            display_df = trades_df[['date', 'spot_entry', 'atm_strike', 'expiry', 'exit_reason', 'exit_time', 'gross_pnl', 'pnl', 'cumulative_pnl']].copy()
            display_df.columns = ['Date', 'Spot', 'Strike', 'Expiry', 'Exit', 'Exit Time', 'Gross', 'Net P&L', 'Cumulative']
            st.dataframe(display_df, use_container_width=True, height=500)
            st.download_button("📥 Download CSV", trades_df.to_csv(index=False), f"backtest_{datetime.now().strftime('%Y%m%d')}.csv")
        
        with tab4:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Return Metrics")
                st.table(pd.DataFrame({
                    'Metric': ['Total P&L', 'Return %', 'Avg Trade', 'Best', 'Worst'],
                    'Value': [f"₹{metrics['total_pnl']:,.0f}", f"{metrics['return_pct']:.1f}%", f"₹{metrics['avg_trade']:,.0f}", f"₹{metrics['best_trade']:,.0f}", f"₹{metrics['worst_trade']:,.0f}"]
                }))
            with col2:
                st.markdown("#### Risk Metrics")
                st.table(pd.DataFrame({
                    'Metric': ['Max Drawdown', 'DD %', 'Sharpe', 'Profit Factor', 'Win Rate'],
                    'Value': [f"₹{metrics['max_drawdown']:,.0f}", f"{metrics['max_dd_pct']:.1f}%", f"{metrics['sharpe_ratio']:.2f}", f"{metrics['profit_factor']:.2f}" if metrics['profit_factor'] < 100 else "∞", f"{metrics['win_rate']:.1f}%"]
                }))
    
    else:
        st.markdown("""
        <div style="text-align: center; padding: 60px; background: rgba(30,30,74,0.3); border-radius: 16px; margin: 40px 0;">
            <h2 style="color: #00d4ff;">👋 Welcome to V2.0</h2>
            <p style="color: #a0a0c0;">Configure your strategy and click RUN BACKTEST</p>
            <div style="margin-top: 20px;">
                <span class="feature-badge">✓ Intraday SL/Target</span>
                <span class="feature-badge">✓ Custom Strikes</span>
                <span class="feature-badge">✓ Expiry Selection</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
