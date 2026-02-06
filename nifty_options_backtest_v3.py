"""
NIFTY Options Backtesting Platform V3.4 - AIRTIGHT EDITION
============================================================
All V3.3 features + Monte Carlo & PDF export:
- Startup dependency validation
- Python version checking
- Database connection resilience (fixed db_filename bug)
- Exhaustive input validation & SQL parameter sanitization
- Session state protection
- Chart error boundaries with fallbacks
- Circuit breaker pattern for cascading failures
- Global exception handler with restart capability
- Zero-division guards (capital, premium stats, atm_strike)
- Safe date/expiry handling across all DB queries
- Compare mode & Simulator crash prevention
- Streamlit API compatibility (rerun fallback)
- Monte Carlo simulation (confidence intervals, histograms)
- PDF report export (config, metrics, chart, trades)

Author: TradersCafe
Version: 3.4 Airtight
"""

import sys
import os

# ============================================================================
# STARTUP SAFEGUARDS - Phase 1
# ============================================================================

def check_python_version():
    """Ensure compatible Python version"""
    return sys.version_info >= (3, 8)

def check_dependencies():
    """Validate all required packages at startup"""
    required = {
        'streamlit': 'streamlit',
        'pandas': 'pandas', 
        'numpy': 'numpy',
        'duckdb': 'duckdb',
        'plotly': 'plotly'
    }
    missing = []
    for display_name, import_name in required.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(display_name)
    return missing

# Pre-flight checks before any heavy imports
if not check_python_version():
    print("ERROR: Python 3.8+ required")
    sys.exit(1)

_missing_deps = check_dependencies()
if _missing_deps:
    print(f"ERROR: Missing packages: {', '.join(_missing_deps)}")
    print(f"Run: pip install {' '.join(_missing_deps)}")
    sys.exit(1)

# Now safe to import everything
try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import duckdb
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from datetime import datetime, timedelta, time as dt_time
    from typing import Dict, List, Optional, Tuple, Any, Union
    import warnings
    import traceback
    import logging
    from functools import wraps
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

warnings.filterwarnings('ignore')

# Configure logging with file output for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(page_title="Options Backtester V3.4", page_icon="🚀", layout="wide", initial_sidebar_state="expanded")

# CSS - Professional Black Background Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * { font-family: 'Inter', -apple-system, sans-serif; }
    
    /* Black Background */
    .main, .stApp { background: #0a0a0a; }
    [data-testid="stSidebar"] { 
        background: #111111;
        border-right: 1px solid #222222;
    }
    
    /* Headers - White/Gold accent */
    h1 { 
        color: #ffffff !important;
        font-weight: 700 !important; 
        text-align: center;
        -webkit-text-fill-color: #ffffff;
        background: none;
    }
    h2, h3, h4 { color: #e0e0e0 !important; font-weight: 600; }
    p, span, label, .stMarkdown { color: #a0a0a0; }
    
    /* Metric Cards - Dark with subtle border */
    .metric-card { 
        background: #141414; 
        border-radius: 8px; 
        padding: 18px 14px; 
        border: 1px solid #2a2a2a; 
        text-align: center;
    }
    .metric-value { font-size: 26px; font-weight: 700; margin: 0; }
    .metric-value.profit { color: #22c55e; }
    .metric-value.loss { color: #ef4444; }
    .metric-value.neutral { color: #d4af37; }
    .metric-label { font-size: 11px; color: #808080; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 6px; font-weight: 500; }
    
    /* Feature Badges - Gold accent */
    .feature-badge { 
        display: inline-block; 
        background: rgba(212, 175, 55, 0.15); 
        color: #d4af37; 
        padding: 4px 10px; 
        border-radius: 4px; 
        font-size: 10px; 
        font-weight: 600; 
        margin: 2px;
        border: 1px solid rgba(212, 175, 55, 0.3);
    }
    
    /* Section Headers */
    .section-header { 
        background: #1a1a1a; 
        border-left: 3px solid #d4af37; 
        padding: 10px 14px; 
        margin: 12px 0; 
        border-radius: 0 4px 4px 0; 
        color: #d4af37;
        font-weight: 600;
        font-size: 13px;
    }
    
    /* Messages */
    .error-box { background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3); border-radius: 6px; padding: 12px; color: #ef4444; }
    .warning-box { background: rgba(251,191,36,0.1); border: 1px solid rgba(251,191,36,0.3); border-radius: 6px; padding: 12px; color: #fbbf24; }
    
    #MainMenu, footer { visibility: hidden; }
    
    /* Tabs - Dark */
    .stTabs [data-baseweb="tab-list"] { background: #141414; border-radius: 6px; padding: 3px; }
    .stTabs [data-baseweb="tab"] { 
        background: transparent; 
        border-radius: 4px; 
        padding: 10px 20px; 
        color: #808080;
        font-weight: 500;
        font-size: 14px;
    }
    .stTabs [aria-selected="true"] { 
        background: #222222; 
        color: #ffffff !important;
        font-weight: 600;
    }
    
    /* Primary Button - Clean Professional */
    .stButton > button { 
        background: #1f1f1f; 
        color: #ffffff; 
        border: 1px solid #3a3a3a; 
        border-radius: 6px; 
        font-weight: 600;
        font-size: 14px;
        padding: 0.65rem 1.5rem;
        transition: all 0.2s ease;
        letter-spacing: 0.3px;
    }
    .stButton > button:hover {
        background: #2a2a2a;
        border-color: #4a4a4a;
    }
    .stButton > button:active {
        background: #333333;
    }
    
    /* Inputs - Dark */
    .stSelectbox > div > div, 
    .stNumberInput > div > div > input, 
    .stDateInput > div > div > input,
    .stTextInput > div > div > input {
        background-color: #141414 !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 6px !important;
        color: #e0e0e0 !important;
    }
    
    .stCheckbox label, .stRadio label { color: #a0a0a0 !important; }
    .stSlider > div > div > div { background: #2a2a2a; }
    .stSlider > div > div > div > div { background: #d4af37; }
    
    /* Tables - Dark */
    .dataframe { border: 1px solid #2a2a2a !important; border-radius: 6px; background: #111111 !important; }
    .dataframe th { background: #1a1a1a !important; color: #e0e0e0 !important; font-weight: 600; border-bottom: 1px solid #2a2a2a !important; }
    .dataframe td { color: #a0a0a0 !important; border-bottom: 1px solid #1a1a1a !important; background: #111111 !important; }
    
    .streamlit-expanderHeader { background: #141414; border-radius: 6px; color: #e0e0e0 !important; }
    .stProgress > div > div > div { background: #d4af37; }
    
    /* Scrollbar - Dark */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0a0a0a; }
    ::-webkit-scrollbar-thumb { background: #333333; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #444444; }
    
    .stAlert { border-radius: 6px; }
    
    /* Select dropdown */
    [data-baseweb="select"] > div { background-color: #141414 !important; }
    [data-baseweb="menu"] { background-color: #1a1a1a !important; }
    [data-baseweb="menu"] li { color: #e0e0e0 !important; }
    [data-baseweb="menu"] li:hover { background-color: #222222 !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SAFE UTILITIES - Phase 2 (Enhanced)
# ============================================================================

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with zero check and NaN handling"""
    try:
        if denominator == 0 or pd.isna(denominator) or np.isinf(denominator):
            return default
        if pd.isna(numerator) or np.isinf(numerator):
            return default
        result = numerator / denominator
        return default if np.isnan(result) or np.isinf(result) else result
    except:
        return default

def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert to float with comprehensive checks"""
    try:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            if np.isnan(value) or np.isinf(value):
                return default
            return float(value)
        if isinstance(value, str):
            value = value.strip().replace(',', '')
            if not value:
                return default
        return float(value)
    except:
        return default

def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert to int"""
    try:
        if value is None:
            return default
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return default
        return int(value)
    except:
        return default

def safe_format(value: float, format_str: str = ",.0f", prefix: str = "", suffix: str = "") -> str:
    """Safely format a number with comprehensive edge case handling"""
    try:
        if value is None:
            return "N/A"
        if isinstance(value, float):
            if np.isnan(value):
                return "N/A"
            if np.isinf(value):
                return "∞" if value > 0 else "-∞"
        return f"{prefix}{value:{format_str}}{suffix}"
    except:
        return "N/A"

def safe_dataframe(df: Any, default_cols: List[str] = None) -> pd.DataFrame:
    """Safely get a DataFrame, returning empty with default columns if invalid"""
    try:
        if df is None:
            return pd.DataFrame(columns=default_cols or [])
        if not isinstance(df, pd.DataFrame):
            return pd.DataFrame(columns=default_cols or [])
        if len(df) == 0:
            return pd.DataFrame(columns=default_cols or list(df.columns))
        return df
    except:
        return pd.DataFrame(columns=default_cols or [])

def safe_list(value: Any, default: List = None) -> List:
    """Safely get a list"""
    try:
        if value is None:
            return default or []
        if isinstance(value, list):
            return value
        if isinstance(value, (pd.Series, np.ndarray)):
            return list(value)
        return [value]
    except:
        return default or []

def safe_dict(value: Any, default: Dict = None) -> Dict:
    """Safely get a dict"""
    try:
        if value is None:
            return default or {}
        if isinstance(value, dict):
            return value
        return default or {}
    except:
        return default or {}

def safe_datetime(value: Any, default: datetime = None) -> Optional[datetime]:
    """Safely parse datetime from various formats"""
    try:
        if value is None:
            return default
        if isinstance(value, datetime):
            return value
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        if isinstance(value, str):
            return pd.to_datetime(value).to_pydatetime()
        return default
    except:
        return default

def safe_percentage(value: float, min_val: float = 0.0, max_val: float = 100.0, default: float = 0.0) -> float:
    """Safely get a percentage within bounds"""
    try:
        v = safe_float(value, default)
        return max(min_val, min(max_val, v))
    except:
        return default

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value within range"""
    try:
        v = safe_float(value)
        return max(min_val, min(max_val, v))
    except:
        return min_val

# ============================================================================
# SESSION STATE PROTECTION - Phase 5
# ============================================================================

def safe_session_get(key: str, default: Any = None, validate_type: type = None) -> Any:
    """Safely get session state with type validation"""
    try:
        if key not in st.session_state:
            return default
        value = st.session_state[key]
        if validate_type is not None:
            if not isinstance(value, validate_type):
                logger.warning(f"Session state type mismatch for {key}")
                return default
        return value
    except Exception as e:
        logger.error(f"Session get error for {key}: {e}")
        return default

def safe_session_set(key: str, value: Any) -> bool:
    """Safely set session state with error handling"""
    try:
        st.session_state[key] = value
        return True
    except Exception as e:
        logger.error(f"Session set error for {key}: {e}")
        return False

def clear_session_state():
    """Safely clear all session state"""
    try:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
    except:
        pass

# ============================================================================
# CIRCUIT BREAKER PATTERN - Phase 7
# ============================================================================

class CircuitBreaker:
    """Prevent cascading failures with circuit breaker pattern"""
    
    def __init__(self, max_failures: int = 5, reset_timeout: int = 60):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self._failures = 0
        self._last_failure = None
        self._is_open = False
    
    def record_failure(self):
        """Record a failure"""
        self._failures += 1
        self._last_failure = datetime.now()
        if self._failures >= self.max_failures:
            self._is_open = True
            logger.warning(f"Circuit breaker opened after {self._failures} failures")
    
    def record_success(self):
        """Record a success - partial reset"""
        if self._failures > 0:
            self._failures = max(0, self._failures - 1)
    
    def reset(self):
        """Full reset"""
        self._failures = 0
        self._is_open = False
        self._last_failure = None
    
    def should_skip(self) -> bool:
        """Check if operations should be skipped"""
        if not self._is_open:
            return False
        # Auto-reset after timeout
        if self._last_failure:
            elapsed = (datetime.now() - self._last_failure).seconds
            if elapsed >= self.reset_timeout:
                self.reset()
                return False
        return True
    
    @property
    def failure_count(self) -> int:
        return self._failures

# Global circuit breakers for different operations
_db_breaker = CircuitBreaker(max_failures=3, reset_timeout=30)
_backtest_breaker = CircuitBreaker(max_failures=10, reset_timeout=60)

# ============================================================================
# DATABASE LAYER WITH RESILIENCE - Phase 3
# ============================================================================

def _get_db_path() -> str:
    """Get database file path with fallback locations"""
    locations = [
        'nifty_data.duckdb',
        os.path.join(os.path.dirname(__file__), 'nifty_data.duckdb'),
        os.path.join(os.getcwd(), 'nifty_data.duckdb')
    ]
    for loc in locations:
        if os.path.exists(loc):
            return loc
    return 'nifty_data.duckdb'  # Default

def _test_connection(conn) -> bool:
    """Test if connection is still alive"""
    try:
        if conn is None:
            return False
        conn.execute("SELECT 1").fetchone()
        return True
    except:
        return False

@st.cache_resource
def get_db():
    """Get database connection with health check and error handling"""
    global _db_breaker
    
    # Check circuit breaker
    if _db_breaker.should_skip():
        logger.warning("Database circuit breaker is open")
        return None
    
    try:
        db_path = _get_db_path()
        
        if not os.path.exists(db_path):
            db_name = os.path.basename(db_path)
            st.error(f"❌ Database file not found: {db_path}")
            st.info(f"💡 Please ensure '{db_name}' is in the TradersCafe folder.")
            _db_breaker.record_failure()
            return None
        
        conn = duckdb.connect(db_path, read_only=True)
        
        # Test connection
        if not _test_connection(conn):
            raise Exception("Connection test failed")
        
        _db_breaker.record_success()
        return conn
        
    except FileNotFoundError:
        db_name = os.path.basename(_get_db_path())
        st.error(f"❌ Database file '{db_name}' not found.")
        st.info("💡 Please ensure the database file is in the TradersCafe folder.")
        _db_breaker.record_failure()
        return None
    except duckdb.IOException as e:
        st.error(f"❌ Database I/O error: {str(e)}")
        st.info("💡 The database may be locked by another process. Close other applications and try again.")
        _db_breaker.record_failure()
        return None
    except Exception as e:
        st.error(f"❌ Database connection failed: {str(e)}")
        logger.error(f"Database error: {traceback.format_exc()}")
        _db_breaker.record_failure()
        return None

def execute_query_safe(conn, query: str, default=None):
    """Execute a query with comprehensive error handling"""
    try:
        if conn is None:
            return default
        result = conn.execute(query).fetchone()
        return result if result else default
    except duckdb.CatalogException as e:
        logger.error(f"Table/column not found: {e}")
        return default
    except duckdb.ParserException as e:
        logger.error(f"Query syntax error: {e}")
        return default
    except Exception as e:
        logger.error(f"Query error: {e}")
        return default

def execute_query_df_safe(conn, query: str) -> pd.DataFrame:
    """Execute a query returning DataFrame with error handling"""
    try:
        if conn is None:
            return pd.DataFrame()
        return conn.execute(query).fetchdf()
    except Exception as e:
        logger.error(f"Query error: {e}")
        return pd.DataFrame()

def get_data_info(_conn) -> Tuple[Optional[datetime], Optional[datetime], int]:
    """Get data range with error handling - no caching to prevent stale data"""
    try:
        if _conn is None:
            logger.error("get_data_info: Database connection is None")
            return None, None, 0
        result = _conn.execute("SELECT MIN(date), MAX(date), COUNT(DISTINCT date) FROM spot_data").fetchone()
        if result and result[0] and result[1]:
            logger.info(f"get_data_info: Data range {result[0]} to {result[1]}, {result[2]} days")
            return result[0], result[1], result[2]
        logger.warning("get_data_info: No data found in database")
        return None, None, 0
    except Exception as e:
        logger.error(f"Error getting data info: {e}")
        return None, None, 0

def _safe_date_str(val: Any) -> str:
    """
    Safely convert date to YYYY-MM-DD string for SQL.
    Handles: date objects, datetime objects, pandas Timestamps, string formats.
    """
    try:
        if val is None:
            return ""
        
        # Handle date/datetime objects directly
        if hasattr(val, 'strftime'):
            return val.strftime('%Y-%m-%d')
        
        # Handle objects with year/month/day attributes (like date)
        if hasattr(val, 'year') and hasattr(val, 'month') and hasattr(val, 'day'):
            return f"{val.year:04d}-{val.month:02d}-{val.day:02d}"
        
        # Convert to string for pattern matching
        s = str(val).strip()
        if not s:
            return ""
        
        # Try pandas conversion for various formats
        try:
            ts = pd.to_datetime(s, errors='coerce')
            if ts is not None and ts is not pd.NaT:
                return ts.strftime('%Y-%m-%d')
        except Exception:
            pass
        
        # Handle already-formatted dates (YYYY-MM-DD)
        if len(s) >= 10 and s[0].isdigit():
            # Try to extract YYYY-MM-DD pattern
            date_part = s[:10]
            if '-' in date_part or '/' in date_part:
                date_part = date_part.replace('/', '-')
                parts = date_part.split('-')
                if len(parts) == 3:
                    try:
                        year = int(parts[0])
                        month = int(parts[1])
                        day = int(parts[2])
                        if 2000 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                            return f"{year:04d}-{month:02d}-{day:02d}"
                    except ValueError:
                        pass
        
        return ""
    except Exception:
        return ""


def _safe_to_date(val: Any):
    try:
        if val is None:
            return None
        ts = pd.to_datetime(val, errors='coerce')
        if ts is pd.NaT:
            return None
        return ts.date()
    except Exception:
        return None


def _safe_time_str(val: Any, default: str = "09:20:00") -> str:
    try:
        if val is None:
            return default
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return default
            t = pd.Timestamp(f"2000-01-01 {s}")
            return t.strftime('%H:%M:%S')
        if hasattr(val, 'strftime'):
            return val.strftime('%H:%M:%S')
        t = pd.to_datetime(val, errors='coerce')
        if t is pd.NaT:
            return default
        return t.strftime('%H:%M:%S')
    except Exception:
        return default


def _normalize_strike_offset(offset: Any, step: Any = 50) -> int:
    try:
        step_i = safe_int(step, 50)
        if step_i <= 0:
            step_i = 50
        off_i = safe_int(offset, 0)
        return int(round(off_i / step_i) * step_i)
    except Exception:
        return 0


def get_trading_days(_conn, start: str, end: str) -> List:
    """
    Get trading days with comprehensive error handling.
    Removed caching to prevent stale data issues.
    """
    try:
        if _conn is None:
            logger.error("get_trading_days: Database connection is None")
            return []
        
        # Convert dates to standard format
        start_s = _safe_date_str(start)
        end_s = _safe_date_str(end)
        
        if not start_s:
            logger.warning(f"get_trading_days: Could not parse start date: {start}")
            # Try direct conversion for date objects
            try:
                if hasattr(start, 'strftime'):
                    start_s = start.strftime('%Y-%m-%d')
                elif hasattr(start, 'year'):
                    start_s = f"{start.year:04d}-{start.month:02d}-{start.day:02d}"
            except Exception:
                pass
        
        if not end_s:
            logger.warning(f"get_trading_days: Could not parse end date: {end}")
            # Try direct conversion for date objects
            try:
                if hasattr(end, 'strftime'):
                    end_s = end.strftime('%Y-%m-%d')
                elif hasattr(end, 'year'):
                    end_s = f"{end.year:04d}-{end.month:02d}-{end.day:02d}"
            except Exception:
                pass
        
        if not start_s or not end_s:
            logger.error(f"get_trading_days: Invalid dates after parsing - start: {start_s}, end: {end_s}")
            return []
        
        # Validate date order
        try:
            start_ts = pd.Timestamp(start_s)
            end_ts = pd.Timestamp(end_s)
            if start_ts > end_ts:
                logger.warning(f"get_trading_days: Start date {start_s} is after end date {end_s}")
                return []
        except Exception as e:
            logger.error(f"get_trading_days: Date comparison failed: {e}")
            return []
        
        # Execute query with error handling
        try:
            query = f"SELECT DISTINCT date FROM spot_data WHERE date >= '{start_s}' AND date <= '{end_s}' ORDER BY date"
            df = _conn.execute(query).fetchdf()
            
            if df is None or len(df) == 0:
                logger.info(f"get_trading_days: No data found for range {start_s} to {end_s}")
                # Try to get any data to verify DB is working
                check_df = _conn.execute("SELECT COUNT(*) as cnt FROM spot_data").fetchdf()
                if check_df is not None and len(check_df) > 0:
                    total_rows = check_df['cnt'].iloc[0]
                    logger.info(f"get_trading_days: Database has {total_rows} total rows")
                return []
            
            # Convert dates to string list
            dates = []
            for d in df['date'].tolist():
                if d is not None:
                    try:
                        if hasattr(d, 'strftime'):
                            dates.append(d.strftime('%Y-%m-%d'))
                        else:
                            dates.append(str(d)[:10])
                    except Exception:
                        dates.append(str(d)[:10])
            
            logger.info(f"get_trading_days: Found {len(dates)} trading days from {start_s} to {end_s}")
            return dates
            
        except Exception as e:
            logger.error(f"get_trading_days: Query execution failed: {e}")
            # Try reconnecting
            try:
                df = _conn.execute("SELECT 1").fetchdf()
                logger.info("get_trading_days: DB connection test passed")
            except Exception as conn_e:
                logger.error(f"get_trading_days: DB connection test failed: {conn_e}")
            return []
            
    except Exception as e:
        logger.error(f"get_trading_days: Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def get_spot(_conn, date: str, time_str: str) -> Optional[float]:
    """Get spot price with error handling"""
    try:
        if _conn is None:
            return None
        date_s = _safe_date_str(date)
        if not date_s:
            return None
        time_s = _safe_time_str(time_str, '09:20:00')
        r = _conn.execute(f"SELECT close FROM spot_data WHERE date = '{date_s}' AND CAST(datetime AS TIME) >= '{time_s}' ORDER BY datetime LIMIT 1").fetchone()
        return safe_float(r[0]) if r else None
    except Exception as e:
        logger.error(f"Error getting spot for {date} {time_str}: {e}")
        return None

def get_prev_day_data(_conn, date: str) -> Optional[Dict]:
    """Get previous day OHLC with error handling"""
    try:
        if _conn is None:
            return None
        date_s = _safe_date_str(date)
        if not date_s:
            return None
        r = _conn.execute(f"""
            SELECT open, high, low, close 
            FROM spot_data 
            WHERE date < '{date_s}' 
            ORDER BY date DESC, datetime DESC LIMIT 1
        """).fetchone()
        if not r:
            return None
        return {
            'open': safe_float(r[0]),
            'high': safe_float(r[1]),
            'low': safe_float(r[2]),
            'close': safe_float(r[3])
        }
    except Exception as e:
        logger.error(f"Error getting prev day data: {e}")
        return None

def get_option(_conn, date: str, time_str: str, strike: int, opt_type: str, expiry=None) -> Tuple[Optional[float], Optional[Any]]:
    """Get option price with error handling"""
    try:
        if _conn is None:
            return None, None
        date_s = _safe_date_str(date)
        time_s = _safe_time_str(time_str, '09:20:00')
        strike = safe_int(strike, 0)
        opt_type_s = (opt_type or '').upper().strip()
        if not date_s or strike <= 0 or opt_type_s not in {'CALL', 'PUT'}:
            return None, None
        q = f"SELECT close, expiry_date FROM options_data WHERE date = '{date_s}' AND CAST(datetime AS TIME) >= '{time_s}' AND strike_price = {strike} AND option_type = '{opt_type_s}'"
        q += f" AND expiry_date = '{_safe_date_str(expiry)}'" if expiry and _safe_date_str(expiry) else f" AND expiry_date >= '{date_s}'"
        q += " ORDER BY expiry_date, datetime LIMIT 1"
        r = _conn.execute(q).fetchone()
        return (safe_float(r[0]), r[1]) if r else (None, None)
    except Exception as e:
        logger.error(f"Error getting option {strike} {opt_type}: {e}")
        return None, None

def get_expiry(_conn, date: str, exp_type: str = 'weekly') -> Optional[Any]:
    """Get expiry with error handling"""
    try:
        if _conn is None:
            return None
        date_s = _safe_date_str(date)
        if not date_s:
            return None
        expiries = _conn.execute(f"SELECT DISTINCT expiry_date FROM options_data WHERE date = '{date_s}' AND expiry_date >= '{date_s}' ORDER BY expiry_date").fetchdf()['expiry_date'].tolist()
        if not expiries:
            return None
        if exp_type == 'weekly':
            return expiries[0]
        if exp_type == 'next_week':
            return expiries[1] if len(expiries) > 1 else expiries[0]
        if exp_type == 'monthly':
            for e in expiries:
                try:
                    day = getattr(e, 'day', None) or (pd.Timestamp(e).day if e else None)
                    if day is not None and day > 20:
                        return e
                except Exception:
                    continue
            return expiries[-1]
        return expiries[0]
    except Exception as e:
        logger.error(f"Error getting expiry: {e}")
        return None

def get_intraday_options(_conn, date: str, start: str, end: str, strike: int, opt_type: str, expiry) -> pd.DataFrame:
    """Get intraday options with error handling"""
    try:
        if _conn is None:
            return pd.DataFrame()
        date_s = _safe_date_str(date)
        expiry_s = _safe_date_str(expiry) if expiry else ""
        if not date_s or not expiry_s:
            return pd.DataFrame()
        start_s = _safe_time_str(start, '09:15:00')
        end_s = _safe_time_str(end, '15:30:00')
        try:
            if pd.Timestamp(f"2000-01-01 {start_s}") > pd.Timestamp(f"2000-01-01 {end_s}"):
                return pd.DataFrame()
        except Exception:
            return pd.DataFrame()
        strike = safe_int(strike, 0)
        if strike <= 0:
            return pd.DataFrame()
        opt_type_s = (opt_type or '').upper().strip()
        if opt_type_s not in {'CALL', 'PUT'}:
            return pd.DataFrame()
        return _conn.execute(f"SELECT datetime, close FROM options_data WHERE date = '{date_s}' AND CAST(datetime AS TIME) >= '{start_s}' AND CAST(datetime AS TIME) <= '{end_s}' AND strike_price = {strike} AND option_type = '{opt_type_s}' AND expiry_date = '{expiry_s}' ORDER BY datetime").fetchdf()
    except Exception as e:
        logger.error(f"Error getting intraday options: {e}")
        return pd.DataFrame()

def get_intraday_spot(_conn, date: str, start: str = '09:15:00', end: str = '15:30:00') -> pd.DataFrame:
    """Get intraday spot with error handling"""
    try:
        if _conn is None:
            return pd.DataFrame()
        date_s = _safe_date_str(date)
        if not date_s:
            return pd.DataFrame()
        start_s = _safe_time_str(start, '09:15:00')
        end_s = _safe_time_str(end, '15:30:00')
        try:
            if pd.Timestamp(f"2000-01-01 {start_s}") > pd.Timestamp(f"2000-01-01 {end_s}"):
                return pd.DataFrame()
        except Exception:
            return pd.DataFrame()
        return _conn.execute(f"SELECT datetime, open, high, low, close FROM spot_data WHERE date = '{date_s}' AND CAST(datetime AS TIME) >= '{start_s}' AND CAST(datetime AS TIME) <= '{end_s}' ORDER BY datetime").fetchdf()
    except Exception as e:
        logger.error(f"Error getting intraday spot: {e}")
        return pd.DataFrame()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def atm_strike(spot: float, interval: int = 50) -> int:
    """Calculate ATM strike"""
    try:
        if interval is None or interval <= 0:
            interval = 50
        spot = safe_float(spot, 0)
        if spot <= 0:
            return 0
        return int(round(spot / interval) * interval)
    except (ZeroDivisionError, TypeError, ValueError):
        return 0

def validate_config(config: Dict) -> Tuple[bool, str]:
    """Basic validation (backward compatible wrapper)"""
    valid, errors, _ = validate_config_exhaustive(config)
    return valid, "; ".join(errors) if errors else ""

def validate_config_exhaustive(config: Dict) -> Tuple[bool, List[str], List[str]]:
    """
    Exhaustive validation with detailed error messages and warnings.
    Returns: (is_valid, errors, warnings)
    """
    errors = []
    warnings = []
    
    # ===== DATE VALIDATION =====
    if not config.get('start_date'):
        errors.append("Start date is required")
    if not config.get('end_date'):
        errors.append("End date is required")
    
    if config.get('start_date') and config.get('end_date'):
        try:
            start = pd.Timestamp(config['start_date'])
            end = pd.Timestamp(config['end_date'])
            
            if start > end:
                errors.append("Start date must be before end date")
            
            # Warn on very long backtests
            days = (end - start).days
            if days > 365 * 3:
                warnings.append(f"Very long backtest ({days} days) may be slow")
            
            # Warn on future dates
            if end > pd.Timestamp.now():
                warnings.append("End date is in the future - data may not exist")
                
        except Exception as e:
            errors.append(f"Invalid date format: {str(e)}")
    
    # ===== TIME VALIDATION =====
    entry_time = config.get('entry_time', '09:20:00')
    exit_time = config.get('exit_time', '15:15:00')
    
    try:
        entry_dt = pd.Timestamp(f"2020-01-01 {entry_time}")
        exit_dt = pd.Timestamp(f"2020-01-01 {exit_time}")
        
        if entry_dt >= exit_dt:
            errors.append("Entry time must be before exit time")
        
        # Market hours check
        market_open = pd.Timestamp("2020-01-01 09:15:00")
        market_close = pd.Timestamp("2020-01-01 15:30:00")
        
        if entry_dt < market_open:
            warnings.append("Entry time before market open (9:15)")
        if exit_dt > market_close:
            warnings.append("Exit time after market close (15:30)")
            
    except Exception as e:
        errors.append(f"Invalid time format: {str(e)}")
    
    # ===== POSITION SIZE VALIDATION =====
    lot_size = config.get('lot_size', 25)
    num_lots = config.get('num_lots', 1)
    
    if not isinstance(lot_size, (int, float)) or lot_size <= 0:
        errors.append("Lot size must be a positive number")
    elif lot_size > 900:
        warnings.append(f"Unusual lot size ({lot_size}) - NIFTY lot is 25")
    
    if not isinstance(num_lots, (int, float)) or num_lots <= 0:
        errors.append("Number of lots must be positive")
    elif num_lots > 100:
        warnings.append(f"Very large position ({num_lots} lots)")
    
    # ===== STOP LOSS / TARGET VALIDATION =====
    if config.get('sl_enabled', False):
        sl_pct = config.get('sl_pct', 50)
        if not isinstance(sl_pct, (int, float)) or sl_pct < 0:
            errors.append("Stop loss % must be non-negative")
        elif sl_pct > 100:
            warnings.append("Stop loss > 100% - unusual configuration")
        elif sl_pct < 5:
            warnings.append("Very tight stop loss may trigger frequently")
    
    if config.get('target_enabled', False):
        target_pct = config.get('target_pct', 50)
        if not isinstance(target_pct, (int, float)) or target_pct < 0:
            errors.append("Target % must be non-negative")
        elif target_pct > 200:
            warnings.append("Target > 200% - may rarely trigger")
    
    # ===== TRAILING SL VALIDATION =====
    if config.get('trail_enabled', False):
        trail_trigger = config.get('trail_trigger', 20)
        trail_lock = config.get('trail_lock', 0)
        
        if trail_lock > trail_trigger:
            warnings.append("Trail lock > trigger - SL will lock before trail activates")
    
    # ===== SLIPPAGE VALIDATION =====
    slippage = config.get('slippage', 0.05)
    if not isinstance(slippage, (int, float)) or slippage < 0:
        errors.append("Slippage must be non-negative")
    elif slippage > 2:
        warnings.append(f"High slippage ({slippage}%) may significantly impact results")
    
    # ===== BROKERAGE VALIDATION =====
    brokerage = config.get('brokerage', 20)
    if not isinstance(brokerage, (int, float)) or brokerage < 0:
        errors.append("Brokerage must be non-negative")
    
    # ===== DAY FILTER VALIDATION =====
    if config.get('dow_filter_enabled', False):
        allowed_days = config.get('allowed_days', [])
        if not allowed_days:
            errors.append("Day filter enabled but no days selected")
        elif len(allowed_days) < 2:
            warnings.append("Only 1 day selected - limited trading opportunities")
    
    # ===== RISK LIMITS VALIDATION =====
    if config.get('max_daily_loss_enabled', False):
        max_loss = config.get('max_daily_loss', 5000)
        if not isinstance(max_loss, (int, float)) or max_loss <= 0:
            errors.append("Max daily loss must be positive")
    
    if config.get('consec_loss_enabled', False):
        max_consec = config.get('max_consec_losses', 3)
        if not isinstance(max_consec, (int, float)) or max_consec < 1:
            errors.append("Consecutive loss limit must be at least 1")
    
    return len(errors) == 0, errors, warnings

def check_entry_conditions(conn, date: str, config: Dict, prev_day: Optional[Dict]) -> Tuple[bool, str]:
    """Check entry conditions with error handling"""
    try:
        # Day of week filter
        if config.get('dow_filter_enabled'):
            try:
                day_name = pd.Timestamp(date).day_name()
                allowed = config.get('allowed_days', ['Monday','Tuesday','Wednesday','Thursday','Friday'])
                if day_name not in allowed:
                    return False, f"Skipped: {day_name} not allowed"
            except:
                pass
        
        # Expiry day filter
        if config.get('expiry_filter_enabled'):
            try:
                expiry = get_expiry(conn, date, 'weekly')
                date_s = _safe_date_str(date)
                expiry_s = _safe_date_str(expiry) if expiry else ""
                is_expiry = (expiry_s == date_s) if (expiry_s and date_s) else False
                if config.get('expiry_filter') == 'avoid' and is_expiry:
                    return False, "Skipped: Expiry day"
                if config.get('expiry_filter') == 'only' and not is_expiry:
                    return False, "Skipped: Not expiry day"
            except:
                pass
        
        # Gap filter
        if config.get('gap_filter_enabled') and prev_day and prev_day.get('close'):
            try:
                open_price = get_spot(conn, date, '09:15:00')
                if open_price and prev_day['close']:
                    gap_pct = safe_divide(open_price - prev_day['close'], prev_day['close']) * 100
                    min_gap = config.get('min_gap_pct', 0.5)
                    gap_direction = config.get('gap_direction', 'any')
                    
                    if gap_direction == 'up' and gap_pct < min_gap:
                        return False, f"Skipped: Gap {gap_pct:.2f}% < {min_gap}%"
                    if gap_direction == 'down' and gap_pct > -min_gap:
                        return False, f"Skipped: Gap {gap_pct:.2f}% > -{min_gap}%"
                    if gap_direction == 'any' and abs(gap_pct) < min_gap:
                        return False, f"Skipped: Gap {gap_pct:.2f}% < {min_gap}%"
            except:
                pass
        
        # Range filter
        if config.get('range_filter_enabled') and prev_day:
            try:
                if prev_day.get('high') and prev_day.get('low') and prev_day['low'] > 0:
                    prev_range = safe_divide(prev_day['high'] - prev_day['low'], prev_day['low']) * 100
                    min_range = config.get('min_prev_range', 1.0)
                    if prev_range < min_range:
                        return False, f"Skipped: Range {prev_range:.2f}% < {min_range}%"
            except:
                pass
        
        return True, "OK"
    except Exception as e:
        logger.error(f"Error checking entry conditions: {e}")
        return True, "OK"  # Default to allow entry on error

# ============================================================================
# STRATEGIES
# ============================================================================

STRATEGIES = {
    "Short Straddle": {"legs": [{"type": "CALL", "action": "SELL", "offset": 0}, {"type": "PUT", "action": "SELL", "offset": 0}], "cat": "Neutral"},
    "Long Straddle": {"legs": [{"type": "CALL", "action": "BUY", "offset": 0}, {"type": "PUT", "action": "BUY", "offset": 0}], "cat": "Neutral"},
    "Short Strangle": {"legs": [{"type": "CALL", "action": "SELL", "offset": 100}, {"type": "PUT", "action": "SELL", "offset": -100}], "cat": "Neutral"},
    "Long Strangle": {"legs": [{"type": "CALL", "action": "BUY", "offset": 100}, {"type": "PUT", "action": "BUY", "offset": -100}], "cat": "Neutral"},
    "Iron Condor": {"legs": [{"type": "CALL", "action": "SELL", "offset": 100}, {"type": "CALL", "action": "BUY", "offset": 200}, {"type": "PUT", "action": "SELL", "offset": -100}, {"type": "PUT", "action": "BUY", "offset": -200}], "cat": "Neutral"},
    "Bull Call Spread": {"legs": [{"type": "CALL", "action": "BUY", "offset": 0}, {"type": "CALL", "action": "SELL", "offset": 100}], "cat": "Bullish"},
    "Bear Put Spread": {"legs": [{"type": "PUT", "action": "BUY", "offset": 0}, {"type": "PUT", "action": "SELL", "offset": -100}], "cat": "Bearish"},
}

# ============================================================================
# METRICS CALCULATION WITH ERROR HANDLING
# ============================================================================

def calculate_metrics(df: pd.DataFrame, capital: float = 100000) -> Dict:
    """Calculate metrics with comprehensive error handling"""
    default_metrics = {
        'pnl': 0, 'trades': 0, 'winners': 0, 'losers': 0, 'win_rate': 0,
        'avg_win': 0, 'avg_loss': 0, 'avg_trade': 0, 'best': 0, 'worst': 0,
        'pf': 0, 'max_dd': 0, 'max_dd_pct': 0, 'sharpe': 0, 'sortino': 0,
        'return_pct': 0, 'exit_reasons': {}, 'max_win_streak': 0, 'max_loss_streak': 0
    }
    
    try:
        capital = max(safe_float(capital, 100000), 1.0)  # Prevent zero division
        if df is None or len(df) == 0 or 'pnl' not in df.columns:
            return default_metrics
        
        # Filter out any non-numeric pnl values
        df = df.copy()
        df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0)
        
        n = len(df)
        if n == 0:
            return default_metrics
        
        w = df[df['pnl'] > 0]
        l = df[df['pnl'] < 0]
        
        pnl = safe_float(df['pnl'].sum())
        wr = safe_divide(len(w), n) * 100
        
        # Cumulative and drawdown
        try:
            cum = df['pnl'].cumsum()
            dd = cum - cum.cummax()
            max_dd = abs(safe_float(dd.min()))
        except:
            max_dd = 0
        
        # Sharpe and Sortino
        try:
            ret = df['pnl'] / capital
            std = ret.std()
            sharpe = safe_divide(ret.mean(), std) * np.sqrt(252) if std and std > 0 else 0
            neg_ret = ret[ret < 0]
            neg_std = neg_ret.std() if len(neg_ret) > 0 else 0
            sortino = safe_divide(ret.mean(), neg_std) * np.sqrt(252) if neg_std and neg_std > 0 else 0
        except:
            sharpe = sortino = 0
        
        # Profit Factor
        try:
            total_wins = safe_float(w['pnl'].sum())
            total_losses = abs(safe_float(l['pnl'].sum()))
            pf = safe_divide(total_wins, total_losses, float('inf'))
        except:
            pf = 0
        
        # Exit reasons
        try:
            exit_reasons = df['exit_reason'].value_counts().to_dict() if 'exit_reason' in df.columns else {}
        except:
            exit_reasons = {}
        
        # Streaks
        try:
            max_win_streak = max_loss_streak = current = 0
            last = None
            for p in df['pnl']:
                curr = 'w' if p > 0 else 'l'
                current = current + 1 if curr == last else 1
                if curr == 'w':
                    max_win_streak = max(max_win_streak, current)
                else:
                    max_loss_streak = max(max_loss_streak, current)
                last = curr
        except:
            max_win_streak = max_loss_streak = 0
        
        return {
            'pnl': pnl,
            'trades': n,
            'winners': len(w),
            'losers': len(l),
            'win_rate': wr,
            'avg_win': safe_float(w['pnl'].mean()) if len(w) > 0 else 0,
            'avg_loss': abs(safe_float(l['pnl'].mean())) if len(l) > 0 else 0,
            'avg_trade': safe_divide(pnl, n),
            'best': safe_float(df['pnl'].max()),
            'worst': safe_float(df['pnl'].min()),
            'pf': pf,
            'max_dd': max_dd,
            'max_dd_pct': safe_divide(max_dd, capital) * 100,
            'sharpe': sharpe,
            'sortino': sortino,
            'return_pct': safe_divide(pnl, capital) * 100,
            'exit_reasons': exit_reasons,
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak
        }
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return default_metrics

# ============================================================================
# BACKTEST ENGINE WITH ERROR HANDLING
# ============================================================================

def run_backtest_v3(conn, strategy_name: str, config: Dict, progress_cb=None) -> pd.DataFrame:
    """Robust backtest engine"""
    
    if conn is None:
        st.error("Database connection not available")
        return pd.DataFrame()
    
    # Validate config
    valid, error = validate_config(config)
    if not valid:
        st.error(f"Invalid configuration: {error}")
        return pd.DataFrame()
    
    try:
        # Handle custom strategy
        if strategy_name == "🛠️ Custom Strategy":
            if 'custom_legs' not in config or not config['custom_legs']:
                st.error("Custom strategy has no legs configured")
                return pd.DataFrame()
            legs = config['custom_legs']
        else:
            strategy = STRATEGIES.get(strategy_name)
            if not strategy:
                st.error(f"Unknown strategy: {strategy_name}")
                return pd.DataFrame()
            legs = strategy['legs']
        
        days = get_trading_days(conn, config['start_date'], config['end_date'])
        
        if not days:
            st.warning("No trading days found in selected range")
            return pd.DataFrame()
        
        results = []
        daily_pnl = 0
        current_day = None
        consecutive_losses = 0
        total = len(days)
        errors_count = 0
        max_errors = total // 10 + 5  # Allow ~10% error rate
        
        for i, date in enumerate(days):
            if current_day != date:
                daily_pnl = 0
                current_day = date
            if progress_cb:
                try:
                    progress_cb((i + 1) / total)
                except:
                    pass
            
            try:
                # Max daily loss check
                if config.get('max_daily_loss_enabled'):
                    max_loss = max(safe_float(config.get('max_daily_loss', 5000), 5000), 0.0)
                    if abs(daily_pnl) >= max_loss:
                        results.append({'date': date, 'pnl': 0, 'exit_reason': 'SKIPPED', 'skip_reason': 'Max daily loss'})
                        continue
                
                # Consecutive loss check
                if config.get('consec_loss_enabled'):
                    max_consec = max(safe_int(config.get('max_consec_losses', 3), 3), 1)
                    if consecutive_losses >= max_consec:
                        results.append({'date': date, 'pnl': 0, 'exit_reason': 'SKIPPED', 'skip_reason': 'Consec loss limit'})
                        continue
                
                # Previous day data
                prev_day = get_prev_day_data(conn, date)
                
                # Entry conditions
                can_enter, reason = check_entry_conditions(conn, date, config, prev_day)
                if not can_enter:
                    results.append({'date': date, 'pnl': 0, 'exit_reason': 'SKIPPED', 'skip_reason': reason})
                    continue
                
                # Get spot
                entry_time_s = _safe_time_str(config.get('entry_time'), '09:20:00')
                exit_time_s = _safe_time_str(config.get('exit_time'), '15:15:00')
                spot = get_spot(conn, date, entry_time_s)
                if not spot or spot <= 0:
                    continue
                
                atm = atm_strike(spot, config.get('strike_step', 50))
                expiry = get_expiry(conn, date, config.get('expiry_type', 'weekly'))
                if not expiry:
                    continue
                
                # Execute legs
                leg_data = []
                skip = False
                entry_premium = 0
                user_strike_off = _normalize_strike_offset(config.get('strike_offset', 0), config.get('strike_step', 50))
                
                for leg in legs:
                    base_off = safe_int(leg.get('offset', 0), 0)
                    offset = base_off + (user_strike_off if leg.get('type') == 'CALL' else -user_strike_off)
                    strike = safe_int(atm + offset, 0)
                    if strike <= 0:
                        skip = True
                        break
                    
                    price, _ = get_option(conn, date, entry_time_s, strike, leg.get('type'), expiry)
                    if price is None or price <= 0:
                        skip = True
                        break
                    
                    slip = config.get('slippage', 0.05) / 100
                    price = price * (1 + slip) if leg.get('action') == 'BUY' else price * (1 - slip)
                    
                    leg_data.append({'strike': strike, 'type': leg.get('type'), 'action': leg.get('action'), 'entry': price})
                    entry_premium += price * (1 if leg.get('action') == 'SELL' else -1)
                
                if skip:
                    continue
                
                # Intraday monitoring
                exit_reason = 'TIME'
                exit_time = exit_time_s
                peak_profit = 0
                sl_level = -config.get('sl_pct', 50) if config.get('sl_enabled') else -100
                
                if config.get('sl_enabled') or config.get('target_enabled') or config.get('trail_enabled'):
                    try:
                        all_prices = {}
                        for leg in leg_data:
                            prices = get_intraday_options(conn, date, entry_time_s, exit_time_s, leg['strike'], leg['type'], expiry)
                            if len(prices) > 0 and 'datetime' in prices.columns and 'close' in prices.columns:
                                prices = prices.copy()
                                prices['datetime'] = pd.to_datetime(prices['datetime'], errors='coerce')
                                prices = prices.dropna(subset=['datetime'])
                                if len(prices) > 0:
                                    prices['close'] = pd.to_numeric(prices['close'], errors='coerce')
                                    all_prices[f"{leg['strike']}_{leg['type']}"] = prices.set_index('datetime')['close']
                        
                        if all_prices:
                            combined = pd.DataFrame(all_prices)
                            
                            for ts, row in combined.iterrows():
                                current_pnl = 0
                                for leg in leg_data:
                                    key = f"{leg['strike']}_{leg['type']}"
                                    val = row.get(key, np.nan)
                                    if pd.notna(val):
                                        curr_price = safe_float(val, 0)
                                        if curr_price > 0:
                                            leg_pnl = (leg['entry'] - curr_price) if leg['action'] == 'SELL' else (curr_price - leg['entry'])
                                            current_pnl += leg_pnl
                                
                                denom = abs(entry_premium)
                                pnl_pct = safe_divide(current_pnl, denom) * 100 if denom and denom > 1e-9 else 0
                                
                                if pnl_pct > peak_profit:
                                    peak_profit = pnl_pct
                                
                                # Trailing SL
                                if config.get('trail_enabled') and peak_profit >= config.get('trail_trigger', 30):
                                    sl_level = max(sl_level, config.get('trail_lock', 0))
                                
                                # Time-based SL
                                if config.get('time_sl_enabled') and hasattr(ts, 'time'):
                                    try:
                                        tighten = dt_time(int(config.get('tighten_hour', 14)), int(config.get('tighten_min', 0)))
                                        if ts.time() >= tighten:
                                            sl_level = max(sl_level, -config.get('sl_pct', 50) / 2)
                                    except:
                                        pass
                                
                                # Check SL
                                if config.get('sl_enabled') and pnl_pct < sl_level:
                                    exit_reason = 'SL HIT'
                                    exit_time = _safe_time_str(ts.time() if hasattr(ts, 'time') else ts, exit_time_s)
                                    break
                                
                                # Check target
                                if config.get('target_enabled') and pnl_pct >= config.get('target_pct', 50):
                                    exit_reason = 'TARGET'
                                    exit_time = _safe_time_str(ts.time() if hasattr(ts, 'time') else ts, exit_time_s)
                                    break
                    except Exception as e:
                        logger.error(f"Intraday monitoring error: {e}")
                
                # Get exit prices
                final_pnl = 0
                missing_exit = False
                for leg in leg_data:
                    exit_price, _ = get_option(conn, date, exit_time, leg['strike'], leg['type'], expiry)
                    if exit_price is None or exit_price <= 0:
                        exit_price, _ = get_option(conn, date, exit_time_s, leg['strike'], leg['type'], expiry)
                    if exit_price is None or exit_price <= 0:
                        missing_exit = True
                        break
                    
                    slip = config.get('slippage', 0.05) / 100
                    exit_price = exit_price * (1 - slip) if leg['action'] == 'BUY' else exit_price * (1 + slip)
                    leg['exit'] = exit_price
                    leg_pnl = (leg['entry'] - exit_price) if leg['action'] == 'SELL' else (exit_price - leg['entry'])
                    final_pnl += leg_pnl
                
                if missing_exit:
                    results.append({'date': date, 'pnl': 0, 'exit_reason': 'SKIPPED', 'skip_reason': 'Missing exit price'})
                    continue
                
                lot_size = max(safe_int(config.get('lot_size', 25), 25), 1) * max(safe_int(config.get('num_lots', 1), 1), 1)
                gross = final_pnl * lot_size
                brok = safe_float(config.get('brokerage', 20), 20) * len(legs) * 2
                net = gross - brok
                
                daily_pnl += net
                consecutive_losses = consecutive_losses + 1 if net < 0 else 0
                
                results.append({
                    'date': date, 'expiry': expiry, 'spot': spot, 'atm': atm,
                    'entry_premium': entry_premium, 'gross_pnl': gross, 'brokerage': brok, 'pnl': net,
                    'exit_reason': exit_reason, 'exit_time': exit_time, 'legs': leg_data, 'skip_reason': None
                })
                
            except Exception as e:
                errors_count += 1
                logger.error(f"Error on {date}: {e}")
                if errors_count > max_errors:
                    st.warning(f"Too many errors ({errors_count}). Stopping backtest.")
                    break
                continue
        
        if errors_count > 0:
            st.info(f"ℹ️ {errors_count} days skipped due to data issues")
        
        return pd.DataFrame(results)
        
    except Exception as e:
        st.error(f"Backtest failed: {str(e)}")
        logger.error(f"Backtest error: {traceback.format_exc()}")
        return pd.DataFrame()

def monte_carlo(pnl_series: pd.Series, n: int = 1000) -> pd.DataFrame:
    """Monte Carlo with error handling"""
    try:
        if len(pnl_series) == 0:
            return pd.DataFrame({'final_pnl': [0], 'max_dd': [0]})
        
        results = []
        values = pnl_series.values
        for _ in range(n):
            shuffled = np.random.permutation(values)
            cumsum = np.cumsum(shuffled)
            results.append({
                'final_pnl': cumsum[-1],
                'max_dd': np.min(cumsum - np.maximum.accumulate(cumsum))
            })
        return pd.DataFrame(results)
    except Exception as e:
        logger.error(f"Monte Carlo error: {e}")
        return pd.DataFrame({'final_pnl': [0], 'max_dd': [0]})


def generate_backtest_pdf(
    strategy_name: str,
    config: Dict,
    m: Dict,
    traded: pd.DataFrame,
    capital: float,
    equity_fig=None
) -> Optional[bytes]:
    """Generate PDF report. Returns bytes or None if reportlab not available."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        import io
    except ImportError:
        return None
    
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            name='CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            alignment=TA_CENTER,
            spaceAfter=12
        )
        story.append(Paragraph("NIFTY Options Backtest Report", title_style))
        story.append(Paragraph(f"Strategy: {strategy_name}", styles['Normal']))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Config summary
        story.append(Paragraph("Configuration", styles['Heading2']))
        config_data = [
            ['Parameter', 'Value'],
            ['Date Range', f"{config.get('start_date', '')} to {config.get('end_date', '')}"],
            ['Entry / Exit', f"{config.get('entry_time', '')} / {config.get('exit_time', '')}"],
            ['Expiry', config.get('expiry_type', 'weekly')],
            ['Lot Size', str(config.get('lot_size', 25) * config.get('num_lots', 1))],
            ['SL / Target', f"{config.get('sl_pct', 50)}% / {config.get('target_pct', 50)}%"],
            ['Capital', f"₹{capital:,.0f}"],
        ]
        t = Table(config_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
        ]))
        story.append(t)
        story.append(Spacer(1, 20))
        
        # Metrics
        story.append(Paragraph("Performance Metrics", styles['Heading2']))
        metrics_data = [
            ['Metric', 'Value'],
            ['Total P&L', safe_format(m.get('pnl', 0), ",.0f", "₹")],
            ['Trades', str(m.get('trades', 0))],
            ['Win Rate', safe_format(m.get('win_rate', 0), ".1f", "", "%")],
            ['Avg Trade', safe_format(m.get('avg_trade', 0), ",.0f", "₹")],
            ['Max Drawdown', safe_format(m.get('max_dd', 0), ",.0f", "₹")],
            ['Max DD %', safe_format(m.get('max_dd_pct', 0), ".1f", "", "%")],
            ['Sharpe', safe_format(m.get('sharpe', 0), ".2f")],
            ['Profit Factor', safe_format(m.get('pf', 0), ".2f")],
        ]
        t2 = Table(metrics_data)
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
        ]))
        story.append(t2)
        story.append(Spacer(1, 20))
        
        # Equity chart (if plotly + kaleido available)
        if equity_fig is not None:
            try:
                img_bytes = equity_fig.to_image(format="png", width=600, height=400)
                img = Image(io.BytesIO(img_bytes), width=6*inch, height=4*inch)
                story.append(PageBreak())
                story.append(Paragraph("Equity Curve", styles['Heading2']))
                story.append(img)
                story.append(Spacer(1, 20))
            except Exception:
                pass
        
        # Trade list (first 50 rows)
        story.append(Paragraph("Trade List (first 50)", styles['Heading2']))
        cols = [c for c in ['date', 'spot', 'atm', 'exit_reason', 'pnl'] if c in traded.columns]
        if not cols:
            cols = list(traded.columns)[:5]
        trade_df = traded[cols].head(50)
        trade_data = [list(trade_df.columns)] + [[str(v) for v in row] for row in trade_df.values.tolist()]
        if len(trade_data) > 1:
            t3 = Table(trade_data, repeatRows=1)
            t3.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
            ]))
            story.append(t3)
        
        doc.build(story)
        return buffer.getvalue()
    except Exception as e:
        logger.error(f"PDF generation error: {e}")
        return None

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Version Selector in Sidebar
    with st.sidebar:
        st.markdown("## 📦 Version")
        versions = {
            "V3.4 (Current)": {"file": "nifty_options_backtest_v3.py", "desc": "Airtight, Bulletproof, All Features", "current": True},
            "V2 (Legacy)": {"file": "nifty_options_backtest_v2.py", "desc": "Previous stable version", "current": False},
            "V1 (Original)": {"file": "nifty_options_backtest.py", "desc": "Original basic version", "current": False}
        }
        selected_version = st.selectbox(
            "Select Version",
            list(versions.keys()),
            index=0,
            help="Switch between different versions of the backtester"
        )
        
        if not versions[selected_version]["current"]:
            st.warning(f"⚠️ You selected **{selected_version}**")
            st.caption(versions[selected_version]["desc"])
            
            # Launch button for other versions
            if st.button(f"🚀 Launch {selected_version}", use_container_width=True):
                import subprocess
                import sys
                target_file = versions[selected_version]["file"]
                try:
                    # Launch the selected version in a new process
                    subprocess.Popen([
                        sys.executable, "-m", "streamlit", "run", target_file,
                        "--server.port", "8502"
                    ], cwd=os.path.dirname(__file__))
                    st.success(f"✅ Launching {selected_version} on port 8502...")
                    st.info("🔗 Open http://localhost:8502 in your browser")
                except Exception as e:
                    st.error(f"Failed to launch: {e}")
            
            st.markdown("---")
        else:
            st.success("✅ Running latest version")
            st.caption(versions[selected_version]["desc"])
            st.markdown("---")
    
    st.markdown("# 🚀 NIFTY BACKTESTER V3.4")
    st.markdown("""<p style='text-align:center;color:#a0a0c0;'>
    <span class="feature-badge">AIRTIGHT</span>
    <span class="feature-badge">BULLETPROOF</span>
    <span class="feature-badge">ZERO CRASHES</span>
    <span class="feature-badge">ALL FEATURES</span>
    </p>""", unsafe_allow_html=True)
    
    # Database connection
    conn = get_db()
    if conn is None:
        return
    
    min_date, max_date, n_days = get_data_info(conn)
    if min_date is None or max_date is None:
        st.error("❌ No data found in database")
        return

    min_date_ui = _safe_to_date(min_date)
    max_date_ui = _safe_to_date(max_date)
    if min_date_ui is None or max_date_ui is None:
        st.error("❌ Invalid date values found in database")
        return
    
    try:
        min_str = pd.Timestamp(min_date).strftime('%b %Y') if min_date else "N/A"
        max_str = pd.Timestamp(max_date).strftime('%b %Y') if max_date else "N/A"
        st.info(f"📊 Data: {min_str} to {max_str} | {safe_int(n_days)} days")
    except Exception:
        st.info(f"📊 Data loaded | {safe_int(n_days)} days")
    
    # Mode selection
    mode = st.radio("Mode", ["📊 Backtest", "🔄 Compare", "🎮 Simulator"], horizontal=True)
    
    if mode == "📊 Backtest":
        with st.sidebar:
            st.markdown("## ⚙️ Config")
            
            # Strategy Selection with Custom Option
            strategy_options = list(STRATEGIES.keys()) + ["🛠️ Custom Strategy"]
            strategy_name = st.selectbox("Strategy", strategy_options)
            
            # Custom Strategy Builder
            if strategy_name == "🛠️ Custom Strategy":
                try:
                    st.markdown('<div class="section-header">🛠️ Custom Legs</div>', unsafe_allow_html=True)
                    
                    # Initialize session state for custom legs with safe defaults
                    if 'custom_legs' not in st.session_state:
                        st.session_state.custom_legs = [
                            {"type": "CALL", "action": "SELL", "offset": 0},
                            {"type": "PUT", "action": "SELL", "offset": 0}
                        ]
                    
                    # Validate existing legs structure
                    if not isinstance(st.session_state.custom_legs, list):
                        st.session_state.custom_legs = [{"type": "CALL", "action": "SELL", "offset": 0}]
                    
                    # Add/Remove leg buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        add_leg = st.button("➕ Add Leg", use_container_width=True, key="add_leg_btn")
                    with col2:
                        remove_leg = st.button("➖ Remove Leg", use_container_width=True, key="remove_leg_btn")
                    
                    # Handle add/remove (without immediate rerun)
                    if add_leg and len(st.session_state.custom_legs) < 6:
                        st.session_state.custom_legs.append({"type": "CALL", "action": "BUY", "offset": 100})
                    if remove_leg and len(st.session_state.custom_legs) > 1:
                        st.session_state.custom_legs.pop()
                    
                    st.caption(f"📊 {len(st.session_state.custom_legs)} legs configured")
                    
                    # Display each leg with configuration
                    custom_legs = []
                    for i, leg in enumerate(st.session_state.custom_legs):
                        # Ensure leg has valid structure
                        leg_type_default = leg.get("type", "CALL") if isinstance(leg, dict) else "CALL"
                        leg_action_default = leg.get("action", "SELL") if isinstance(leg, dict) else "SELL"
                        leg_offset_default = leg.get("offset", 0) if isinstance(leg, dict) else 0
                        
                        with st.expander(f"Leg {i+1}", expanded=True):
                            c1, c2, c3 = st.columns(3)
                            leg_type = c1.selectbox("Type", ["CALL", "PUT"], 
                                                   index=0 if leg_type_default == "CALL" else 1, 
                                                   key=f"leg_type_{i}")
                            leg_action = c2.selectbox("Action", ["BUY", "SELL"], 
                                                      index=0 if leg_action_default == "BUY" else 1,
                                                      key=f"leg_action_{i}")
                            leg_offset = c3.number_input("Offset", value=int(leg_offset_default), step=50, key=f"leg_offset_{i}")
                            
                            custom_legs.append({
                                "type": leg_type,
                                "action": leg_action,
                                "offset": leg_offset
                            })
                    
                    # Update session state
                    st.session_state.custom_legs = custom_legs
                    
                    # Show leg summary
                    st.markdown("**Leg Summary:**")
                    for i, leg in enumerate(custom_legs):
                        action_emoji = "📈" if leg["action"] == "BUY" else "📉"
                        offset_str = f"+{leg['offset']}" if leg['offset'] >= 0 else str(leg['offset'])
                        st.caption(f"{action_emoji} {leg['action']} {leg['type']} @ ATM{offset_str}")
                except Exception as e:
                    st.error(f"Error in custom strategy builder: {e}")
                    st.session_state.custom_legs = [{"type": "CALL", "action": "SELL", "offset": 0}]
            
            st.markdown('<div class="section-header">📅 Dates</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            start_date = c1.date_input("Start", min_date_ui, min_value=min_date_ui, max_value=max_date_ui)
            end_date = c2.date_input("End", max_date_ui, min_value=min_date_ui, max_value=max_date_ui)
            
            st.markdown('<div class="section-header">⏰ Times</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            entry_time = c1.time_input("Entry", dt_time(9, 20))
            exit_time = c2.time_input("Exit", dt_time(15, 15))
            
            st.markdown('<div class="section-header">🎯 Options</div>', unsafe_allow_html=True)
            expiry_type = st.selectbox("Expiry", ["weekly", "next_week", "monthly"])
            strike_offset = st.number_input("Strike Offset", 0, step=50)
            
            st.markdown('<div class="section-header">🚦 Filters</div>', unsafe_allow_html=True)
            gap_filter = st.checkbox("Gap Filter")
            gap_dir = st.selectbox("Gap Dir", ["any", "up", "down"]) if gap_filter else "any"
            min_gap = st.number_input("Min Gap%", 0.5) if gap_filter else 0.5
            
            dow_filter = st.checkbox("Day Filter")
            days = st.multiselect("Days", ['Monday','Tuesday','Wednesday','Thursday','Friday'], 
                                  default=['Monday','Tuesday','Wednesday','Thursday','Friday']) if dow_filter else []
            
            st.markdown('<div class="section-header">🛡️ Risk</div>', unsafe_allow_html=True)
            sl_enabled = st.checkbox("Stop Loss", True)
            sl_pct = st.slider("SL%", 10, 100, 50) if sl_enabled else 50
            target_enabled = st.checkbox("Target", True)
            target_pct = st.slider("Target%", 10, 100, 50) if target_enabled else 50
            trail_enabled = st.checkbox("Trailing SL")
            trail_trigger = st.number_input("Trail Trigger%", 20) if trail_enabled else 20
            trail_lock = st.number_input("Lock%", 0) if trail_enabled else 0
            
            st.markdown('<div class="section-header">📦 Position</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            lot_size = c1.number_input("Lot", value=25, min_value=1)
            num_lots = c2.number_input("Lots", value=1, min_value=1)
            brokerage = st.number_input("Brokerage", value=20, min_value=0)
            slippage = st.slider("Slippage%", 0.0, 1.0, 0.05)
            capital = st.number_input("Capital", value=100000, min_value=10000, step=10000)
            
            run_btn = st.button("🚀 RUN BACKTEST", use_container_width=True)
        
        config = {
            'strike_step': 50,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'entry_time': entry_time.strftime('%H:%M:%S'),
            'exit_time': exit_time.strftime('%H:%M:%S'),
            'expiry_type': expiry_type,
            'strike_offset': _normalize_strike_offset(strike_offset, 50),
            'gap_filter_enabled': gap_filter,
            'gap_direction': gap_dir,
            'min_gap_pct': min_gap,
            'dow_filter_enabled': dow_filter,
            'allowed_days': days if dow_filter else ['Monday','Tuesday','Wednesday','Thursday','Friday'],
            'sl_enabled': sl_enabled,
            'sl_pct': sl_pct,
            'target_enabled': target_enabled,
            'target_pct': target_pct,
            'trail_enabled': trail_enabled,
            'trail_trigger': trail_trigger,
            'trail_lock': trail_lock,
            'lot_size': lot_size,
            'num_lots': num_lots,
            'brokerage': brokerage,
            'slippage': slippage
        }
        
        # Add custom legs if Custom Strategy is selected
        if strategy_name == "🛠️ Custom Strategy":
            config['custom_legs'] = st.session_state.get('custom_legs', [])
        
        if run_btn:
            # Run exhaustive validation and show warnings
            valid, errors, warnings = validate_config_exhaustive(config)
            
            if not valid:
                for err in errors:
                    st.error(f"❌ {err}")
                return
            
            # Show warnings but continue
            for warn in warnings:
                st.warning(f"⚠️ {warn}")
            
            prog = st.progress(0)
            stat = st.empty()
            
            def update(p):
                try:
                    p_val = min(max(safe_float(p, 0), 0), 1.0)
                    prog.progress(p_val)
                    stat.text(f"Processing... {int(p_val * 100)}%")
                except Exception:
                    pass
            
            df = run_backtest_v3(conn, strategy_name, config, update)
            prog.empty()
            stat.empty()
            
            if df is None or len(df) == 0:
                st.warning("No trades executed - check your filters and date range")
                return
            
            traded = df[df['exit_reason'] != 'SKIPPED']
            if len(traded) == 0:
                st.warning("All trades were skipped due to filters")
                # Show skip reasons
                if 'skip_reason' in df.columns:
                    skip_reasons = df['skip_reason'].value_counts().head(5)
                    st.info(f"Top skip reasons: {dict(skip_reasons)}")
                return
            
            safe_session_set('df', df)
            safe_session_set('metrics', calculate_metrics(traded, capital))
            safe_session_set('strategy_name', strategy_name)
            safe_session_set('config', config)
            safe_session_set('capital', capital)
        
        # Display results
        if 'df' in st.session_state and 'metrics' in st.session_state:
            df = st.session_state['df']
            m = st.session_state['metrics']
            
            if not m or m.get('trades', 0) == 0:
                st.warning("No valid trades to display")
                return
            
            traded = df[df['exit_reason'] != 'SKIPPED']
            skipped = df[df['exit_reason'] == 'SKIPPED']
            
            st.markdown("### 📈 Performance")
            cols = st.columns(8)
            items = [
                ('P&L', safe_format(m['pnl'], ",.0f", "₹"), 'profit' if m['pnl'] >= 0 else 'loss'),
                ('Win%', safe_format(m['win_rate'], ".0f", "", "%"), 'neutral'),
                ('Sharpe', safe_format(m['sharpe'], ".2f"), 'profit' if m['sharpe'] > 0 else 'loss'),
                ('MaxDD', safe_format(m['max_dd_pct'], ".0f", "", "%"), 'loss'),
                ('PF', safe_format(m['pf'], ".1f"), 'profit' if m['pf'] >= 1 else 'loss'),
                ('Avg', safe_format(m['avg_trade'], ",.0f", "₹"), 'profit' if m['avg_trade'] >= 0 else 'loss'),
                ('Trades', str(m['trades']), 'neutral'),
                ('Skip', str(len(skipped)), 'neutral')
            ]
            for col, (l, v, s) in zip(cols, items):
                col.markdown(f'<div class="metric-card"><p class="metric-value {s}">{v}</p><p class="metric-label">{l}</p></div>', unsafe_allow_html=True)
            
            # Exit Analysis
            if m.get('exit_reasons'):
                st.markdown("### 🚪 Exits")
                cols = st.columns(len(m['exit_reasons']))
                colors = {'TARGET': '#10b981', 'SL HIT': '#ef4444', 'TIME': '#7c3aed'}
                for col, (r, c) in zip(cols, m['exit_reasons'].items()):
                    pct = safe_divide(c, m['trades']) * 100
                    col.markdown(f'<div class="metric-card"><p class="metric-value" style="color:{colors.get(r,"#fff")}">{c}</p><p class="metric-label">{r} ({pct:.0f}%)</p></div>', unsafe_allow_html=True)
            
            # Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Equity", "📋 Trades", "📉 Risk", "🎲 Monte Carlo"])
            
            with tab1:
                try:
                    traded_copy = traded.copy()
                    if len(traded_copy) == 0:
                        st.info("No trade data to display")
                    elif 'pnl' not in traded_copy.columns or 'date' not in traded_copy.columns:
                        st.warning("Required data missing from results (need date and pnl)")
                    else:
                        traded_copy['date'] = pd.to_datetime(traded_copy['date'], errors='coerce')
                        traded_copy['pnl'] = pd.to_numeric(traded_copy['pnl'], errors='coerce')
                        traded_copy = traded_copy.dropna(subset=['date', 'pnl'])
                        if len(traded_copy) == 0:
                            st.info("No valid date/pnl rows to chart")
                            raise Exception("No valid data")
                        traded_copy['cum'] = traded_copy['pnl'].cumsum()
                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                        fig.add_trace(go.Scatter(x=traded_copy['date'], y=traded_copy['cum'], fill='tozeroy', line=dict(color='#00d4ff'), name='Cumulative P&L'), row=1, col=1)
                        dd = traded_copy['cum'] - traded_copy['cum'].cummax()
                        fig.add_trace(go.Scatter(x=traded_copy['date'], y=dd, fill='tozeroy', line=dict(color='#ef4444'), name='Drawdown'), row=2, col=1)
                        fig.update_layout(height=500, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                        fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
                        fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    logger.error(f"Equity chart error: {traceback.format_exc()}")
                    st.warning(f"⚠️ Chart display failed: {str(e)}")
                    # Fallback: show data table instead
                    st.markdown("**📊 Fallback: Data Table**")
                    try:
                        fallback_df = traded[['date', 'pnl']].copy() if 'date' in traded.columns and 'pnl' in traded.columns else pd.DataFrame()
                        if len(fallback_df) > 0:
                            fallback_df['date'] = pd.to_datetime(fallback_df['date'], errors='coerce')
                            fallback_df['pnl'] = pd.to_numeric(fallback_df['pnl'], errors='coerce')
                            fallback_df = fallback_df.dropna(subset=['date', 'pnl'])
                            fallback_df['cum_pnl'] = fallback_df['pnl'].cumsum()
                            st.dataframe(fallback_df.head(20), use_container_width=True)
                        else:
                            st.info("No data available for fallback display")
                    except Exception:
                        st.info("No data available for fallback display")
            
            with tab2:
                try:
                    required_cols = ['date', 'spot', 'atm', 'exit_reason', 'pnl']
                    available = [c for c in required_cols if c in traded.columns]
                    if available:
                        display = traded[available].copy()
                        st.dataframe(display, use_container_width=True, height=400)
                    else:
                        st.dataframe(traded, use_container_width=True, height=400)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.download_button("📥 CSV", traded.to_csv(index=False), f"backtest_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv", key="csv_dl")
                    with c2:
                        # PDF export
                        strategy_name = safe_session_get('strategy_name', '')
                        config = safe_session_get('config', {})
                        capital = safe_session_get('capital', 100000)
                        equity_fig = None
                        try:
                            traded_copy = traded.copy()
                            if 'pnl' in traded_copy.columns and len(traded_copy) > 0:
                                traded_copy['cum'] = traded_copy['pnl'].cumsum()
                                equity_fig = make_subplots(rows=1, cols=1)
                                equity_fig.add_trace(go.Scatter(x=traded_copy['date'], y=traded_copy['cum'], fill='tozeroy', line=dict(color='#00d4ff'), name='Cumulative P&L'))
                                equity_fig.update_layout(paper_bgcolor='white', plot_bgcolor='white', font=dict(color='black'))
                                equity_fig.update_layout(height=400, width=600)
                        except Exception:
                            pass
                        pdf_bytes = generate_backtest_pdf(strategy_name, config, m, traded, capital, equity_fig)
                        if pdf_bytes:
                            st.download_button("📄 Export PDF", pdf_bytes, f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf", mime="application/pdf", key="pdf_dl")
                        else:
                            st.caption("Install reportlab for PDF: pip install reportlab kaleido")
                except Exception as e:
                    logger.error(f"Table error: {e}")
                    st.error(f"Table error: {str(e)}")
            
            with tab3:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### Returns")
                    st.table(pd.DataFrame({
                        'Metric': ['P&L', 'Return%', 'Avg Trade', 'Best', 'Worst'],
                        'Value': [safe_format(m['pnl'], ",.0f", "₹"), safe_format(m['return_pct'], ".1f", "", "%"), 
                                  safe_format(m['avg_trade'], ",.0f", "₹"), safe_format(m['best'], ",.0f", "₹"), 
                                  safe_format(m['worst'], ",.0f", "₹")]
                    }))
                with c2:
                    st.markdown("#### Risk")
                    st.table(pd.DataFrame({
                        'Metric': ['Max DD', 'DD%', 'Sharpe', 'Sortino', 'PF'],
                        'Value': [safe_format(m['max_dd'], ",.0f", "₹"), safe_format(m['max_dd_pct'], ".1f", "", "%"),
                                  safe_format(m['sharpe'], ".2f"), safe_format(m['sortino'], ".2f"), safe_format(m['pf'], ".2f")]
                    }))
            
            with tab4:
                st.markdown("#### Monte Carlo Simulation")
                st.caption("Randomly shuffles trade order to simulate path-dependent outcomes. Shows distribution of possible final P&L and max drawdown.")
                mc_col1, mc_col2 = st.columns([1, 3])
                with mc_col1:
                    n_sims = st.number_input("Simulations", value=1000, min_value=100, max_value=10000, step=500, key="mc_n")
                    run_mc = st.button("Run Monte Carlo", key="mc_run")
                if run_mc and 'pnl' in traded.columns and len(traded) > 0:
                    with st.spinner("Running Monte Carlo..."):
                        mc_df = monte_carlo(traded['pnl'], n=int(n_sims))
                    if len(mc_df) > 0:
                        safe_session_set('mc_results', mc_df)
                        safe_session_set('last_df_len', len(traded))
                mc_df = safe_session_get('mc_results')
                last_len = safe_session_get('last_df_len', -1)
                if mc_df is not None and len(mc_df) > 0 and last_len == len(traded) and len(traded) > 0:
                            # Percentiles
                            p5 = np.percentile(mc_df['final_pnl'], 5)
                            p25 = np.percentile(mc_df['final_pnl'], 25)
                            p50 = np.percentile(mc_df['final_pnl'], 50)
                            p75 = np.percentile(mc_df['final_pnl'], 75)
                            p95 = np.percentile(mc_df['final_pnl'], 95)
                            dd5 = np.percentile(mc_df['max_dd'], 95)  # 95th percentile DD = 5th percentile (worst)
                            dd50 = np.percentile(mc_df['max_dd'], 50)
                            dd95 = np.percentile(mc_df['max_dd'], 5)
                            prob_profit = (mc_df['final_pnl'] > 0).mean() * 100
                            
                            st.markdown("##### Confidence Intervals (Final P&L)")
                            st.table(pd.DataFrame({
                                'Percentile': ['5th', '25th', '50th (Median)', '75th', '95th'],
                                'Final P&L (₹)': [f"{p5:,.0f}", f"{p25:,.0f}", f"{p50:,.0f}", f"{p75:,.0f}", f"{p95:,.0f}"]
                            }))
                            st.markdown("##### Max Drawdown Distribution")
                            st.table(pd.DataFrame({
                                'Percentile': ['5th (Best)', '50th (Median)', '95th (Worst)'],
                                'Max DD (₹)': [f"{dd95:,.0f}", f"{dd50:,.0f}", f"{dd5:,.0f}"]
                            }))
                            st.metric("Probability of Profit", f"{prob_profit:.1f}%")
                            
                            # Histograms
                            fig_mc = make_subplots(rows=1, cols=2, subplot_titles=("Final P&L Distribution", "Max Drawdown Distribution"))
                            fig_mc.add_trace(go.Histogram(x=mc_df['final_pnl'], nbinsx=50, name="P&L", marker_color='#00d4ff'), row=1, col=1)
                            fig_mc.add_trace(go.Histogram(x=mc_df['max_dd'], nbinsx=50, name="Max DD", marker_color='#ef4444'), row=1, col=2)
                            fig_mc.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=400)
                            fig_mc.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
                            fig_mc.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
                            st.plotly_chart(fig_mc, use_container_width=True)
                else:
                    st.info("Click **Run Monte Carlo** to simulate path-dependent outcomes.")
    
    elif mode == "🔄 Compare":
        st.markdown("### 🔄 Strategy Comparison")
        st.caption("Compare predefined strategies or build custom strategies with your own leg configurations.")
        
        # Strategy options including Custom
        strategy_options_compare = list(STRATEGIES.keys()) + ["🛠️ Custom Strategy"]
        
        # Strategy Selection
        st.markdown("#### 📊 Select Strategies")
        c1, c2 = st.columns(2)
        s1 = c1.selectbox("Strategy 1", strategy_options_compare, key='compare_s1')
        s2 = c2.selectbox("Strategy 2", strategy_options_compare, index=1, key='compare_s2')
        
        # Custom Strategy Builder for Strategy 1
        if s1 == "🛠️ Custom Strategy":
            with st.expander("🛠️ Configure Custom Strategy 1", expanded=True):
                # Initialize session state
                if 'compare_custom_legs_1' not in st.session_state:
                    st.session_state.compare_custom_legs_1 = [
                        {"type": "CALL", "action": "SELL", "offset": 0},
                        {"type": "PUT", "action": "SELL", "offset": 0}
                    ]
                
                # Validate structure
                if not isinstance(st.session_state.compare_custom_legs_1, list):
                    st.session_state.compare_custom_legs_1 = [{"type": "CALL", "action": "SELL", "offset": 0}]
                
                # Add/Remove buttons
                btn_c1, btn_c2 = st.columns(2)
                with btn_c1:
                    if st.button("➕ Add Leg", key="add_leg_s1") and len(st.session_state.compare_custom_legs_1) < 6:
                        st.session_state.compare_custom_legs_1.append({"type": "CALL", "action": "BUY", "offset": 100})
                with btn_c2:
                    if st.button("➖ Remove Leg", key="rm_leg_s1") and len(st.session_state.compare_custom_legs_1) > 1:
                        st.session_state.compare_custom_legs_1.pop()
                
                st.caption(f"📊 {len(st.session_state.compare_custom_legs_1)} legs configured")
                
                # Build legs
                custom_legs_1 = []
                for i, leg in enumerate(st.session_state.compare_custom_legs_1):
                    leg_type_def = leg.get("type", "CALL") if isinstance(leg, dict) else "CALL"
                    leg_action_def = leg.get("action", "SELL") if isinstance(leg, dict) else "SELL"
                    leg_offset_def = leg.get("offset", 0) if isinstance(leg, dict) else 0
                    
                    lc1, lc2, lc3 = st.columns(3)
                    leg_type = lc1.selectbox("Type", ["CALL", "PUT"], index=0 if leg_type_def == "CALL" else 1, key=f"s1_leg_type_{i}")
                    leg_action = lc2.selectbox("Action", ["BUY", "SELL"], index=0 if leg_action_def == "BUY" else 1, key=f"s1_leg_action_{i}")
                    leg_offset = lc3.number_input("Offset", value=int(leg_offset_def), step=50, key=f"s1_leg_offset_{i}")
                    custom_legs_1.append({"type": leg_type, "action": leg_action, "offset": leg_offset})
                
                st.session_state.compare_custom_legs_1 = custom_legs_1
                
                # Summary
                st.markdown("**Legs:**")
                for leg in custom_legs_1:
                    emoji = "📈" if leg["action"] == "BUY" else "📉"
                    off_str = f"+{leg['offset']}" if leg['offset'] >= 0 else str(leg['offset'])
                    st.caption(f"{emoji} {leg['action']} {leg['type']} @ ATM{off_str}")
        
        # Custom Strategy Builder for Strategy 2
        if s2 == "🛠️ Custom Strategy":
            with st.expander("🛠️ Configure Custom Strategy 2", expanded=True):
                # Initialize session state
                if 'compare_custom_legs_2' not in st.session_state:
                    st.session_state.compare_custom_legs_2 = [
                        {"type": "CALL", "action": "BUY", "offset": 0},
                        {"type": "PUT", "action": "BUY", "offset": 0}
                    ]
                
                # Validate structure
                if not isinstance(st.session_state.compare_custom_legs_2, list):
                    st.session_state.compare_custom_legs_2 = [{"type": "CALL", "action": "BUY", "offset": 0}]
                
                # Add/Remove buttons
                btn_c1, btn_c2 = st.columns(2)
                with btn_c1:
                    if st.button("➕ Add Leg", key="add_leg_s2") and len(st.session_state.compare_custom_legs_2) < 6:
                        st.session_state.compare_custom_legs_2.append({"type": "PUT", "action": "BUY", "offset": -100})
                with btn_c2:
                    if st.button("➖ Remove Leg", key="rm_leg_s2") and len(st.session_state.compare_custom_legs_2) > 1:
                        st.session_state.compare_custom_legs_2.pop()
                
                st.caption(f"📊 {len(st.session_state.compare_custom_legs_2)} legs configured")
                
                # Build legs
                custom_legs_2 = []
                for i, leg in enumerate(st.session_state.compare_custom_legs_2):
                    leg_type_def = leg.get("type", "CALL") if isinstance(leg, dict) else "CALL"
                    leg_action_def = leg.get("action", "BUY") if isinstance(leg, dict) else "BUY"
                    leg_offset_def = leg.get("offset", 0) if isinstance(leg, dict) else 0
                    
                    lc1, lc2, lc3 = st.columns(3)
                    leg_type = lc1.selectbox("Type", ["CALL", "PUT"], index=0 if leg_type_def == "CALL" else 1, key=f"s2_leg_type_{i}")
                    leg_action = lc2.selectbox("Action", ["BUY", "SELL"], index=0 if leg_action_def == "BUY" else 1, key=f"s2_leg_action_{i}")
                    leg_offset = lc3.number_input("Offset", value=int(leg_offset_def), step=50, key=f"s2_leg_offset_{i}")
                    custom_legs_2.append({"type": leg_type, "action": leg_action, "offset": leg_offset})
                
                st.session_state.compare_custom_legs_2 = custom_legs_2
                
                # Summary
                st.markdown("**Legs:**")
                for leg in custom_legs_2:
                    emoji = "📈" if leg["action"] == "BUY" else "📉"
                    off_str = f"+{leg['offset']}" if leg['offset'] >= 0 else str(leg['offset'])
                    st.caption(f"{emoji} {leg['action']} {leg['type']} @ ATM{off_str}")
        
        st.markdown("---")
        
        # Date Range
        st.markdown("#### 📅 Date Range")
        c1, c2 = st.columns(2)
        compare_start = min_date_ui
        compare_end = max_date_ui
        start = c1.date_input("Start Date", compare_start, min_value=min_date_ui, max_value=max_date_ui, key='compare_start')
        end = c2.date_input("End Date", compare_end, min_value=min_date_ui, max_value=max_date_ui, key='compare_end')
        
        # Comparison Settings Expander
        with st.expander("⚙️ Comparison Settings", expanded=False):
            st.caption("Configure common settings for fair comparison")
            c1, c2 = st.columns(2)
            compare_entry_time = c1.time_input("Entry Time", dt_time(9, 20), key='compare_entry')
            compare_exit_time = c2.time_input("Exit Time", dt_time(15, 15), key='compare_exit')
            
            c1, c2, c3 = st.columns(3)
            compare_expiry = c1.selectbox("Expiry Type", ["weekly", "next_week", "monthly"], key='compare_expiry')
            compare_sl = c2.slider("Stop Loss %", 10, 100, 50, key='compare_sl')
            compare_target = c3.slider("Target %", 10, 100, 50, key='compare_target')
            
            c1, c2 = st.columns(2)
            compare_sl_enabled = c1.checkbox("Enable SL", True, key='compare_sl_en')
            compare_target_enabled = c2.checkbox("Enable Target", True, key='compare_tgt_en')
        
        # Run Comparison
        if st.button("🔄 Run Comparison", use_container_width=True):
            try:
                if start is None or end is None:
                    st.warning("Please select valid start and end dates.")
                elif start > end:
                    st.error("Start date must be before end date.")
                else:
                    # Base config
                    base_config = {
                        'start_date': start.strftime('%Y-%m-%d'), 
                        'end_date': end.strftime('%Y-%m-%d'),
                        'entry_time': compare_entry_time.strftime('%H:%M:%S'), 
                        'exit_time': compare_exit_time.strftime('%H:%M:%S'), 
                        'expiry_type': compare_expiry,
                        'strike_step': 50,
                        'strike_offset': 0,
                        'sl_enabled': compare_sl_enabled, 
                        'sl_pct': compare_sl, 
                        'target_enabled': compare_target_enabled, 
                        'target_pct': compare_target,
                        'trail_enabled': False,
                        'trail_trigger': 20,
                        'trail_lock': 0,
                        'gap_filter_enabled': False,
                        'dow_filter_enabled': False,
                        'allowed_days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                        'lot_size': 25, 
                        'num_lots': 1, 
                        'brokerage': 20, 
                        'slippage': 0.05
                    }
                    
                    results = {}
                    strategies_to_run = [(s1, 'compare_custom_legs_1'), (s2, 'compare_custom_legs_2')]
                    
                    progress = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, (strat, legs_key) in enumerate(strategies_to_run):
                        status_text.text(f"Running backtest for {strat}...")
                        progress.progress((idx) / 2)
                        
                        # Create config copy for this strategy
                        config = base_config.copy()
                        
                        # Add custom legs if this is a custom strategy
                        if strat == "🛠️ Custom Strategy":
                            config['custom_legs'] = st.session_state.get(legs_key, [])
                            display_name = f"Custom {idx + 1}"
                        else:
                            display_name = strat
                        
                        df = run_backtest_v3(conn, strat, config)
                        df = safe_dataframe(df)
                        traded = df[df['exit_reason'] != 'SKIPPED'] if len(df) > 0 and 'exit_reason' in df.columns else df
                        
                        if len(traded) > 0 and 'pnl' in traded.columns:
                            traded = traded.copy()
                            traded['cum'] = traded['pnl'].cumsum()
                            results[display_name] = {'df': traded, 'm': calculate_metrics(traded)}
                    
                    progress.progress(1.0)
                    status_text.empty()
                    progress.empty()
                    
                    if results:
                        st.markdown("### 📊 Comparison Results")
                        
                        # Metrics Table
                        data = []
                        for s, r in results.items():
                            m = r['m']
                            data.append({
                                'Strategy': s, 
                                'P&L': safe_format(m['pnl'], ",.0f", "₹"), 
                                'Win%': safe_format(m['win_rate'], ".0f", "", "%"),
                                'Trades': str(m.get('trades', 0)),
                                'Avg Trade': safe_format(m.get('avg_trade', 0), ",.0f", "₹"),
                                'Max DD': safe_format(m.get('max_dd', 0), ",.0f", "₹"),
                                'Sharpe': safe_format(m['sharpe'], ".2f"),
                                'Profit Factor': safe_format(m.get('pf', 0), ".2f")
                            })
                        st.table(pd.DataFrame(data))
                        
                        # Equity Curves Chart
                        st.markdown("#### 📈 Equity Curves")
                        fig = go.Figure()
                        colors = ['#00d4ff', '#10b981', '#f59e0b', '#ef4444']
                        for idx, (s, r) in enumerate(results.items()):
                            if 'date' in r['df'].columns and 'cum' in r['df'].columns:
                                try:
                                    x = pd.to_datetime(r['df']['date'], errors='coerce')
                                    y = pd.to_numeric(r['df']['cum'], errors='coerce')
                                    mask = x.notna() & y.notna()
                                    fig.add_trace(go.Scatter(
                                        x=x[mask], y=y[mask], 
                                        name=s,
                                        line=dict(color=colors[idx % len(colors)], width=2)
                                    ))
                                except Exception:
                                    fig.add_trace(go.Scatter(x=r['df']['date'], y=r['df']['cum'], name=s))
                        
                        fig.update_layout(
                            height=450, 
                            paper_bgcolor='rgba(0,0,0,0)', 
                            plot_bgcolor='rgba(0,0,0,0)', 
                            font=dict(color='white'),
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                            xaxis_title="Date",
                            yaxis_title="Cumulative P&L (₹)"
                        )
                        fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
                        fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Winner Declaration
                        if len(results) >= 2:
                            pnls = {s: r['m']['pnl'] for s, r in results.items()}
                            winner = max(pnls, key=pnls.get)
                            st.success(f"🏆 **Winner: {winner}** with P&L of {safe_format(pnls[winner], ',.0f', '₹')}")
                    else:
                        st.warning("No valid backtest results to compare. Check your data range and strategy settings.")
            except Exception as e:
                logger.error(f"Compare error: {traceback.format_exc()}")
                st.error(f"Comparison failed: {str(e)}")
    
    elif mode == "🎮 Simulator":
        st.markdown("### 🎮 Trade Simulator - Intraday Replay")
        
        # Controls
        c1, c2, c3, c4 = st.columns(4)
        desired_sim = _safe_to_date(datetime(2025, 12, 15))
        sim_default = desired_sim if desired_sim is not None else min_date_ui
        if sim_default is not None:
            if sim_default < min_date_ui:
                sim_default = min_date_ui
            if sim_default > max_date_ui:
                sim_default = max_date_ui
        sim_date = c1.date_input("Date", sim_default, min_value=min_date_ui, max_value=max_date_ui)
        strategy = c2.selectbox("Strategy", list(STRATEGIES.keys()))
        timeframe = c3.selectbox("Timeframe", ["1 min", "5 min", "15 min", "30 min", "1 hour"], index=0)
        show_premium = c4.checkbox("Show ATM Premium", value=True)
        
        if st.button("▶️ Simulate Day", use_container_width=True):
            try:
                date_str = sim_date.strftime('%Y-%m-%d')
                
                # Get raw 1-minute data
                spot_data = get_intraday_spot(conn, date_str)
                
                if len(spot_data) == 0:
                    st.warning(f"No data for {date_str}")
                else:
                    # Get spot data
                    spot = get_spot(conn, date_str, '09:20:00')
                    
                    if spot and spot > 0:
                        atm = atm_strike(spot)
                        
                        # Get expiry safely
                        expiry = get_expiry(conn, date_str)
                        
                        # Info bar
                        st.markdown(f"""
                        <div style="background: rgba(0,212,255,0.1); border-radius: 8px; padding: 12px; margin-bottom: 16px;">
                            <span style="color: #00d4ff; font-weight: 600;">📅 {date_str}</span> | 
                            <span style="color: #10b981;">Spot: ₹{spot:,.2f}</span> | 
                            <span style="color: #7c3aed;">ATM: {atm}</span> | 
                            <span style="color: #fb923c;">Expiry: {expiry}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Resample data based on timeframe
                        resampled = None
                        spot_data = spot_data.copy()
                        spot_data['datetime'] = pd.to_datetime(spot_data['datetime'], errors='coerce')
                        spot_data = spot_data.dropna(subset=['datetime'])
                        if len(spot_data) == 0:
                            st.warning("No valid datetime data for chart.")
                        else:
                            spot_data = spot_data.set_index('datetime')
                            for col in ['open', 'high', 'low', 'close']:
                                if col not in spot_data.columns:
                                    spot_data[col] = 0
                            
                            tf_map = {"1 min": "1min", "5 min": "5min", "15 min": "15min", "30 min": "30min", "1 hour": "1h"}
                            tf = tf_map.get(timeframe, "1min")
                            
                            if tf != "1min":
                                resampled = spot_data.resample(tf).agg({
                                    'open': 'first',
                                    'high': 'max',
                                    'low': 'min',
                                    'close': 'last'
                                }).dropna()
                            else:
                                resampled = spot_data
                            
                            resampled = resampled.reset_index()
                            if len(resampled) == 0:
                                st.warning("No data after resampling.")
                                resampled = spot_data.reset_index()
                            
                            # Create figure (only when we have resampled data)
                            if show_premium and expiry:
                                # Get ATM Call and Put premiums
                                call_data = get_intraday_options(conn, date_str, '09:15:00', '15:30:00', atm, 'CALL', expiry)
                                put_data = get_intraday_options(conn, date_str, '09:15:00', '15:30:00', atm, 'PUT', expiry)
                                
                                # Create subplots: NIFTY on top, Premium on bottom
                                fig = make_subplots(
                                    rows=2, cols=1,
                                    shared_xaxes=True,
                                    row_heights=[0.6, 0.4],
                                    vertical_spacing=0.08,
                                    subplot_titles=(f"NIFTY {timeframe} Chart", f"ATM {atm} Premium")
                                )
                                
                                # NIFTY Candlestick
                                fig.add_trace(go.Candlestick(
                                    x=resampled['datetime'],
                                    open=resampled['open'],
                                    high=resampled['high'],
                                    low=resampled['low'],
                                    close=resampled['close'],
                                    name="NIFTY",
                                    increasing_line_color='#10b981',
                                    decreasing_line_color='#ef4444'
                                ), row=1, col=1)
                                
                                # ATM Premium - Call
                                if len(call_data) > 0:
                                    call_data['datetime'] = pd.to_datetime(call_data['datetime'])
                                    fig.add_trace(go.Scatter(
                                        x=call_data['datetime'],
                                        y=call_data['close'],
                                        name=f"CE {atm}",
                                        line=dict(color='#00d4ff', width=2)
                                    ), row=2, col=1)
                                
                                # ATM Premium - Put
                                if len(put_data) > 0:
                                    put_data['datetime'] = pd.to_datetime(put_data['datetime'])
                                    fig.add_trace(go.Scatter(
                                        x=put_data['datetime'],
                                        y=put_data['close'],
                                        name=f"PE {atm}",
                                        line=dict(color='#ff006e', width=2)
                                    ), row=2, col=1)
                                
                                # Combined premium (for straddle)
                                if len(call_data) > 0 and len(put_data) > 0:
                                    try:
                                        call_df = call_data.set_index('datetime')['close']
                                        put_df = put_data.set_index('datetime')['close']
                                        combined = call_df + put_df
                                        combined = combined.dropna().reset_index()
                                        combined.columns = ['datetime', 'premium']
                                        
                                        fig.add_trace(go.Scatter(
                                            x=combined['datetime'],
                                            y=combined['premium'],
                                            name="Straddle",
                                            line=dict(color='#7c3aed', width=2, dash='dot')
                                        ), row=2, col=1)
                                    except Exception:
                                        pass
                                
                                fig.update_layout(
                                    height=700,
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color='white'),
                                    xaxis_rangeslider_visible=False,
                                    legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'),
                                    margin=dict(t=60, b=40)
                                )
                            else:
                                # Just NIFTY chart
                                fig = go.Figure(go.Candlestick(
                                    x=resampled['datetime'],
                                    open=resampled['open'],
                                    high=resampled['high'],
                                    low=resampled['low'],
                                    close=resampled['close'],
                                    increasing_line_color='#10b981',
                                    decreasing_line_color='#ef4444'
                                ))
                                
                                fig.update_layout(
                                    height=500,
                                    title=f"NIFTY {timeframe} Chart",
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color='white'),
                                    xaxis_rangeslider_visible=False
                                )
                            
                            # Common styling
                            fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', showgrid=True)
                            fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', showgrid=True)
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Premium stats
                            if show_premium and expiry:
                                st.markdown("### 📊 Premium Statistics")
                                c1, c2, c3, c4 = st.columns(4)
                                
                                if len(call_data) > 0:
                                    ce_open = safe_float(call_data['close'].iloc[0])
                                    ce_close = safe_float(call_data['close'].iloc[-1])
                                    c1.metric("CE Open", f"₹{ce_open:,.1f}")
                                    chg = safe_divide(ce_close - ce_open, ce_open, 0) * 100 if ce_open and ce_open != 0 else 0
                                    c2.metric("CE Close", f"₹{ce_close:,.1f}", f"{chg:+.1f}%")
                                
                                if len(put_data) > 0:
                                    pe_open = safe_float(put_data['close'].iloc[0])
                                    pe_close = safe_float(put_data['close'].iloc[-1])
                                    c3.metric("PE Open", f"₹{pe_open:,.1f}")
                                    chg = safe_divide(pe_close - pe_open, pe_open, 0) * 100 if pe_open and pe_open != 0 else 0
                                    c4.metric("PE Close", f"₹{pe_close:,.1f}", f"{chg:+.1f}%")
                                
                                # Straddle P&L - safe division
                                if len(call_data) > 0 and len(put_data) > 0:
                                    straddle_open = ce_open + pe_open
                                    straddle_close = ce_close + pe_close
                                    straddle_change = straddle_close - straddle_open
                                    chg_pct = safe_divide(straddle_change, straddle_open, 0) * 100 if straddle_open and straddle_open != 0 else 0
                                    
                                    st.markdown(f"""
                                <div style="background: rgba(124,58,237,0.2); border-radius: 8px; padding: 16px; margin-top: 16px; text-align: center;">
                                    <span style="color: #a0a0c0; font-size: 12px;">STRADDLE PREMIUM</span><br/>
                                    <span style="font-size: 24px; font-weight: 700; color: {'#10b981' if straddle_change < 0 else '#ef4444'};">
                                        ₹{straddle_open:,.1f} → ₹{straddle_close:,.1f} ({chg_pct:+.1f}%)
                                    </span><br/>
                                    <span style="color: {'#10b981' if straddle_change < 0 else '#ef4444'}; font-size: 14px;">
                                        {'(Seller Profit: ₹' + f'{-straddle_change*25:,.0f}' + ' per lot)' if straddle_change < 0 else '(Seller Loss: ₹' + f'{-straddle_change*25:,.0f}' + ' per lot)'}
                                    </span>
                                </div>
                                """, unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"Simulation error: {e}")
                logger.error(f"Simulator error: {traceback.format_exc()}")

# ============================================================================
# GLOBAL EXCEPTION HANDLER - Phase 8
# ============================================================================

def show_error_recovery(error_msg: str, error_details: str = None):
    """Display user-friendly error recovery UI"""
    st.markdown("""
    <div style="background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.4); 
                border-radius: 12px; padding: 24px; margin: 20px 0; text-align: center;">
        <h2 style="color: #ef4444; margin-bottom: 16px;">🔴 Application Error</h2>
        <p style="color: #a0a0c0; margin-bottom: 16px;">Something went wrong, but don't worry - your data is safe.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.error(f"**Error:** {error_msg}")
    
    if error_details:
        with st.expander("🔍 Technical Details (for debugging)"):
            st.code(error_details, language="text")
    
    st.markdown("### 🔧 Recovery Options")
    
    col1, col2, col3 = st.columns(3)
    
    def _rerun():
        try:
            st.rerun()
        except AttributeError:
            try:
                st.experimental_rerun()
            except Exception:
                st.info("Please refresh the page manually.")
    
    with col1:
        if st.button("🔄 Restart Application", use_container_width=True):
            clear_session_state()
            _rerun()
    
    with col2:
        if st.button("🗑️ Clear Session", use_container_width=True):
            clear_session_state()
            st.success("Session cleared! Refreshing...")
            _rerun()
    
    with col3:
        if st.button("🔄 Auto-Refresh", use_container_width=True):
            import time
            st.info("Refreshing in 2 seconds...")
            time.sleep(2)
            _rerun()
    
    st.markdown("""
    <div style="background: rgba(0,212,255,0.1); border-radius: 8px; padding: 16px; margin-top: 20px;">
        <h4 style="color: #00d4ff; margin-bottom: 8px;">💡 If the error persists:</h4>
        <ol style="color: #a0a0c0; margin: 0; padding-left: 20px;">
            <li>Close this browser tab</li>
            <li>Stop the Streamlit server (Ctrl+C in terminal)</li>
            <li>Run: <code>streamlit run nifty_options_backtest_v3.py</code></li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

def main_safe():
    """
    Completely safe main function wrapper.
    Guarantees the application never crashes without recovery options.
    """
    try:
        # Pre-flight checks at runtime
        if not check_python_version():
            st.error("❌ Python 3.8 or higher is required")
            st.info("Current version: " + sys.version)
            return
        
        missing = check_dependencies()
        if missing:
            st.error(f"❌ Missing required packages: {', '.join(missing)}")
            st.code(f"pip install {' '.join(missing)}", language="bash")
            return
        
        # Run the main application
        main()
        
    except st.runtime.scriptrunner.StopException:
        # Normal Streamlit stop - don't show error
        pass
    except st.runtime.scriptrunner.RerunException:
        # Normal Streamlit rerun - don't show error
        raise
    except KeyboardInterrupt:
        # User interrupted - don't show error
        pass
    except MemoryError:
        logger.error("Memory error - out of memory")
        show_error_recovery(
            "Out of memory! Try reducing the date range or closing other applications.",
            "MemoryError: The system ran out of memory while processing your request."
        )
    except duckdb.IOException as e:
        logger.error(f"Database I/O error: {e}")
        show_error_recovery(
            "Database access error. The database file may be locked or corrupted.",
            str(e)
        )
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Unhandled error in main: {error_details}")
        show_error_recovery(str(e), error_details)

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main_safe()

