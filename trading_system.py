import streamlit as st
import pandas as pd
import yfinance as yf
import vectorbt as vbt
import plotly.graph_objects as go
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Configure page
st.set_page_config(layout="wide", page_title="Trading System Pro", page_icon="📈")
st.title("📈 Trading System with Buy/Sell Signals")


# ========== Sidebar Controls ==========
st.sidebar.header("Strategy Configuration")
symbol = st.sidebar.text_input("Ticker Symbol", "AAPL").upper()
start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.now())
fast_period = st.sidebar.slider("Fast MA Period", 10, 50, 20)
slow_period = st.sidebar.slider("Slow MA Period", 50, 200, 50)
initial_capital = st.sidebar.number_input("Initial Capital ($)", 1000, 1000000, 10000)

# Ensure start_date is before end_date
if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

# Number of years between start and end date
delta = relativedelta(end_date, start_date)
difference_in_years = delta.years +(delta.months / 12)

# ========== Data Loading ==========
@st.cache_data
def load_data(symbol, start, end):
    return yf.download(symbol, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))

with st.spinner(f"Loading {symbol} data..."):
    data = load_data(symbol, start_date, end_date)

if data.empty:
    st.error("No data found for this symbol!")
    st.stop()

# ========== Signal Calculation ==========
def calculate_signals(data):
    close_price = data['Close']
    
    # Calculate indicators with VectorBT (returns pandas Series)
    fast_ma = vbt.MA.run(close_price, fast_period).ma
    slow_ma = vbt.MA.run(close_price, slow_period).ma
    
    # Create aligned DataFrame using concat
    aligned_data = pd.concat([close_price, fast_ma, slow_ma], axis=1).dropna()
    aligned_data.columns = ['Close', 'fast_ma', 'slow_ma']
    
    # Generate signals using aligned data
    entries = (aligned_data['fast_ma'] > aligned_data['slow_ma']) & \
              (aligned_data['fast_ma'].shift(1) <= aligned_data['slow_ma'].shift(1))
    exits = (aligned_data['fast_ma'] < aligned_data['slow_ma']) & \
            (aligned_data['fast_ma'].shift(1) >= aligned_data['slow_ma'].shift(1))
    
    # Ensure we don't have conflicting signals
    entries = entries & ~exits
    
    return entries, exits, aligned_data['fast_ma'], aligned_data['slow_ma'], aligned_data['Close']

entries, exits, fast_ma, slow_ma, aligned_close = calculate_signals(data)

# ========== Backtesting Engine ==========
try:
    portfolio = vbt.Portfolio.from_signals(
        aligned_close,
        entries,
        exits,
        init_cash=initial_capital,
        fees=0.001,
        freq='D',
        size=1.0,  # Use 100% of available cash for each trade
        size_type='percent',  # Size is a percentage of available cash
        cash_sharing=True,  # Share cash across all assets
        call_seq='auto',  # Automatically determine the call sequence
        group_by=True,  # Group by column (asset)
        broadcast_kwargs=dict(require_kwargs=dict(requirements='W')),  # Ensure proper broadcasting
    )
except Exception as e:
    st.error(f"Error creating portfolio: {str(e)}")
    st.stop()

# ========== Enhanced Header with Signals ==========

# Extract buy and sell signal dates
buy_dates = entries[entries].index if not entries.empty else []
sell_dates = exits[exits].index if not exits.empty else []

# Extract the most recent buy and sell signal dates
most_recent_buy_date = buy_dates[-1] if len(buy_dates) > 0 else "No Buy Signal"
most_recent_sell_date = sell_dates[-1] if len(sell_dates) > 0 else "No Sell Signal"

# Determine the most recent signal type (buy or sell)
if most_recent_buy_date != "No Buy Signal" and (
    most_recent_sell_date == "No Sell Signal" or most_recent_buy_date > most_recent_sell_date
):
    most_recent_signal = f"Buy on {most_recent_buy_date.strftime('%Y-%m-%d')}"
    signal_color = "green"
else:
    most_recent_signal = f"Sell on {most_recent_sell_date.strftime('%Y-%m-%d')}"
    signal_color = "red"

# Add custom CSS for dark background and flashing effect
st.markdown(
    f"""
    <style>
        .custom-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: #333; /* Dark background */
            color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.3);
        }}
        .custom-header h2 {{
            margin: 0;
            font-size: 28px;
            font-weight: bold;
        }}
        .custom-header p {{
            margin: 5px 0;
            font-size: 16px;
        }}
        .signal {{
            font-weight: bold;
            color: {signal_color};
            animation: flash 1.5s infinite alternate; /* Subtle flashing effect */
        }}
        @keyframes flash {{
            from {{ opacity: 1; }}
            to {{ opacity: 0.6; }}
        }}
    </style>
    <div class="custom-header">
        <div>
            <h2>Trading Signals for {symbol}</h2>
            <p>Date Range: <strong>{start_date.strftime('%Y-%m-%d')}</strong> to <strong>{end_date.strftime('%Y-%m-%d')}</strong></p>
        </div>
        <div>
            <p>Most Recent Signal: <span class="signal">{most_recent_signal}</span></p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)



# Proper calculation for total wealth
if portfolio.trades.count() > 0:
    first_trade = portfolio.trades.records.iloc[0]
    # st.write("First trade details:", first_trade)
    # st.write("First trade details:", portfolio.trades.records.head())
else:
    st.write("No trades were executed during the backtest period.")



# Create tabs for main content and debugging
tab1, tab2 = st.tabs(["Main Dashboard", "Debug Info"])

# Main Dashboard Content
with tab1:
    # ========== Visualization ==========
    st.header(f"Trading Signals for {symbol}")

    # Create interactive price chart
    fig = go.Figure()

    # Price and MAs
    fig.add_trace(go.Scatter(x=aligned_close.index, y=aligned_close, name='Price', line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=fast_ma.index, y=fast_ma, name=f'MA {fast_period}', line=dict(color='orange', width=1.5)))
    fig.add_trace(go.Scatter(x=slow_ma.index, y=slow_ma, name=f'MA {slow_period}', line=dict(color='purple', width=1.5)))

    # Buy signals
    buy_dates = entries[entries].index
    buy_prices = aligned_close.loc[buy_dates]
    fig.add_trace(go.Scatter(
        x=buy_dates,
        y=buy_prices,
        mode='markers',
        name='Buy',
        marker=dict(color='green', size=10, symbol='triangle-up')
    ))

    # Sell signals
    sell_dates = exits[exits].index
    sell_prices = aligned_close.loc[sell_dates]
    fig.add_trace(go.Scatter(
        x=sell_dates,
        y=sell_prices,
        mode='markers',
        name='Sell',
        marker=dict(color='red', size=10, symbol='triangle-down')
    ))

    fig.update_layout(
        title=f'{symbol} Price with Trading Signals',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

    # ========== Performance Dashboard ==========
    st.header("Performance Metrics")

    col1, col2, col3 = st.columns(3)
    # If total_return() returns a Series or DataFrame due to grouping/multiple assets:
    if isinstance(portfolio.total_return(), pd.Series) or isinstance(portfolio.total_return(), pd.DataFrame):
        total_return = portfolio.total_return().mean()  # Aggregate if needed
    else:
        total_return = portfolio.total_return()

    col1.metric("Total Return", f"{total_return * 100:.2f}%")

    #col1.metric("Total Return", f"${portfolio.total_return():.2f}")
    col2.metric("Annualized Return", f"{portfolio.annualized_return()*100:.2f}%")
    col3.metric("Max Drawdown", f"{portfolio.max_drawdown()*100:.2f}%")


    # ========== Trade Performance Summary ==========
    # Extract trade returns from the portfolio
    trade_returns = portfolio.trades.records_readable['Return']

    # Calculate metrics using filtered returns
    positive_returns = trade_returns[trade_returns > 0]
    negative_returns = trade_returns[trade_returns < 0]

    max_positive_return = (positive_returns.max() if not positive_returns.empty else 0)*100
    max_negative_return = (negative_returns.min() if not negative_returns.empty else 0)*100  # Changed to min()

    # ========== Buy-and-Hold Comparison ==========
    # st.subheader("💵 Buy-and-Hold vs Strategy Performance")

    # # Create columns for side-by-side comparison
    # col1, col2 = st.columns(2)

    # try:
    #     # Ensure data exists and calculate buy-and-hold returns
    #     if not data.empty:
    #         bh_start_price = data['Close'].iloc[0]  # First closing price in the selected range
    #         bh_end_price = data['Close'].iloc[-1]  # Last closing price in the selected range

    #         # Calculate buy-and-hold return percentage and final portfolio value
    #         bh_return_pct = ((bh_end_price - bh_start_price) / bh_start_price) * 100
    #         bh_final_value = initial_capital * (1 + (bh_return_pct / 100))
    #         print("Start value")
    #         print(bh_start_price)
    #         print(bh_final_value)
    #     else:
    #         raise ValueError("No data available for the selected ticker and date range.")

    #     # Display Buy-and-Hold results
    #     with col1:
    #         st.markdown(f"""
    #         **Buy-and-Hold Strategy**  
    #         - Initial Investment: ${initial_capital:,.2f}  
    #         - Final Value: ${bh_final_value:,.2f}  
    #         - Return Percentage: {bh_return_pct:.2f}%  
    #         """)
    # except Exception as e:
    #     # Handle errors gracefully
    #     with col1:
    #         st.error(f"Could not calculate buy-and-hold returns: {str(e)}")
    #     bh_return_pct = 0  # Default to 0 if calculation fails

    # # Display strategy performance from portfolio results
    # with col2:
    #     try:
    #         strategy_final_value = portfolio.total_profit() + initial_capital
    #         strategy_return_pct = portfolio.total_return() * 100

    #         st.markdown(f"""
    #         **Trading Strategy Results**  
    #         - Portfolio Value: ${strategy_final_value:,.2f}  
    #         - Return Percentage: {strategy_return_pct:.2f}%  
    #         """)
    #     except Exception as e:
    #         st.error(f"Could not calculate strategy performance: {str(e)}")
    #         strategy_return_pct = 0

    # # Add performance comparison summary
    # try:
    #     performance_difference = strategy_return_pct - bh_return_pct
    #     comparison_text = (
    #         f"**The trading strategy outperformed buy-and-hold by {performance_difference:.2f} percentage points.**"
    #         if performance_difference > 0 else
    #         f"**The trading strategy underperformed buy-and-hold by {abs(performance_difference):.2f} percentage points.**"
    #     )
    #     st.markdown(comparison_text)
    # except Exception as e:
    #     st.error(f"Could not compare performances: {str(e)}")


    # ========== Buy-and-Hold Strategy Returns ==========
   # ========== Buy-and-Hold Strategy Returns ==========
    st.subheader("💵 Buy-and-Hold Strategy Returns")

    if not data.empty:
        # Ensure 'Close' column exists and calculate buy-and-hold returns
        bh_start_price = data['Close'].iloc[0]  # First closing price in the selected range
        bh_end_price = data['Close'].iloc[-1]  # Last closing price in the selected range

        # Calculate buy-and-hold return percentage
        bh_return_pct = ((bh_end_price - bh_start_price) / bh_start_price) * 100

        # Calculate final portfolio value based on initial capital
        bh_final_value = initial_capital * (1 + (bh_return_pct / 100))

        cagr = (bh_final_value/initial_capital)**(1/difference_in_years) - 1
        # Display Buy-and-Hold Results
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Initial Investment", f"${initial_capital:,.2f}")
        col2.metric("Final Value", f"${bh_final_value[symbol]:,.2f}")
        col3.metric("Return Percentage", f"{bh_return_pct[symbol]:,.2f}%")
        col4.metric("Annualized Return", f"{cagr[symbol]*100:,.2f}%")
    else:
        raise ValueError("No data available for the selected ticker and date range.")

   # Add performance comparison summary
    try:
        performance_difference = portfolio.annualized_return() - cagr[symbol]
        comparison_text = (
            f"**The trading strategy outperformed buy-and-hold by {(performance_difference)*100:.2f} percentage points.**"
            if performance_difference > 0 else
            f"**The trading strategy underperformed buy-and-hold by {abs(performance_difference)*100:.2f} percentage points.**"
        )
        st.markdown(comparison_text)
    except Exception as e:
        st.error(f"Could not compare performances: {str(e)}")
   
    # Display summary in Streamlit
    st.subheader("📊 Trade Performance Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Number of Positive Trades", f"{len(positive_returns)}")
    col2.metric("Number of Negative Trades", f"{len(negative_returns)}")
    col3.metric("Maximum Positive Return", f"{max_positive_return:.2f}%")
    col4.metric("Maximum Negative Return", f"{max_negative_return:.2f}%")


    # Trade list
    st.subheader("Trade History")

    trades_df = portfolio.trades.records

    new_trades_df = pd.DataFrame({
        'Entry Date': portfolio.wrapper.index[portfolio.trades.records['entry_idx']],
        'Exit Date': portfolio.wrapper.index[portfolio.trades.records['exit_idx']],
        'Entry Price': trades_df['entry_price'],
        'Exit Price': trades_df['exit_price'],
        'Return (%)': trades_df['return'] * 100,
        'Duration (days)': trades_df['exit_idx'] - trades_df['entry_idx']
    })

    st.dataframe(new_trades_df.style.format({
        'Entry Price': '${:.2f}',
        'Exit Price': '${:.2f}',
        'Return (%)': '{:.2f}%',
        'Duration (days)': '{:.0f}'
    }), height=300)


    # ========== Trade Performance Summary ==========
    # Ensure new_trades_df exists and has a 'returns' column
    if 'trades_df' in locals() and 'return' in trades_df.columns:
        # Calculate metrics
        positive_returns = trades_df[trades_df['return'] > 0]['return']
        negative_returns = trades_df[trades_df['return'] < 0]['return']

        positive_trades_count = len(positive_returns)
        negative_trades_count = len(negative_returns)
        max_positive_return = (positive_returns.max() if not positive_returns.empty else 0)*100
        max_negative_return = (negative_returns.min() if not negative_returns.empty else 0)*100

        # Display summary in Streamlit
        st.subheader("📊 Trade Performance Summary")
        st.markdown(f"""
        - **Number of Positive Trades:** {positive_trades_count}
        - **Number of Negative Trades:** {negative_trades_count}
        - **Maximum Positive Return:** {max_positive_return:.2f}%
        - **Maximum Negative Return:** {max_negative_return:.2f}%
        """)
    else:
        st.warning("Trade data (new_trades_df) is not available or does not contain the 'returns' column.")


    # Equity curve
    st.subheader("Equity Curve")
    fig = go.Figure(go.Scatter(x=portfolio.value().index, y=portfolio.value(), mode='lines'))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Debug Information Tab
with tab2:
    ## Summary
    st.header("Debug Information")
    st.write("Entries Signal:", entries)
    st.write("Exits Signal:", exits)
    st.write("Aligned Close Prices:", aligned_close)
    st.write("Number of trades:", len(portfolio.trades.records))
    st.write("Portfolio Trades:", portfolio.trades.records)

    # Trade list with Days Held
    st.subheader("Trade History")

    # # Convert Entry Date and Exit Date to datetime (if not already in datetime format)
    # trades_df['Entry Date'] = pd.to_datetime(trades_df['Entry Date'])
    # trades_df['Exit Date'] = pd.to_datetime(trades_df['Exit Date'])


    # Create a DataFrame for trades
    trades_df = pd.DataFrame({
        'Entry Date': portfolio.wrapper.index[portfolio.trades.records['entry_idx']],
        'Exit Date': portfolio.wrapper.index[portfolio.trades.records['exit_idx']],
        'Entry Price': portfolio.trades.records['entry_price'],
        'Exit Price': portfolio.trades.records['exit_price'],
        'Return (%)': portfolio.trades.records['return'] * 100,
        'Duration (days)': portfolio.trades.records['exit_idx'] - portfolio.trades.records['entry_idx']
    })

    # Display the DataFrame with formatting
    st.dataframe(trades_df.style.format({
        'Entry Price': '${:.2f}',
        'Exit Price': '${:.2f}',
        'Return (%)': '{:.2f}%',
        'Duration (days)': '{:.0f}'
    }), height=300)


    try:
        st.write("Portfolio value:", portfolio.total_wealth())
    except AttributeError:
        st.write("Portfolio value:", portfolio.value())


