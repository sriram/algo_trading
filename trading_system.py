import streamlit as st
import pandas as pd
import yfinance as yf
import vectorbt as vbt
import plotly.graph_objects as go
from datetime import datetime
from dateutil.relativedelta import relativedelta
import streamlit as st
import requests
import os
from tensorflow.keras.models import load_model
import numpy as np
import ta
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# Directory to save LSTM models
MODEL_DIR = "saved_models"

# Ensure the directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Configure page
st.set_page_config(layout="wide", page_title="Trading System Pro", page_icon="📈")
st.title("📈 Trading System with Buy/Sell Signals")

# Initialize session state variables
if "search_query" not in st.session_state:
    st.session_state["search_query"] = ""
if "search_results" not in st.session_state:
    st.session_state["search_results"] = []
if "selected_symbol" not in st.session_state:
    st.session_state["selected_symbol"] = None

# Function to fetch data from Yahoo Finance
def fetch_yahoo_finance_data(query):
    try:
        search_url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        response = requests.get(search_url, headers=headers, timeout=10)
        if response.status_code == 200:
            results = response.json().get("quotes", [])
            return [
                {"name": f"{item['shortname']} ({item['symbol']})", "symbol": item['symbol']}
                for item in results if 'symbol' in item and 'shortname' in item
            ]
        else:
            st.error(f"Error: Received status code {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {str(e)}")
        return []

# Function to reset session state variables
def reset_app():
    st.session_state["search_query"] = ""
    st.session_state["search_results"] = []
    st.session_state["selected_symbol"] = None


 # Function to calculate CAGR
def calculate_cagr(start_value, end_value, years):
    return (end_value / start_value) ** (1 / years) - 1

# Step 1: Search Box
st.header("Step 1: Search for a Stock/ETF")
search_query = st.text_input(
    "Enter Stock or ETF Name:",
    value=st.session_state.get("search_query", ""),
    key="search_box"
)

# Trigger search when user presses Enter
if search_query and search_query != st.session_state.get("search_query", ""):
    st.session_state["search_query"] = search_query
    with st.spinner("Searching..."):
        results = fetch_yahoo_finance_data(search_query)
        if results:
            # Add a placeholder option at the start of the dropdown
            st.session_state["search_results"] = [{"name": "Select an option...", "symbol": None}] + results
        else:
            st.warning("No matching stocks or ETFs found. Please refine your search.")
            st.session_state["search_results"] = []

# Step 2: Dropdown for Search Results
if st.session_state["search_results"]:
    selected_option = st.selectbox(
        "Select a Stock/ETF from the search results:",
        [result["name"] for result in st.session_state["search_results"]],
        key="results_dropdown"
    )

    # Update selected symbol based on user selection (ignore placeholder)
    if selected_option != "Select an option...":
        selected_symbol = next(
            (result["symbol"] for result in st.session_state["search_results"] if result["name"] == selected_option),
            None
        )
        if selected_symbol:
            st.success(f"You selected: {selected_option}")
            st.session_state["selected_symbol"] = selected_symbol

# Display Reset Button Only When Necessary
if (
    st.session_state.get("search_query")
    or st.session_state.get("search_results")
    or st.session_state.get("selected_symbol")
):
    # Reset button to clear all inputs and selections
    st.button("Reset Search", on_click=reset_app)

# Proceed only if a valid ticker symbol is selected
if st.session_state.get("selected_symbol"):
    symbol = st.session_state.get("selected_symbol")  # Ensure symbol is defined here
    with st.spinner(f"Loading {symbol} data..."):
        # Add your data loading and dashboard logic here...
        st.write(f"Proceeding with ticker: {symbol}")
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

        
        def calculate_signals(data, ticker):
            # Feature Engineering Function
            def create_features(df):
                df = df.copy()
                # Ensure proper pandas Series with index
                high = pd.Series(df['High'].squeeze(), index=df.index)
                low = pd.Series(df['Low'].squeeze(), index=df.index)
                close = pd.Series(df['Close'].squeeze(), index=df.index)
                
                # Calculate ATR with explicit index handling
                atr_indicator = ta.volatility.AverageTrueRange(
                    high=high,
                    low=low,
                    close=close,
                    window=14
                )
                df['atr'] = atr_indicator.average_true_range()
                print(df[['High', 'Low', 'Close', 'atr']].head(10))

                df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
                df['close_ratio'] = df['Close'] / df['Close'].rolling(50).mean()
                rsi_indicator = ta.momentum.RSIIndicator(close=df['Close'].squeeze(), window=14)
                df['RSI'] = rsi_indicator.rsi()
                print(rsi_indicator.rsi().shape)  # Should now be (1318,)
                print(df['RSI'].shape)  # Confirms 1D structure

                df['MACD'] = ta.trend.MACD(close=df['Close'].squeeze()).macd_diff()
                df['BB_Width'] = ta.volatility.BollingerBands(close=df['Close'].squeeze()).bollinger_wband()
                df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['Close'].squeeze(), df['Volume'].squeeze()).on_balance_volume()
                for lag in [1, 3, 5, 8, 13]:
                    df[f'ret_lag_{lag}'] = df['returns'].shift(lag)
                df['target'] = np.where(
                    df['returns'].shift(-3) > 0.015, 1,
                    np.where(df['returns'].shift(-3) < -0.015, -1, 0)
                )
                return df.dropna()

            # Feature Engineering
            with st.spinner("Creating features..."):
                df = create_features(data)

            # Feature Scaling
            scaler = StandardScaler()
            features = df.drop('target', axis=1)
            X = scaler.fit_transform(features)
            X = X.reshape((X.shape[0], 1, X.shape[1]))  # Reshape for LSTM

            # Train/Test Split (Time-based)
            split = int(len(df) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = df['target'].iloc[:split], df['target'].iloc[split:]

            # Model File Path
            model_path = os.path.join(MODEL_DIR, f"{ticker}_model.keras")

            # Check if model already exists
            if os.path.exists(model_path):
                with st.spinner(f"Loading saved model for {ticker}..."):
                    model = load_model(model_path)
                    st.success(f"Model loaded successfully!")
            else:
                # Model Architecture
                def create_model():
                    model = Sequential([
                        LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
                        Dropout(0.3),
                        LSTM(32),
                        Dropout(0.2),
                        Dense(1, activation='tanh')  # Output between -1 and 1
                    ])
                    model.compile(optimizer=Adam(0.001), loss='mean_squared_error', metrics=['accuracy'])
                    return model

                with st.spinner(f"Training new model for {ticker}..."):
                    model = create_model()
                    early_stop = EarlyStopping(monitor='val_loss', patience=5)
                    history = model.fit(
                        X_train, y_train,
                        epochs=50,
                        batch_size=32,
                        validation_split=0.2,
                        callbacks=[early_stop],
                        verbose=0
                    )
                    # Save the trained model
                    model.save(model_path)
                    st.success(f"Model saved successfully for {ticker}!")

            # Generate Predictions
            with st.spinner("Generating signals..."):
                train_pred = model.predict(X_train).flatten()
                test_pred = model.predict(X_test).flatten()
                full_pred = np.concatenate([train_pred, test_pred])

            # Signal Generation with Dynamic Thresholds
            upper_threshold = st.session_state.get('upper_threshold', 0.65)
            lower_threshold = st.session_state.get('lower_threshold', 0.35)

            df['signal'] = np.where(full_pred > upper_threshold, 1,
                                    np.where(full_pred < -lower_threshold, -1, 0))

            # Align signals with original data
            aligned_signals = df['signal'].reindex(data.index).ffill().dropna()
            aligned_close = data['Close'].reindex(aligned_signals.index)

            # Convert to boolean signals for VectorBT
            entries = (aligned_signals == 1) & (aligned_signals.shift(1) != 1)
            exits = (aligned_signals == -1) & (aligned_signals.shift(1) != -1)

            return entries, exits, aligned_close

        entries, exits, aligned_close = calculate_signals(data, symbol)

        st.subheader("Debug: Signal Analysis")
        st.write("Entries Signal (Buy):", entries)
        st.write("Exits Signal (Sell):", exits)
        st.write("Number of Buy Signals:", entries.sum())
        st.write("Number of Sell Signals:", exits.sum())


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
        elif most_recent_sell_date != "No Sell Signal":
            most_recent_signal = f"Sell on {most_recent_sell_date.strftime('%Y-%m-%d')}"
            signal_color = "red"
        else:
            most_recent_signal = f"No signals yet"
            signal_color = "gray"

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
        else:
            st.write("No trades were executed during the backtest period.")


        # Create tabs for main content and debugging
        tab1, tab2 = st.tabs(["Main Dashboard", "Debug Info"])

        # Main Dashboard Content
        with tab1:
            # ========== Visualization ==========
            st.header(f"Trading Signals for {symbol}")

            # Plot price chart with buy/sell signals
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=aligned_close.index, y=aligned_close, mode='lines', name='Close Price'))
            fig.add_trace(go.Scatter(x=aligned_close[entries].index, y=aligned_close[entries], mode='markers', name='Buy Signal', marker=dict(color='green', size=10, symbol='triangle-up')))
            fig.add_trace(go.Scatter(x=aligned_close[exits].index, y=aligned_close[exits], mode='markers', name='Sell Signal', marker=dict(color='red', size=10, symbol='triangle-down')))
            fig.update_layout(title="Price Chart with Buy/Sell Signals", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig)


            # ========== Performance Dashboard ==========
            st.header("Strategy Returns")

            # Example data (replace with actual backtesting results)
            # Calculate start value, end value, and CAGR
            start_value = portfolio.init_cash  # Initial cash used for backtesting
            end_value = portfolio.value().iloc[-1]  # Final portfolio value at the end of backtesting
            years = (end_date - start_date).days / 365.0

            # Calculate CAGR
            strat_cagr = calculate_cagr(start_value, end_value, years)

            # Display Strategy Returns Section
            col1, col2, col3, col4 = st.columns(4)

            # If total_return() returns a Series or DataFrame due to grouping/multiple assets:
            if isinstance(portfolio.total_return(), pd.Series) or isinstance(portfolio.total_return(), pd.DataFrame):
                total_return = portfolio.total_return().mean()  # Aggregate if needed
            else:
                total_return = portfolio.total_return()

            col1.metric("Initial Value", f"${start_value:,.2f}")
            col2.metric("Final Value", f"${end_value:,.2f}")
            col3.metric("Return Percentage", f"{total_return * 100:.2f}%")
            col4.metric("CAGR", f"{strat_cagr * 100:.2f}%")
            # col2.metric("Annualized Return", f"{portfolio.annualized_return()*100:.2f}%")

            # col1, col2, col3 = st.columns(3)
            # col1.metric("Total Return", f"{total_return * 100:.2f}%")
            # col2.metric("Annualized Return", f"{portfolio.annualized_return()*100:.2f}%")
            # col3.metric("Max Drawdown", f"{portfolio.max_drawdown()*100:.2f}%")


            # ========== Trade Performance Summary ==========
            # Extract trade returns from the portfolio
            trade_returns = portfolio.trades.records_readable['Return']

            # Calculate metrics using filtered returns
            positive_returns = trade_returns[trade_returns > 0]
            negative_returns = trade_returns[trade_returns < 0]

            max_positive_return = (positive_returns.max() if not positive_returns.empty else 0)*100
            max_negative_return = (negative_returns.min() if not negative_returns.empty else 0)*100  # Changed to min()

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
                col4.metric("CAGR", f"{cagr[symbol]*100:,.2f}%")
            else:
                raise ValueError("No data available for the selected ticker and date range.")

            # Add performance comparison summary
            try:
                performance_difference = strat_cagr - cagr[symbol]
                # Construct comparison text with dynamic color
                if performance_difference > 0:
                    comparison_text = (
                        f"<p style='color:green; font-size:18px;'>"
                        f"**The trading strategy outperformed buy-and-hold by {performance_difference * 100:.2f} percentage points.**"
                        f"</p>"
                    )
                else:
                    comparison_text = (
                        f"<p style='color:red; font-size:18px;'>"
                        f"**The trading strategy underperformed buy-and-hold by {abs(performance_difference) * 100:.2f} percentage points.**"
                        f"</p>"
                    )

                # Display the styled text using Markdown
                st.markdown(comparison_text, unsafe_allow_html=True)
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
else:
    if not search_query:
        # Display initial info message only when no input has been provided yet.
        st.info("Please enter a stock or ETF name to begin.")
       