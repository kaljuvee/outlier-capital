#!/usr/bin/env python3
"""
Outlier Capital - Stock Selection & Backtesting Platform
Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date, timedelta
import logging
import time

# Import our modules
from data_collection_fixed import DataCollector
from stock_analysis import FeatureEngineer, OutlierDetector, SimilarityAnalyzer, PredictiveModeler
from performance_engine import PerformanceCalculator, BacktestEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Outlier Capital",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'feature_matrix' not in st.session_state:
    st.session_state.feature_matrix = None
if 'predictive_modeler' not in st.session_state:
    st.session_state.predictive_modeler = PredictiveModeler()

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize analysis components"""
    POLYGON_API_KEY = "3lKo1IgQ3hXMjMCkmbQACTJySZHkfld7"
    
    collector = DataCollector(POLYGON_API_KEY)
    feature_engineer = FeatureEngineer()
    outlier_detector = OutlierDetector()
    similarity_analyzer = SimilarityAnalyzer()
    performance_calculator = PerformanceCalculator()
    backtest_engine = BacktestEngine()
    
    return collector, feature_engineer, outlier_detector, similarity_analyzer, performance_calculator, backtest_engine

collector, feature_engineer, outlier_detector, similarity_analyzer, performance_calculator, backtest_engine = initialize_components()

# Helper functions
def format_percentage(value):
    """Format value as percentage"""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.2f}%"

def format_number(value, decimals=2):
    """Format number with specified decimals"""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"

def format_currency(value):
    """Format value as currency"""
    if value is None or pd.isna(value):
        return "N/A"
    return f"${value:,.0f}"

# Main application
def main():
    # Header
    st.title("📈 Outlier Capital")
    st.markdown("### AI-Powered Stock Selection & Backtesting Platform")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🔍 Stock Analysis", "👥 Similar Stocks", "📊 Performance Analysis", "⏮️ Backtesting"]
    )
    
    # Page routing
    if page == "🔍 Stock Analysis":
        stock_analysis_page()
    elif page == "👥 Similar Stocks":
        similar_stocks_page()
    elif page == "📊 Performance Analysis":
        performance_analysis_page()
    elif page == "⏮️ Backtesting":
        backtesting_page()

def stock_analysis_page():
    """Stock Analysis and Outlier Detection Page"""
    st.header("🔍 Stock Analysis & Outlier Detection")
    
    # Input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Configuration")
        
        # Default tickers
        default_tickers = "AAPL, GOOGL, MSFT, TSLA, AMZN, META, NVDA, NFLX, CRM, ADBE"
        tickers_input = st.text_area(
            "Stock Tickers (comma-separated):",
            value=default_tickers,
            height=100,
            help="Enter stock ticker symbols separated by commas"
        )
        
        lookback_days = st.slider(
            "Lookback Period (days):",
            min_value=30,
            max_value=1825,
            value=365,
            help="Number of days to look back for analysis"
        )
        
        top_n = st.slider(
            "Top N Outliers:",
            min_value=1,
            max_value=50,
            value=10,
            help="Number of top outlier stocks to identify"
        )
    
    with col2:
        st.subheader("Analysis")
        
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            run_stock_analysis(tickers_input, lookback_days, top_n)
        
        # Display analysis info
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            st.success(f"✅ Analysis completed!")
            st.info(f"📅 Period: {results['period']}")
            st.info(f"📊 Stocks analyzed: {results['total_stocks_analyzed']}")
    
    # Display results
    if st.session_state.analysis_results:
        display_analysis_results()

def run_stock_analysis(tickers_input, lookback_days, top_n):
    """Run the stock analysis"""
    try:
        # Parse tickers
        tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
        
        if not tickers:
            st.error("Please enter at least one ticker symbol")
            return
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Calculate date range
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)
        
        status_text.text("Collecting stock data...")
        progress_bar.progress(20)
        
        # Collect data for all tickers
        stocks_data = {}
        for i, ticker in enumerate(tickers):
            try:
                stock_info = collector.yfinance_adapter.get_stock_info(ticker)
                price_data = collector.yfinance_adapter.get_price_data(ticker, start_date, end_date)
                fundamental_data = collector.yfinance_adapter.get_fundamental_data(ticker)
                
                stocks_data[ticker] = {
                    'stock_info': stock_info,
                    'price_data': price_data,
                    'fundamental_data': fundamental_data
                }
                
                # Update progress
                progress = 20 + (i + 1) / len(tickers) * 40
                progress_bar.progress(int(progress))
                
            except Exception as e:
                st.warning(f"Could not collect data for {ticker}: {e}")
        
        if not stocks_data:
            st.error("No data collected for any ticker")
            return
        
        status_text.text("Creating feature matrix...")
        progress_bar.progress(60)
        
        # Create feature matrix
        feature_matrix = feature_engineer.create_feature_matrix(stocks_data)
        st.session_state.feature_matrix = feature_matrix
        
        if feature_matrix.empty:
            st.error("Could not create feature matrix")
            return
        
        status_text.text("Detecting outliers...")
        progress_bar.progress(70)
        
        # Detect outliers
        outliers = outlier_detector.detect_outliers(feature_matrix, top_n=top_n)
        
        status_text.text("Training predictive models...")
        progress_bar.progress(80)
        
        # Train predictive models
        model_metrics = st.session_state.predictive_modeler.train_models(feature_matrix)
        
        status_text.text("Getting feature importance...")
        progress_bar.progress(90)
        
        # Get feature importance
        feature_importance = st.session_state.predictive_modeler.get_feature_importance()
        
        # Store results
        st.session_state.analysis_results = {
            'analysis_date': datetime.now().isoformat(),
            'period': f"{start_date} to {end_date}",
            'total_stocks_analyzed': len(stocks_data),
            'outliers': outliers,
            'model_metrics': model_metrics,
            'feature_importance': feature_importance
        }
        
        progress_bar.progress(100)
        status_text.text("Analysis completed!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        st.rerun()
        
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        logger.error(f"Analysis error: {e}")

def display_analysis_results():
    """Display the analysis results"""
    results = st.session_state.analysis_results
    
    st.markdown("---")
    st.header("📊 Analysis Results")
    
    # Model Performance Metrics
    st.subheader("🧠 Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    metrics = results['model_metrics']
    
    with col1:
        st.metric(
            "Regression MSE",
            format_number(metrics.get('regression_mse', 0), 6),
            help="Mean Squared Error for return prediction"
        )
    
    with col2:
        st.metric(
            "Classification Accuracy",
            format_percentage(metrics.get('classification_accuracy', 0)),
            help="Accuracy for performance category prediction"
        )
    
    with col3:
        st.metric(
            "Samples Analyzed",
            metrics.get('n_samples', 0),
            help="Number of stocks used for training"
        )
    
    # Top Outlier Stocks
    st.subheader("⭐ Top Outlier Stocks")
    
    outliers_data = []
    for outlier in results['outliers']:
        outliers_data.append({
            'Rank': outlier.rank,
            'Ticker': outlier.ticker,
            'Outlier Score': format_number(outlier.outlier_score, 3),
            'Confidence': format_percentage(outlier.confidence),
            'Total Return': format_percentage(outlier.total_return),
            'Sharpe Ratio': format_number(outlier.sharpe_ratio),
            'Max Drawdown': format_percentage(outlier.max_drawdown),
            'Key Factors': ', '.join(outlier.reasons[:3]) if outlier.reasons else 'N/A'
        })
    
    if outliers_data:
        df_outliers = pd.DataFrame(outliers_data)
        st.dataframe(df_outliers, use_container_width=True)
        
        # Outlier visualization
        fig_outliers = go.Figure()
        
        tickers = [outlier.ticker for outlier in results['outliers']]
        scores = [outlier.outlier_score for outlier in results['outliers']]
        returns = [outlier.total_return for outlier in results['outliers']]
        
        fig_outliers.add_trace(go.Scatter(
            x=scores,
            y=returns,
            mode='markers+text',
            text=tickers,
            textposition="top center",
            marker=dict(
                size=12,
                color=scores,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Outlier Score")
            ),
            name="Stocks"
        ))
        
        fig_outliers.update_layout(
            title="Outlier Score vs Total Return",
            xaxis_title="Outlier Score",
            yaxis_title="Total Return",
            height=500
        )
        
        st.plotly_chart(fig_outliers, use_container_width=True)
    
    # Feature Importance
    st.subheader("📈 Feature Importance")
    
    feature_importance = results['feature_importance']
    
    if feature_importance:
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
        
        features, importance_values = zip(*sorted_features)
        
        fig_importance = go.Figure()
        fig_importance.add_trace(go.Bar(
            x=importance_values,
            y=features,
            orientation='h',
            marker=dict(color='lightblue')
        ))
        
        fig_importance.update_layout(
            title="Top 15 Most Important Features",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=600,
            margin=dict(l=150)
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)

def similar_stocks_page():
    """Similar Stocks Analysis Page"""
    st.header("👥 Find Similar Stocks")
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_ticker = st.text_input(
            "Target Stock Ticker:",
            value="AAPL",
            help="Enter the ticker symbol to find similar stocks"
        ).upper()
        
        n_similar = st.slider(
            "Number of Similar Stocks:",
            min_value=1,
            max_value=20,
            value=5
        )
    
    with col2:
        lookback_days = st.slider(
            "Lookback Period (days):",
            min_value=30,
            max_value=1825,
            value=365
        )
        
        comparison_tickers = st.text_area(
            "Comparison Universe (comma-separated):",
            value="GOOGL, MSFT, AMZN, TSLA, META, NVDA, NFLX, CRM, ADBE, ORCL",
            help="Stocks to compare against"
        )
    
    if st.button("🔍 Find Similar Stocks", type="primary", use_container_width=True):
        find_similar_stocks(target_ticker, n_similar, lookback_days, comparison_tickers)

def find_similar_stocks(target_ticker, n_similar, lookback_days, comparison_tickers):
    """Find similar stocks"""
    try:
        if not target_ticker:
            st.error("Please enter a target ticker")
            return
        
        # Parse comparison tickers
        comparison_list = [t.strip().upper() for t in comparison_tickers.split(',') if t.strip()]
        all_tickers = list(set([target_ticker] + comparison_list))
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Calculate date range
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)
        
        status_text.text("Collecting data...")
        progress_bar.progress(20)
        
        # Collect data
        stocks_data = {}
        for i, ticker in enumerate(all_tickers):
            try:
                stock_info = collector.yfinance_adapter.get_stock_info(ticker)
                price_data = collector.yfinance_adapter.get_price_data(ticker, start_date, end_date)
                fundamental_data = collector.yfinance_adapter.get_fundamental_data(ticker)
                
                stocks_data[ticker] = {
                    'stock_info': stock_info,
                    'price_data': price_data,
                    'fundamental_data': fundamental_data
                }
                
                progress = 20 + (i + 1) / len(all_tickers) * 60
                progress_bar.progress(int(progress))
                
            except Exception as e:
                st.warning(f"Could not collect data for {ticker}: {e}")
        
        status_text.text("Analyzing similarities...")
        progress_bar.progress(80)
        
        # Create feature matrix
        feature_matrix = feature_engineer.create_feature_matrix(stocks_data)
        
        if feature_matrix.empty or target_ticker not in feature_matrix['ticker'].values:
            st.error(f"Could not analyze {target_ticker}")
            return
        
        # Find similar stocks
        similar_stocks = similarity_analyzer.find_similar_stocks(target_ticker, feature_matrix, n_similar=n_similar)
        
        progress_bar.progress(100)
        status_text.text("Analysis completed!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        st.subheader(f"📊 Stocks Similar to {target_ticker}")
        
        if similar_stocks:
            similar_data = []
            for stock in similar_stocks:
                similar_data.append({
                    'Ticker': stock.ticker,
                    'Similarity Score': format_percentage(stock.similarity_score),
                    'Distance': format_number(stock.distance, 3),
                    'Common Features': ', '.join(stock.common_features[:3]) if stock.common_features else 'N/A'
                })
            
            df_similar = pd.DataFrame(similar_data)
            st.dataframe(df_similar, use_container_width=True)
            
            # Similarity visualization
            tickers = [stock.ticker for stock in similar_stocks]
            similarities = [stock.similarity_score for stock in similar_stocks]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=tickers,
                y=similarities,
                marker=dict(color='lightgreen')
            ))
            
            fig.update_layout(
                title=f"Similarity Scores to {target_ticker}",
                xaxis_title="Stock Ticker",
                yaxis_title="Similarity Score",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No similar stocks found")
        
    except Exception as e:
        st.error(f"Similarity analysis failed: {e}")

def performance_analysis_page():
    """Performance Analysis Page"""
    st.header("📊 Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ticker = st.text_input(
            "Stock Ticker:",
            value="AAPL",
            help="Enter the ticker symbol for performance analysis"
        ).upper()
    
    with col2:
        lookback_days = st.slider(
            "Lookback Period (days):",
            min_value=30,
            max_value=1825,
            value=365
        )
    
    if st.button("📈 Analyze Performance", type="primary", use_container_width=True):
        analyze_performance(ticker, lookback_days)

def analyze_performance(ticker, lookback_days):
    """Analyze stock performance"""
    try:
        if not ticker:
            st.error("Please enter a ticker symbol")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Calculate date range
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)
        
        status_text.text("Collecting price data...")
        progress_bar.progress(30)
        
        # Collect price data
        price_data = collector.yfinance_adapter.get_price_data(ticker, start_date, end_date)
        
        if not price_data:
            st.error(f"No price data found for {ticker}")
            return
        
        status_text.text("Collecting benchmark data...")
        progress_bar.progress(60)
        
        # Get benchmark data (SPY)
        spy_data = collector.yfinance_adapter.get_price_data('SPY', start_date, end_date)
        
        status_text.text("Calculating metrics...")
        progress_bar.progress(80)
        
        # Convert to pandas Series
        prices = pd.Series([p.close_price for p in price_data],
                          index=[p.date for p in price_data])
        prices = prices.sort_index()
        
        benchmark_prices = None
        if spy_data:
            benchmark_prices = pd.Series([p.close_price for p in spy_data],
                                       index=[p.date for p in spy_data])
            benchmark_prices = benchmark_prices.sort_index()
        
        # Calculate comprehensive metrics
        metrics = performance_calculator.calculate_comprehensive_metrics(prices, benchmark_prices)
        
        progress_bar.progress(100)
        status_text.text("Analysis completed!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        display_performance_results(ticker, metrics, price_data)
        
    except Exception as e:
        st.error(f"Performance analysis failed: {e}")

def display_performance_results(ticker, metrics, price_data):
    """Display performance analysis results"""
    st.subheader(f"📊 Performance Metrics for {ticker}")
    
    # Performance metrics grid
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Return", format_percentage(metrics.get('total_return', 0)))
        st.metric("Volatility", format_percentage(metrics.get('volatility', 0)))
    
    with col2:
        st.metric("Annualized Return", format_percentage(metrics.get('annualized_return', 0)))
        st.metric("Sharpe Ratio", format_number(metrics.get('sharpe_ratio', 0)))
    
    with col3:
        st.metric("Max Drawdown", format_percentage(metrics.get('max_drawdown', 0)))
        st.metric("Sortino Ratio", format_number(metrics.get('sortino_ratio', 0)))
    
    with col4:
        st.metric("VaR (95%)", format_percentage(metrics.get('var_95', 0)))
        st.metric("CVaR (95%)", format_percentage(metrics.get('cvar_95', 0)))
    
    # Price chart
    st.subheader("📈 Price Chart")
    
    df = pd.DataFrame([{
        'Date': p.date,
        'Close': p.close_price,
        'Volume': p.volume
    } for p in price_data])
    
    df = df.sort_values('Date')
    df['Returns'] = df['Close'].pct_change()
    df['Cumulative Returns'] = (1 + df['Returns']).cumprod() - 1
    
    # Price chart
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name=f'{ticker} Price',
        line=dict(color='blue', width=2)
    ))
    
    fig_price.update_layout(
        title=f'{ticker} Price Performance',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=400
    )
    
    st.plotly_chart(fig_price, use_container_width=True)
    
    # Returns distribution and cumulative returns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Returns Distribution")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df['Returns'].dropna(),
            nbinsx=50,
            name='Daily Returns',
            opacity=0.7
        ))
        fig_hist.update_layout(
            title='Daily Returns Distribution',
            xaxis_title='Daily Returns',
            yaxis_title='Frequency',
            height=400
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.subheader("📈 Cumulative Returns")
        fig_cumret = go.Figure()
        fig_cumret.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Cumulative Returns'] * 100,
            mode='lines',
            name='Cumulative Returns',
            line=dict(color='green', width=2)
        ))
        fig_cumret.update_layout(
            title='Cumulative Returns',
            xaxis_title='Date',
            yaxis_title='Cumulative Returns (%)',
            height=400
        )
        st.plotly_chart(fig_cumret, use_container_width=True)

def backtesting_page():
    """Backtesting Page"""
    st.header("⏮️ Strategy Backtesting")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Strategy Configuration")
        
        strategy_name = st.text_input("Strategy Name:", value="My Strategy")
        
        tickers_input = st.text_area(
            "Stock Tickers (comma-separated):",
            value="AAPL, GOOGL, MSFT",
            height=100
        )
        
        initial_capital = st.number_input(
            "Initial Capital ($):",
            min_value=1000,
            value=100000,
            step=1000
        )
    
    with col2:
        st.subheader("Backtest Parameters")
        
        rebalance_frequency = st.selectbox(
            "Rebalance Frequency:",
            ["daily", "weekly", "monthly", "quarterly"],
            index=2
        )
        
        position_sizing = st.selectbox(
            "Position Sizing:",
            ["equal_weight", "market_cap_weight"],
            index=0
        )
        
        lookback_days = st.slider(
            "Lookback Period (days):",
            min_value=30,
            max_value=1825,
            value=365
        )
    
    if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
        run_backtest_analysis(strategy_name, tickers_input, initial_capital, 
                            rebalance_frequency, position_sizing, lookback_days)

def run_backtest_analysis(strategy_name, tickers_input, initial_capital, 
                         rebalance_frequency, position_sizing, lookback_days):
    """Run backtesting analysis"""
    try:
        # Parse tickers
        tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
        
        if not tickers:
            st.error("Please enter at least one ticker symbol")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Calculate date range
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)
        
        status_text.text("Setting up backtest...")
        progress_bar.progress(20)
        
        # Strategy configuration
        strategy_config = {
            'name': strategy_name,
            'tickers': tickers,
            'rebalance_frequency': rebalance_frequency,
            'position_sizing': position_sizing,
            'transaction_cost': 0.001
        }
        
        status_text.text("Running backtest simulation...")
        progress_bar.progress(50)
        
        # Initialize backtest engine
        custom_engine = BacktestEngine(initial_capital=initial_capital)
        
        # Run backtest
        result = custom_engine.run_backtest(strategy_config, start_date, end_date)
        
        progress_bar.progress(100)
        status_text.text("Backtest completed!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        display_backtest_results(result)
        
    except Exception as e:
        st.error(f"Backtest failed: {e}")

def display_backtest_results(result):
    """Display backtest results"""
    st.subheader(f"📊 Backtest Results: {result.strategy_name}")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Initial Capital", format_currency(result.initial_capital))
        st.metric("Final Capital", format_currency(result.final_capital))
    
    with col2:
        st.metric("Total Return", format_percentage(result.total_return))
        st.metric("Annualized Return", format_percentage(result.annualized_return))
    
    with col3:
        st.metric("Volatility", format_percentage(result.volatility))
        st.metric("Sharpe Ratio", format_number(result.sharpe_ratio))
    
    with col4:
        st.metric("Max Drawdown", format_percentage(result.max_drawdown))
        st.metric("Win Rate", format_percentage(result.win_rate))
    
    # Additional metrics
    st.subheader("📈 Additional Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Sortino Ratio", format_number(result.sortino_ratio))
        st.metric("Information Ratio", format_number(result.information_ratio))
    
    with col2:
        st.metric("VaR (95%)", format_percentage(result.var_95))
        st.metric("CVaR (95%)", format_percentage(result.cvar_95))
    
    with col3:
        st.metric("Beta", format_number(result.beta))
        st.metric("Alpha", format_percentage(result.alpha))
    
    # Portfolio value chart
    if not result.portfolio_values.empty:
        st.subheader("📈 Portfolio Value Over Time")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=result.portfolio_values.index,
            y=result.portfolio_values.values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title='Portfolio Value Over Time',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()

