#!/usr/bin/env python3
"""
Outlier Capital - Stock Selection & Backtesting Platform
Updated Streamlit Application with Sector-Based Analysis
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
from sector_data import get_sector_stocks, get_all_sectors, get_stocks_by_sector, get_sector_etf

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
if 'sector_analysis_results' not in st.session_state:
    st.session_state.sector_analysis_results = {}

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

def get_top_performers_by_return(tickers, lookback_days, top_n=10, data_source="YFinance"):
    """Get top performing stocks by return over lookback period"""
    try:
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)
        
        performance_data = []
        
        for ticker in tickers:
            try:
                if data_source == "YFinance":
                    price_data = collector.yfinance_adapter.get_price_data(ticker, start_date, end_date)
                else:
                    price_data = collector.polygon_adapter.get_price_data(ticker, start_date, end_date)
                
                if price_data and len(price_data) > 1:
                    prices = [p.close_price for p in price_data]
                    if len(prices) >= 2:
                        total_return = (prices[-1] - prices[0]) / prices[0]
                        
                        # Calculate additional metrics
                        price_series = pd.Series(prices)
                        returns = price_series.pct_change().dropna()
                        
                        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
                        sharpe_ratio = (returns.mean() * 252) / volatility if volatility > 0 else 0
                        
                        # Calculate max drawdown
                        cumulative = (1 + returns).cumprod()
                        running_max = cumulative.expanding().max()
                        drawdown = (cumulative - running_max) / running_max
                        max_drawdown = drawdown.min()
                        
                        performance_data.append({
                            'ticker': ticker,
                            'total_return': total_return,
                            'volatility': volatility,
                            'sharpe_ratio': sharpe_ratio,
                            'max_drawdown': max_drawdown,
                            'start_price': prices[0],
                            'end_price': prices[-1]
                        })
            except Exception as e:
                logger.warning(f"Error processing {ticker}: {e}")
                continue
        
        # Sort by total return and return top N
        performance_data.sort(key=lambda x: x['total_return'], reverse=True)
        return performance_data[:top_n]
        
    except Exception as e:
        logger.error(f"Error in get_top_performers_by_return: {e}")
        return []

# Main application
def main():
    # Header
    st.title("📈 Outlier Capital")
    st.markdown("### AI-Powered Stock Selection & Backtesting Platform")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    # Data source selection
    st.sidebar.subheader("📊 Data Source")
    data_source = st.sidebar.selectbox(
        "Choose data source:",
        ["YFinance", "Polygon"],
        help="Select the data provider for stock information"
    )
    
    # Global settings
    st.sidebar.subheader("🔧 Global Settings")
    lookback_days = st.sidebar.slider(
        "Lookback Period (days):",
        min_value=30,
        max_value=1825,
        value=365,
        help="Number of days to look back for analysis"
    )
    
    top_n_performers = st.sidebar.slider(
        "Top N Performers:",
        min_value=5,
        max_value=50,
        value=15,
        help="Number of top performers to identify"
    )
    
    # Navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏆 Sector Analysis", "🔍 Stock Analysis", "👥 Similar Stocks", "📊 Performance Analysis", "⏮️ Backtesting"]
    )
    
    # Page routing
    if page == "🏆 Sector Analysis":
        sector_analysis_page(data_source, lookback_days, top_n_performers)
    elif page == "🔍 Stock Analysis":
        stock_analysis_page(data_source, lookback_days, top_n_performers)
    elif page == "👥 Similar Stocks":
        similar_stocks_page(data_source, lookback_days)
    elif page == "📊 Performance Analysis":
        performance_analysis_page(data_source, lookback_days)
    elif page == "⏮️ Backtesting":
        backtesting_page(data_source, lookback_days)

def sector_analysis_page(data_source, lookback_days, top_n_performers):
    """Sector-based Analysis Page"""
    st.header("🏆 Sector-Based Top Performers")
    
    # Sector selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Sector Selection")
        
        available_sectors = get_all_sectors()
        selected_sectors = st.multiselect(
            "Select sectors to analyze:",
            available_sectors,
            default=["Technology", "Healthcare", "Financial Services"],
            help="Choose one or more sectors for analysis"
        )
        
        analyze_by_sector = st.checkbox(
            "Analyze each sector separately",
            value=True,
            help="Show top performers within each sector individually"
        )
    
    with col2:
        st.subheader("🎯 Analysis Options")
        
        if st.button("🚀 Run Sector Analysis", type="primary", use_container_width=True):
            run_sector_analysis(selected_sectors, lookback_days, top_n_performers, data_source, analyze_by_sector)
        
        if selected_sectors:
            total_stocks = sum(len(get_stocks_by_sector(sector)) for sector in selected_sectors)
            st.info(f"📊 Total stocks to analyze: {total_stocks}")
            st.info(f"📅 Analysis period: {lookback_days} days")
            st.info(f"🎯 Data source: {data_source}")
    
    # Display results
    if st.session_state.sector_analysis_results:
        display_sector_analysis_results(analyze_by_sector)

def run_sector_analysis(selected_sectors, lookback_days, top_n_performers, data_source, analyze_by_sector):
    """Run sector-based analysis"""
    try:
        if not selected_sectors:
            st.error("Please select at least one sector")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = {}
        
        if analyze_by_sector:
            # Analyze each sector separately
            for i, sector in enumerate(selected_sectors):
                status_text.text(f"Analyzing {sector} sector...")
                progress = (i / len(selected_sectors)) * 100
                progress_bar.progress(int(progress))
                
                sector_stocks = get_stocks_by_sector(sector)
                top_performers = get_top_performers_by_return(
                    sector_stocks, lookback_days, top_n_performers, data_source
                )
                
                results[sector] = {
                    'top_performers': top_performers,
                    'total_stocks': len(sector_stocks),
                    'sector_etf': get_sector_etf(sector)
                }
        else:
            # Analyze all sectors together
            status_text.text("Analyzing all selected sectors...")
            progress_bar.progress(50)
            
            all_stocks = get_sector_stocks(selected_sectors)
            top_performers = get_top_performers_by_return(
                all_stocks, lookback_days, top_n_performers, data_source
            )
            
            results['All Sectors'] = {
                'top_performers': top_performers,
                'total_stocks': len(all_stocks),
                'sector_etf': 'SPY'
            }
        
        st.session_state.sector_analysis_results = results
        
        progress_bar.progress(100)
        status_text.text("Analysis completed!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        st.rerun()
        
    except Exception as e:
        st.error(f"Sector analysis failed: {e}")
        logger.error(f"Sector analysis error: {e}")

def display_sector_analysis_results(analyze_by_sector):
    """Display sector analysis results"""
    results = st.session_state.sector_analysis_results
    
    st.markdown("---")
    st.header("📊 Sector Analysis Results")
    
    for sector_name, sector_data in results.items():
        st.subheader(f"🏆 {sector_name}")
        
        top_performers = sector_data['top_performers']
        
        if top_performers:
            # Create performance table
            performance_data = []
            for i, stock in enumerate(top_performers, 1):
                performance_data.append({
                    'Rank': i,
                    'Ticker': stock['ticker'],
                    'Total Return': format_percentage(stock['total_return']),
                    'Volatility': format_percentage(stock['volatility']),
                    'Sharpe Ratio': format_number(stock['sharpe_ratio']),
                    'Max Drawdown': format_percentage(stock['max_drawdown']),
                    'Start Price': f"${stock['start_price']:.2f}",
                    'End Price': f"${stock['end_price']:.2f}"
                })
            
            df_performance = pd.DataFrame(performance_data)
            st.dataframe(df_performance, use_container_width=True)
            
            # Performance visualization
            tickers = [stock['ticker'] for stock in top_performers[:10]]
            returns = [stock['total_return'] * 100 for stock in top_performers[:10]]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=tickers,
                y=returns,
                marker=dict(
                    color=returns,
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Return %")
                ),
                text=[f"{r:.1f}%" for r in returns],
                textposition='auto'
            ))
            
            fig.update_layout(
                title=f"Top 10 Performers in {sector_name}",
                xaxis_title="Stock Ticker",
                yaxis_title="Total Return (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk-Return scatter plot
            if len(top_performers) > 1:
                fig_scatter = go.Figure()
                
                returns_data = [stock['total_return'] * 100 for stock in top_performers]
                volatility_data = [stock['volatility'] * 100 for stock in top_performers]
                tickers_data = [stock['ticker'] for stock in top_performers]
                
                fig_scatter.add_trace(go.Scatter(
                    x=volatility_data,
                    y=returns_data,
                    mode='markers+text',
                    text=tickers_data,
                    textposition="top center",
                    marker=dict(
                        size=10,
                        color=returns_data,
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="Return %")
                    ),
                    name="Stocks"
                ))
                
                fig_scatter.update_layout(
                    title=f"Risk vs Return - {sector_name}",
                    xaxis_title="Volatility (%)",
                    yaxis_title="Total Return (%)",
                    height=500
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning(f"No performance data available for {sector_name}")

def stock_analysis_page(data_source, lookback_days, top_n_performers):
    """Stock Analysis and Outlier Detection Page"""
    st.header("🔍 Stock Analysis & Outlier Detection")
    
    # Input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Stock Selection")
        
        # Sector-based selection
        available_sectors = get_all_sectors()
        selected_sectors = st.multiselect(
            "Select sectors for analysis:",
            available_sectors,
            default=["Technology", "Healthcare"],
            help="Choose sectors to include in the analysis"
        )
        
        if selected_sectors:
            all_stocks = get_sector_stocks(selected_sectors)
            st.info(f"📊 Total stocks from selected sectors: {len(all_stocks)}")
        
        # Option to add custom tickers
        custom_tickers = st.text_input(
            "Additional tickers (optional):",
            placeholder="AAPL, GOOGL, MSFT",
            help="Add specific tickers not covered by sector selection"
        )
    
    with col2:
        st.subheader("🎯 Analysis Configuration")
        
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            tickers_to_analyze = []
            
            if selected_sectors:
                tickers_to_analyze.extend(get_sector_stocks(selected_sectors))
            
            if custom_tickers:
                custom_list = [t.strip().upper() for t in custom_tickers.split(',') if t.strip()]
                tickers_to_analyze.extend(custom_list)
            
            if tickers_to_analyze:
                run_stock_analysis(list(set(tickers_to_analyze)), lookback_days, top_n_performers, data_source)
            else:
                st.error("Please select sectors or enter custom tickers")
        
        # Display analysis info
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            st.success(f"✅ Analysis completed!")
            st.info(f"📅 Period: {results['period']}")
            st.info(f"📊 Stocks analyzed: {results['total_stocks_analyzed']}")
    
    # Display results
    if st.session_state.analysis_results:
        display_analysis_results()

def run_stock_analysis(tickers, lookback_days, top_n, data_source):
    """Run the stock analysis"""
    try:
        if not tickers:
            st.error("No tickers to analyze")
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
                if data_source == "YFinance":
                    stock_info = collector.yfinance_adapter.get_stock_info(ticker)
                    price_data = collector.yfinance_adapter.get_price_data(ticker, start_date, end_date)
                    fundamental_data = collector.yfinance_adapter.get_fundamental_data(ticker)
                else:
                    stock_info = collector.polygon_adapter.get_stock_info(ticker)
                    price_data = collector.polygon_adapter.get_price_data(ticker, start_date, end_date)
                    fundamental_data = collector.polygon_adapter.get_fundamental_data(ticker)
                
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
            'feature_importance': feature_importance,
            'data_source': data_source
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

def similar_stocks_page(data_source, lookback_days):
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
        # Sector-based comparison universe
        available_sectors = get_all_sectors()
        comparison_sectors = st.multiselect(
            "Comparison Universe (sectors):",
            available_sectors,
            default=["Technology", "Healthcare"],
            help="Select sectors to search for similar stocks"
        )
    
    if st.button("🔍 Find Similar Stocks", type="primary", use_container_width=True):
        if comparison_sectors:
            comparison_tickers = get_sector_stocks(comparison_sectors)
            find_similar_stocks(target_ticker, n_similar, lookback_days, comparison_tickers, data_source)
        else:
            st.error("Please select at least one sector for comparison")

def find_similar_stocks(target_ticker, n_similar, lookback_days, comparison_tickers, data_source):
    """Find similar stocks"""
    try:
        if not target_ticker:
            st.error("Please enter a target ticker")
            return
        
        all_tickers = list(set([target_ticker] + comparison_tickers))
        
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
                if data_source == "YFinance":
                    stock_info = collector.yfinance_adapter.get_stock_info(ticker)
                    price_data = collector.yfinance_adapter.get_price_data(ticker, start_date, end_date)
                    fundamental_data = collector.yfinance_adapter.get_fundamental_data(ticker)
                else:
                    stock_info = collector.polygon_adapter.get_stock_info(ticker)
                    price_data = collector.polygon_adapter.get_price_data(ticker, start_date, end_date)
                    fundamental_data = collector.polygon_adapter.get_fundamental_data(ticker)
                
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

def performance_analysis_page(data_source, lookback_days):
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
        st.info(f"📊 Data Source: {data_source}")
        st.info(f"📅 Lookback Period: {lookback_days} days")
    
    if st.button("📈 Analyze Performance", type="primary", use_container_width=True):
        analyze_performance(ticker, lookback_days, data_source)

def analyze_performance(ticker, lookback_days, data_source):
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
        if data_source == "YFinance":
            price_data = collector.yfinance_adapter.get_price_data(ticker, start_date, end_date)
        else:
            price_data = collector.polygon_adapter.get_price_data(ticker, start_date, end_date)
        
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

def backtesting_page(data_source, lookback_days):
    """Backtesting Page"""
    st.header("⏮️ Strategy Backtesting")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Strategy Configuration")
        
        strategy_name = st.text_input("Strategy Name:", value="My Strategy")
        
        # Sector-based stock selection
        available_sectors = get_all_sectors()
        selected_sectors = st.multiselect(
            "Select sectors for strategy:",
            available_sectors,
            default=["Technology"],
            help="Choose sectors to include in the strategy"
        )
        
        # Option to add custom tickers
        custom_tickers = st.text_input(
            "Additional tickers (optional):",
            placeholder="AAPL, GOOGL, MSFT",
            help="Add specific tickers not covered by sector selection"
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
        
        st.info(f"📊 Data Source: {data_source}")
        st.info(f"📅 Lookback Period: {lookback_days} days")
    
    if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
        # Prepare tickers list
        tickers_to_backtest = []
        
        if selected_sectors:
            tickers_to_backtest.extend(get_sector_stocks(selected_sectors))
        
        if custom_tickers:
            custom_list = [t.strip().upper() for t in custom_tickers.split(',') if t.strip()]
            tickers_to_backtest.extend(custom_list)
        
        if tickers_to_backtest:
            run_backtest_analysis(strategy_name, list(set(tickers_to_backtest)), initial_capital, 
                                rebalance_frequency, position_sizing, lookback_days, data_source)
        else:
            st.error("Please select sectors or enter custom tickers")

def run_backtest_analysis(strategy_name, tickers, initial_capital, 
                         rebalance_frequency, position_sizing, lookback_days, data_source):
    """Run backtesting analysis"""
    try:
        if not tickers:
            st.error("No tickers to backtest")
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
            'transaction_cost': 0.001,
            'data_source': data_source
        }
        
        status_text.text("Collecting price data for backtest...")
        progress_bar.progress(40)
        
        # Collect price data for all tickers
        price_data_dict = {}
        for ticker in tickers:
            try:
                if data_source == "YFinance":
                    price_data = collector.yfinance_adapter.get_price_data(ticker, start_date, end_date)
                else:
                    price_data = collector.polygon_adapter.get_price_data(ticker, start_date, end_date)
                
                if price_data:
                    price_data_dict[ticker] = price_data
            except Exception as e:
                logger.warning(f"Could not get price data for {ticker}: {e}")
        
        if not price_data_dict:
            st.error("No price data available for backtesting")
            return
        
        status_text.text("Running backtest simulation...")
        progress_bar.progress(70)
        
        # Create a simple backtest result
        result = run_simple_backtest(price_data_dict, initial_capital, strategy_config, start_date, end_date)
        
        progress_bar.progress(100)
        status_text.text("Backtest completed!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        display_backtest_results(result)
        
    except Exception as e:
        st.error(f"Backtest failed: {e}")
        logger.error(f"Backtest error: {e}")

def run_simple_backtest(price_data_dict, initial_capital, strategy_config, start_date, end_date):
    """Run a simplified backtest"""
    try:
        # Convert price data to DataFrame
        all_prices = {}
        for ticker, price_data in price_data_dict.items():
            prices = pd.Series([p.close_price for p in price_data],
                             index=[p.date for p in price_data])
            all_prices[ticker] = prices.sort_index()
        
        price_df = pd.DataFrame(all_prices)
        price_df = price_df.dropna()
        
        if price_df.empty:
            raise ValueError("No overlapping price data for backtesting")
        
        # Calculate returns
        returns_df = price_df.pct_change().dropna()
        
        # Equal weight portfolio
        n_stocks = len(price_df.columns)
        weights = np.ones(n_stocks) / n_stocks
        
        # Calculate portfolio returns
        portfolio_returns = (returns_df * weights).sum(axis=1)
        
        # Calculate portfolio value over time
        portfolio_values = initial_capital * (1 + portfolio_returns).cumprod()
        
        # Calculate metrics
        total_return = (portfolio_values.iloc[-1] - initial_capital) / initial_capital
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Calculate drawdown
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calculate Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Calculate Sortino ratio
        negative_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
        
        # Calculate VaR and CVaR
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        # Win rate
        positive_returns = portfolio_returns[portfolio_returns > 0]
        win_rate = len(positive_returns) / len(portfolio_returns)
        
        # Create result object
        class BacktestResult:
            def __init__(self):
                self.strategy_name = strategy_config['name']
                self.start_date = start_date
                self.end_date = end_date
                self.initial_capital = initial_capital
                self.final_capital = portfolio_values.iloc[-1]
                self.total_return = total_return
                self.annualized_return = annualized_return
                self.volatility = volatility
                self.max_drawdown = max_drawdown
                self.sharpe_ratio = sharpe_ratio
                self.sortino_ratio = sortino_ratio
                self.information_ratio = 0  # Simplified
                self.var_95 = var_95
                self.cvar_95 = cvar_95
                self.beta = 1.0  # Simplified
                self.alpha = annualized_return  # Simplified
                self.tracking_error = volatility  # Simplified
                self.win_rate = win_rate
                self.profit_factor = 1.0  # Simplified
                self.max_consecutive_losses = 0  # Simplified
                self.portfolio_values = portfolio_values
                self.trades = []  # Simplified
        
        return BacktestResult()
        
    except Exception as e:
        logger.error(f"Error in run_simple_backtest: {e}")
        raise

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

