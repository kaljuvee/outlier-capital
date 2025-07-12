# Outlier Capital - Stock Selection & Backtesting Platform

🚀 **AI-Powered Stock Selection & Performance Backtesting Platform**

A comprehensive proof-of-concept application that uses machine learning to identify high-performing outlier stocks, analyze their characteristics, and provide backtesting capabilities for investment strategies.

## 🌟 Features

### 📊 Stock Analysis & Outlier Detection
- **Machine Learning-Powered Analysis**: Uses scikit-learn regression and classification models to identify outlier stocks
- **Configurable Parameters**: Analyze 5-50 top performers over customizable time windows (up to 5 years)
- **Feature Importance Analysis**: Identifies the most significant drivers of stock performance
- **Comprehensive Metrics**: Annual return, drawdown, Sharpe ratio, Sortino ratio, and more

### 👥 Similar Stock Discovery
- **Stock Similarity Engine**: Find stocks with similar characteristics to any target stock
- **Feature-Based Matching**: Uses advanced similarity algorithms based on financial and technical features
- **Customizable Universe**: Define your own comparison universe of stocks

### 📈 Performance Analysis
- **Comprehensive Metrics**: Total return, volatility, Sharpe ratio, maximum drawdown, VaR, CVaR
- **Interactive Visualizations**: Price charts, returns distribution, cumulative performance
- **Benchmark Comparison**: Compare against market benchmarks (SPY)

### ⏮️ Strategy Backtesting
- **Portfolio Simulation**: Test investment strategies with configurable parameters
- **Multiple Rebalancing Options**: Daily, weekly, monthly, or quarterly rebalancing
- **Risk Management**: Transaction costs, position sizing, and portfolio optimization
- **Performance Attribution**: Detailed breakdown of strategy performance

## 🛠️ Technology Stack

- **Frontend**: Streamlit (Interactive web application)
- **Backend**: Python with pandas, numpy, scikit-learn
- **Data Sources**: 
  - Yahoo Finance (via yfinance)
  - Polygon.io API
- **Machine Learning**: scikit-learn (regression, classification, clustering)
- **Visualizations**: Plotly (interactive charts and graphs)
- **Data Persistence**: CSV files for historical data and results

## 📋 Requirements

```
streamlit>=1.46.1
yfinance>=0.2.65
polygon-api-client>=1.15.1
scikit-learn>=1.7.0
plotly>=6.2.0
pandas>=2.3.1
numpy>=2.3.1
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kaljuvee/outlier-capital.git
cd outlier-capital
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser and navigate to `http://localhost:8501`

### Usage

#### 1. Stock Analysis
- Navigate to the "🔍 Stock Analysis" page
- Enter stock tickers (comma-separated)
- Configure lookback period and number of outliers
- Click "🚀 Run Analysis" to identify top-performing outlier stocks

#### 2. Similar Stocks
- Go to "👥 Similar Stocks" page
- Enter a target stock ticker
- Specify the number of similar stocks to find
- Define the comparison universe
- Click "🔍 Find Similar Stocks"

#### 3. Performance Analysis
- Visit "📊 Performance Analysis" page
- Enter a stock ticker
- Set the analysis period
- Click "📈 Analyze Performance" for comprehensive metrics and charts

#### 4. Backtesting
- Access "⏮️ Backtesting" page
- Configure your strategy (name, tickers, capital)
- Set rebalancing frequency and position sizing
- Click "🚀 Run Backtest" to simulate strategy performance

## 📊 Data Sources

### Yahoo Finance (yfinance)
- **Stock Prices**: Daily OHLCV data
- **Fundamental Data**: Financial ratios, market cap, sector information
- **Coverage**: Global markets with extensive historical data
- **Cost**: Free

### Polygon.io
- **Real-time Data**: Live market data and news
- **Alternative Data**: Social sentiment, insider trading
- **API Key**: `3lKo1IgQ3hXMjMCkmbQACTJySZHkfld7`
- **Rate Limits**: Configured for development use

## 🧠 Machine Learning Models

### Feature Engineering
- **Technical Indicators**: RSI, MACD, Bollinger Bands, moving averages
- **Fundamental Ratios**: P/E, P/B, ROE, debt ratios
- **Market Metrics**: Beta, correlation, volatility measures
- **Performance Metrics**: Returns, Sharpe ratio, maximum drawdown

### Model Types
1. **Regression Models**: Predict future returns based on features
2. **Classification Models**: Categorize stocks by performance tiers
3. **Clustering**: Group similar stocks for comparison
4. **Outlier Detection**: Identify stocks with unusual characteristics

### Feature Importance
- **Random Forest**: Tree-based feature importance
- **Gradient Boosting**: Advanced ensemble methods
- **Correlation Analysis**: Statistical feature relationships

## 📈 Performance Metrics

### Return Metrics
- **Total Return**: Cumulative performance over period
- **Annualized Return**: Geometric mean annual return
- **Excess Return**: Performance vs benchmark

### Risk Metrics
- **Volatility**: Standard deviation of returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Value at Risk (VaR)**: Potential loss at confidence level
- **Conditional VaR**: Expected loss beyond VaR threshold

### Risk-Adjusted Metrics
- **Sharpe Ratio**: Excess return per unit of risk
- **Sortino Ratio**: Downside risk-adjusted return
- **Information Ratio**: Active return vs tracking error
- **Calmar Ratio**: Annual return vs maximum drawdown

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set custom API endpoints
POLYGON_API_KEY=your_polygon_api_key
```

### Data Storage
- **Location**: `./data/` directory
- **Format**: CSV files for persistence
- **Structure**:
  - `stocks.csv`: Stock information and metadata
  - `prices/`: Individual stock price files
  - `analysis_results/`: Saved analysis outputs

## 📁 Project Structure

```
outlier-capital/
├── app.py                      # Main Streamlit application
├── data_collection_fixed.py    # Data collection modules
├── stock_analysis.py          # ML analysis and outlier detection
├── performance_engine.py      # Performance metrics and backtesting
├── data_models.py             # Data structures and schemas
├── requirements.txt           # Python dependencies
├── data/                      # Data storage directory
│   ├── stocks.csv            # Stock metadata
│   └── prices/               # Price data files
└── README.md                 # This file
```

## 🎯 Use Cases

### Individual Investors
- **Stock Screening**: Find high-potential stocks using AI
- **Portfolio Construction**: Build diversified portfolios
- **Risk Management**: Understand and manage investment risk

### Financial Advisors
- **Client Reporting**: Generate comprehensive performance reports
- **Strategy Development**: Test and validate investment strategies
- **Due Diligence**: Analyze stock characteristics and risks

### Quantitative Researchers
- **Factor Analysis**: Identify performance drivers
- **Model Development**: Test predictive models
- **Strategy Backtesting**: Validate trading strategies

## 🚧 Limitations & Disclaimers

- **Educational Purpose**: This is a proof-of-concept for demonstration
- **Not Financial Advice**: Results should not be used for actual investment decisions
- **Data Quality**: Dependent on external data sources and their limitations
- **Model Risk**: Machine learning models may not predict future performance
- **Market Risk**: Past performance does not guarantee future results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Data Sources**: Yahoo Finance, Polygon.io
- **Libraries**: scikit-learn, pandas, plotly, streamlit
- **Inspiration**: Modern portfolio theory and quantitative finance

## 📞 Contact

- **GitHub**: [@kaljuvee](https://github.com/kaljuvee)
- **Project Link**: [https://github.com/kaljuvee/outlier-capital](https://github.com/kaljuvee/outlier-capital)

---

**⚠️ Risk Warning**: This application is for educational and research purposes only. All investment decisions should be made with proper due diligence and professional advice. Past performance does not guarantee future results.

