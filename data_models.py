#!/usr/bin/env python3
"""
Data Models and CSV Schema Definitions for Stock Selection Platform

This module defines the data structures and CSV schemas used throughout
the stock selection and backtesting platform.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, Dict, List, Any
import pandas as pd
import json
import os

# CSV file paths and directory structure
DATA_DIR = "data"
STOCKS_FILE = os.path.join(DATA_DIR, "stocks.csv")
PRICES_DIR = os.path.join(DATA_DIR, "prices")
FUNDAMENTALS_DIR = os.path.join(DATA_DIR, "fundamentals")
ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")
PERFORMANCE_DIR = os.path.join(DATA_DIR, "performance")

@dataclass
class Stock:
    """
    Stock entity representing individual securities
    """
    ticker: str
    company_name: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    exchange: Optional[str] = None
    currency: str = "USD"
    country: str = "US"
    active_status: bool = True
    listing_date: Optional[date] = None
    delisting_date: Optional[date] = None
    last_updated: datetime = field(default_factory=datetime.now)
    data_source: str = "yfinance"
    
    @classmethod
    def get_csv_columns(cls) -> List[str]:
        """Return CSV column names for Stock entity"""
        return [
            'ticker', 'company_name', 'sector', 'industry', 'market_cap',
            'exchange', 'currency', 'country', 'active_status', 'listing_date',
            'delisting_date', 'last_updated', 'data_source'
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Stock instance to dictionary for CSV storage"""
        return {
            'ticker': self.ticker,
            'company_name': self.company_name,
            'sector': self.sector,
            'industry': self.industry,
            'market_cap': self.market_cap,
            'exchange': self.exchange,
            'currency': self.currency,
            'country': self.country,
            'active_status': self.active_status,
            'listing_date': self.listing_date.isoformat() if self.listing_date else None,
            'delisting_date': self.delisting_date.isoformat() if self.delisting_date else None,
            'last_updated': self.last_updated.isoformat(),
            'data_source': self.data_source
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Stock':
        """Create Stock instance from dictionary (CSV row)"""
        return cls(
            ticker=data['ticker'],
            company_name=data['company_name'],
            sector=data.get('sector'),
            industry=data.get('industry'),
            market_cap=float(data['market_cap']) if data.get('market_cap') else None,
            exchange=data.get('exchange'),
            currency=data.get('currency', 'USD'),
            country=data.get('country', 'US'),
            active_status=bool(data.get('active_status', True)),
            listing_date=date.fromisoformat(data['listing_date']) if data.get('listing_date') else None,
            delisting_date=date.fromisoformat(data['delisting_date']) if data.get('delisting_date') else None,
            last_updated=datetime.fromisoformat(data['last_updated']),
            data_source=data.get('data_source', 'yfinance')
        )

@dataclass
class PriceData:
    """
    Price data entity for historical and real-time price information
    """
    ticker: str
    date: date
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    adjusted_close: float
    volume: int
    dividend_amount: float = 0.0
    split_ratio: float = 1.0
    data_source: str = "yfinance"
    created_timestamp: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def get_csv_columns(cls) -> List[str]:
        """Return CSV column names for PriceData entity"""
        return [
            'ticker', 'date', 'open_price', 'high_price', 'low_price',
            'close_price', 'adjusted_close', 'volume', 'dividend_amount',
            'split_ratio', 'data_source', 'created_timestamp'
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert PriceData instance to dictionary for CSV storage"""
        return {
            'ticker': self.ticker,
            'date': self.date.isoformat(),
            'open_price': self.open_price,
            'high_price': self.high_price,
            'low_price': self.low_price,
            'close_price': self.close_price,
            'adjusted_close': self.adjusted_close,
            'volume': self.volume,
            'dividend_amount': self.dividend_amount,
            'split_ratio': self.split_ratio,
            'data_source': self.data_source,
            'created_timestamp': self.created_timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PriceData':
        """Create PriceData instance from dictionary (CSV row)"""
        return cls(
            ticker=data['ticker'],
            date=date.fromisoformat(data['date']),
            open_price=float(data['open_price']),
            high_price=float(data['high_price']),
            low_price=float(data['low_price']),
            close_price=float(data['close_price']),
            adjusted_close=float(data['adjusted_close']),
            volume=int(data['volume']),
            dividend_amount=float(data.get('dividend_amount', 0.0)),
            split_ratio=float(data.get('split_ratio', 1.0)),
            data_source=data.get('data_source', 'yfinance'),
            created_timestamp=datetime.fromisoformat(data['created_timestamp'])
        )

@dataclass
class FundamentalData:
    """
    Fundamental data entity for financial metrics and ratios
    """
    ticker: str
    report_date: date
    revenue: Optional[float] = None
    net_income: Optional[float] = None
    total_assets: Optional[float] = None
    total_debt: Optional[float] = None
    shareholders_equity: Optional[float] = None
    operating_cash_flow: Optional[float] = None
    free_cash_flow: Optional[float] = None
    price_to_earnings: Optional[float] = None
    price_to_book: Optional[float] = None
    debt_to_equity: Optional[float] = None
    return_on_equity: Optional[float] = None
    return_on_assets: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    data_source: str = "yfinance"
    created_timestamp: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def get_csv_columns(cls) -> List[str]:
        """Return CSV column names for FundamentalData entity"""
        return [
            'ticker', 'report_date', 'revenue', 'net_income', 'total_assets',
            'total_debt', 'shareholders_equity', 'operating_cash_flow',
            'free_cash_flow', 'price_to_earnings', 'price_to_book',
            'debt_to_equity', 'return_on_equity', 'return_on_assets',
            'gross_margin', 'operating_margin', 'net_margin',
            'data_source', 'created_timestamp'
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert FundamentalData instance to dictionary for CSV storage"""
        return {
            'ticker': self.ticker,
            'report_date': self.report_date.isoformat(),
            'revenue': self.revenue,
            'net_income': self.net_income,
            'total_assets': self.total_assets,
            'total_debt': self.total_debt,
            'shareholders_equity': self.shareholders_equity,
            'operating_cash_flow': self.operating_cash_flow,
            'free_cash_flow': self.free_cash_flow,
            'price_to_earnings': self.price_to_earnings,
            'price_to_book': self.price_to_book,
            'debt_to_equity': self.debt_to_equity,
            'return_on_equity': self.return_on_equity,
            'return_on_assets': self.return_on_assets,
            'gross_margin': self.gross_margin,
            'operating_margin': self.operating_margin,
            'net_margin': self.net_margin,
            'data_source': self.data_source,
            'created_timestamp': self.created_timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FundamentalData':
        """Create FundamentalData instance from dictionary (CSV row)"""
        return cls(
            ticker=data['ticker'],
            report_date=date.fromisoformat(data['report_date']),
            revenue=float(data['revenue']) if data.get('revenue') else None,
            net_income=float(data['net_income']) if data.get('net_income') else None,
            total_assets=float(data['total_assets']) if data.get('total_assets') else None,
            total_debt=float(data['total_debt']) if data.get('total_debt') else None,
            shareholders_equity=float(data['shareholders_equity']) if data.get('shareholders_equity') else None,
            operating_cash_flow=float(data['operating_cash_flow']) if data.get('operating_cash_flow') else None,
            free_cash_flow=float(data['free_cash_flow']) if data.get('free_cash_flow') else None,
            price_to_earnings=float(data['price_to_earnings']) if data.get('price_to_earnings') else None,
            price_to_book=float(data['price_to_book']) if data.get('price_to_book') else None,
            debt_to_equity=float(data['debt_to_equity']) if data.get('debt_to_equity') else None,
            return_on_equity=float(data['return_on_equity']) if data.get('return_on_equity') else None,
            return_on_assets=float(data['return_on_assets']) if data.get('return_on_assets') else None,
            gross_margin=float(data['gross_margin']) if data.get('gross_margin') else None,
            operating_margin=float(data['operating_margin']) if data.get('operating_margin') else None,
            net_margin=float(data['net_margin']) if data.get('net_margin') else None,
            data_source=data.get('data_source', 'yfinance'),
            created_timestamp=datetime.fromisoformat(data['created_timestamp'])
        )

@dataclass
class AnalysisResult:
    """
    Analysis results entity for storing ML model outputs and predictions
    """
    analysis_id: str
    ticker: str
    analysis_date: date
    analysis_type: str  # 'outlier_detection', 'similarity', 'prediction'
    model_name: str
    prediction_value: Optional[float] = None
    confidence_score: Optional[float] = None
    feature_importance: Optional[Dict[str, float]] = None
    performance_metrics: Optional[Dict[str, float]] = None
    model_version: str = "1.0"
    created_timestamp: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def get_csv_columns(cls) -> List[str]:
        """Return CSV column names for AnalysisResult entity"""
        return [
            'analysis_id', 'ticker', 'analysis_date', 'analysis_type',
            'model_name', 'prediction_value', 'confidence_score',
            'feature_importance_json', 'performance_metrics_json',
            'model_version', 'created_timestamp'
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert AnalysisResult instance to dictionary for CSV storage"""
        return {
            'analysis_id': self.analysis_id,
            'ticker': self.ticker,
            'analysis_date': self.analysis_date.isoformat(),
            'analysis_type': self.analysis_type,
            'model_name': self.model_name,
            'prediction_value': self.prediction_value,
            'confidence_score': self.confidence_score,
            'feature_importance_json': json.dumps(self.feature_importance) if self.feature_importance else None,
            'performance_metrics_json': json.dumps(self.performance_metrics) if self.performance_metrics else None,
            'model_version': self.model_version,
            'created_timestamp': self.created_timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisResult':
        """Create AnalysisResult instance from dictionary (CSV row)"""
        return cls(
            analysis_id=data['analysis_id'],
            ticker=data['ticker'],
            analysis_date=date.fromisoformat(data['analysis_date']),
            analysis_type=data['analysis_type'],
            model_name=data['model_name'],
            prediction_value=float(data['prediction_value']) if data.get('prediction_value') else None,
            confidence_score=float(data['confidence_score']) if data.get('confidence_score') else None,
            feature_importance=json.loads(data['feature_importance_json']) if data.get('feature_importance_json') else None,
            performance_metrics=json.loads(data['performance_metrics_json']) if data.get('performance_metrics_json') else None,
            model_version=data.get('model_version', '1.0'),
            created_timestamp=datetime.fromisoformat(data['created_timestamp'])
        )

@dataclass
class PerformanceMetrics:
    """
    Performance metrics entity for backtesting and strategy evaluation
    """
    metrics_id: str
    ticker: str
    start_date: date
    end_date: date
    total_return: float
    annualized_return: float
    volatility: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    information_ratio: Optional[float] = None
    var_95: Optional[float] = None
    cvar_95: Optional[float] = None
    beta: Optional[float] = None
    alpha: Optional[float] = None
    tracking_error: Optional[float] = None
    created_timestamp: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def get_csv_columns(cls) -> List[str]:
        """Return CSV column names for PerformanceMetrics entity"""
        return [
            'metrics_id', 'ticker', 'start_date', 'end_date', 'total_return',
            'annualized_return', 'volatility', 'max_drawdown', 'sharpe_ratio',
            'sortino_ratio', 'information_ratio', 'var_95', 'cvar_95',
            'beta', 'alpha', 'tracking_error', 'created_timestamp'
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert PerformanceMetrics instance to dictionary for CSV storage"""
        return {
            'metrics_id': self.metrics_id,
            'ticker': self.ticker,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'information_ratio': self.information_ratio,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'beta': self.beta,
            'alpha': self.alpha,
            'tracking_error': self.tracking_error,
            'created_timestamp': self.created_timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetrics':
        """Create PerformanceMetrics instance from dictionary (CSV row)"""
        return cls(
            metrics_id=data['metrics_id'],
            ticker=data['ticker'],
            start_date=date.fromisoformat(data['start_date']),
            end_date=date.fromisoformat(data['end_date']),
            total_return=float(data['total_return']),
            annualized_return=float(data['annualized_return']),
            volatility=float(data['volatility']),
            max_drawdown=float(data['max_drawdown']),
            sharpe_ratio=float(data['sharpe_ratio']),
            sortino_ratio=float(data['sortino_ratio']),
            information_ratio=float(data['information_ratio']) if data.get('information_ratio') else None,
            var_95=float(data['var_95']) if data.get('var_95') else None,
            cvar_95=float(data['cvar_95']) if data.get('cvar_95') else None,
            beta=float(data['beta']) if data.get('beta') else None,
            alpha=float(data['alpha']) if data.get('alpha') else None,
            tracking_error=float(data['tracking_error']) if data.get('tracking_error') else None,
            created_timestamp=datetime.fromisoformat(data['created_timestamp'])
        )

class DataManager:
    """
    Data manager class for handling CSV file operations and data persistence
    """
    
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.data_dir,
            os.path.join(self.data_dir, "prices"),
            os.path.join(self.data_dir, "fundamentals"),
            os.path.join(self.data_dir, "analysis"),
            os.path.join(self.data_dir, "performance")
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def save_stocks(self, stocks: List[Stock]) -> None:
        """Save stock data to CSV file"""
        df = pd.DataFrame([stock.to_dict() for stock in stocks])
        df.to_csv(os.path.join(self.data_dir, "stocks.csv"), index=False)
    
    def load_stocks(self) -> List[Stock]:
        """Load stock data from CSV file"""
        file_path = os.path.join(self.data_dir, "stocks.csv")
        if not os.path.exists(file_path):
            return []
        
        df = pd.read_csv(file_path)
        return [Stock.from_dict(row.to_dict()) for _, row in df.iterrows()]
    
    def save_price_data(self, ticker: str, price_data: List[PriceData]) -> None:
        """Save price data to CSV file (organized by ticker and year)"""
        if not price_data:
            return
        
        # Group by year for efficient storage
        by_year = {}
        for data in price_data:
            year = data.date.year
            if year not in by_year:
                by_year[year] = []
            by_year[year].append(data)
        
        # Save each year separately
        for year, year_data in by_year.items():
            df = pd.DataFrame([data.to_dict() for data in year_data])
            file_path = os.path.join(self.data_dir, "prices", f"{ticker}_{year}.csv")
            df.to_csv(file_path, index=False)
    
    def load_price_data(self, ticker: str, start_date: Optional[date] = None, 
                       end_date: Optional[date] = None) -> List[PriceData]:
        """Load price data from CSV files"""
        price_data = []
        prices_dir = os.path.join(self.data_dir, "prices")
        
        # Find all files for the ticker
        for filename in os.listdir(prices_dir):
            if filename.startswith(f"{ticker}_") and filename.endswith(".csv"):
                file_path = os.path.join(prices_dir, filename)
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    data = PriceData.from_dict(row.to_dict())
                    
                    # Filter by date range if specified
                    if start_date and data.date < start_date:
                        continue
                    if end_date and data.date > end_date:
                        continue
                    
                    price_data.append(data)
        
        # Sort by date
        price_data.sort(key=lambda x: x.date)
        return price_data
    
    def save_fundamental_data(self, ticker: str, fundamental_data: List[FundamentalData]) -> None:
        """Save fundamental data to CSV file"""
        if not fundamental_data:
            return
        
        df = pd.DataFrame([data.to_dict() for data in fundamental_data])
        file_path = os.path.join(self.data_dir, "fundamentals", f"{ticker}.csv")
        df.to_csv(file_path, index=False)
    
    def load_fundamental_data(self, ticker: str) -> List[FundamentalData]:
        """Load fundamental data from CSV file"""
        file_path = os.path.join(self.data_dir, "fundamentals", f"{ticker}.csv")
        if not os.path.exists(file_path):
            return []
        
        df = pd.read_csv(file_path)
        return [FundamentalData.from_dict(row.to_dict()) for _, row in df.iterrows()]
    
    def save_analysis_results(self, results: List[AnalysisResult]) -> None:
        """Save analysis results to CSV file"""
        if not results:
            return
        
        df = pd.DataFrame([result.to_dict() for result in results])
        file_path = os.path.join(self.data_dir, "analysis", "results.csv")
        
        # Append to existing file if it exists
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        df.to_csv(file_path, index=False)
    
    def load_analysis_results(self, analysis_type: Optional[str] = None,
                            ticker: Optional[str] = None) -> List[AnalysisResult]:
        """Load analysis results from CSV file"""
        file_path = os.path.join(self.data_dir, "analysis", "results.csv")
        if not os.path.exists(file_path):
            return []
        
        df = pd.read_csv(file_path)
        
        # Filter by analysis type and ticker if specified
        if analysis_type:
            df = df[df['analysis_type'] == analysis_type]
        if ticker:
            df = df[df['ticker'] == ticker]
        
        return [AnalysisResult.from_dict(row.to_dict()) for _, row in df.iterrows()]
    
    def save_performance_metrics(self, metrics: List[PerformanceMetrics]) -> None:
        """Save performance metrics to CSV file"""
        if not metrics:
            return
        
        df = pd.DataFrame([metric.to_dict() for metric in metrics])
        file_path = os.path.join(self.data_dir, "performance", "metrics.csv")
        
        # Append to existing file if it exists
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        df.to_csv(file_path, index=False)
    
    def load_performance_metrics(self, ticker: Optional[str] = None) -> List[PerformanceMetrics]:
        """Load performance metrics from CSV file"""
        file_path = os.path.join(self.data_dir, "performance", "metrics.csv")
        if not os.path.exists(file_path):
            return []
        
        df = pd.read_csv(file_path)
        
        # Filter by ticker if specified
        if ticker:
            df = df[df['ticker'] == ticker]
        
        return [PerformanceMetrics.from_dict(row.to_dict()) for _, row in df.iterrows()]

if __name__ == "__main__":
    # Example usage and testing
    data_manager = DataManager()
    
    # Create sample data
    sample_stock = Stock(
        ticker="AAPL",
        company_name="Apple Inc.",
        sector="Technology",
        industry="Consumer Electronics",
        market_cap=3000000000000,
        exchange="NASDAQ"
    )
    
    print("Data models and CSV schemas defined successfully!")
    print(f"Stock CSV columns: {Stock.get_csv_columns()}")
    print(f"Price Data CSV columns: {PriceData.get_csv_columns()}")
    print(f"Sample stock data: {sample_stock.to_dict()}")

