#!/usr/bin/env python3
"""
Data Collection Modules for Stock Selection Platform (Fixed Version)

This module provides data collection adapters for yfinance and Polygon.io APIs,
along with data validation and processing utilities.
"""

import yfinance as yf
from polygon import RESTClient
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple, Any
import time
import logging
from abc import ABC, abstractmethod

from data_models import Stock, PriceData, FundamentalData, DataManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollectionError(Exception):
    """Custom exception for data collection errors"""
    pass

class DataAdapter(ABC):
    """Abstract base class for data adapters"""
    
    @abstractmethod
    def get_stock_info(self, ticker: str) -> Optional[Stock]:
        """Get basic stock information"""
        pass
    
    @abstractmethod
    def get_price_data(self, ticker: str, start_date: date, end_date: date) -> List[PriceData]:
        """Get historical price data"""
        pass
    
    @abstractmethod
    def get_fundamental_data(self, ticker: str) -> Optional[FundamentalData]:
        """Get fundamental data"""
        pass

class YFinanceAdapter(DataAdapter):
    """Data adapter for Yahoo Finance via yfinance library"""
    
    def __init__(self, rate_limit_delay: float = 0.1):
        self.rate_limit_delay = rate_limit_delay
    
    def _rate_limit(self):
        """Apply rate limiting"""
        time.sleep(self.rate_limit_delay)
    
    def get_stock_info(self, ticker: str) -> Optional[Stock]:
        """Get basic stock information from Yahoo Finance"""
        try:
            self._rate_limit()
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.info
            
            if not info or 'symbol' not in info:
                logger.warning(f"No info found for ticker {ticker}")
                return None
            
            # Extract relevant information
            stock = Stock(
                ticker=ticker.upper(),
                company_name=info.get('longName', info.get('shortName', ticker)),
                sector=info.get('sector'),
                industry=info.get('industry'),
                market_cap=info.get('marketCap'),
                exchange=info.get('exchange'),
                currency=info.get('currency', 'USD'),
                country=info.get('country', 'US'),
                data_source='yfinance'
            )
            
            logger.info(f"Retrieved stock info for {ticker}")
            return stock
            
        except Exception as e:
            logger.error(f"Error getting stock info for {ticker}: {e}")
            return None
    
    def get_price_data(self, ticker: str, start_date: date, end_date: date) -> List[PriceData]:
        """Get historical price data from Yahoo Finance"""
        try:
            self._rate_limit()
            yf_ticker = yf.Ticker(ticker)
            
            # Download historical data
            hist = yf_ticker.history(
                start=start_date.isoformat(),
                end=(end_date + timedelta(days=1)).isoformat(),  # yfinance end is exclusive
                auto_adjust=False,
                prepost=False
            )
            
            if hist.empty:
                logger.warning(f"No price data found for {ticker}")
                return []
            
            price_data = []
            for date_idx, row in hist.iterrows():
                # Handle timezone-aware datetime index
                if hasattr(date_idx, 'date'):
                    price_date = date_idx.date()
                else:
                    price_date = date_idx
                
                # Skip if date is outside our range
                if price_date < start_date or price_date > end_date:
                    continue
                
                price_data.append(PriceData(
                    ticker=ticker.upper(),
                    date=price_date,
                    open_price=float(row['Open']),
                    high_price=float(row['High']),
                    low_price=float(row['Low']),
                    close_price=float(row['Close']),
                    adjusted_close=float(row['Adj Close']),
                    volume=int(row['Volume']),
                    dividend_amount=float(row.get('Dividends', 0.0)),
                    split_ratio=float(row.get('Stock Splits', 1.0)),
                    data_source='yfinance'
                ))
            
            logger.info(f"Retrieved {len(price_data)} price records for {ticker}")
            return price_data
            
        except Exception as e:
            logger.error(f"Error getting price data for {ticker}: {e}")
            return []
    
    def get_fundamental_data(self, ticker: str) -> Optional[FundamentalData]:
        """Get fundamental data from Yahoo Finance"""
        try:
            self._rate_limit()
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.info
            
            if not info:
                logger.warning(f"No fundamental data found for {ticker}")
                return None
            
            # Get financial statements
            try:
                financials = yf_ticker.financials
                balance_sheet = yf_ticker.balance_sheet
                cash_flow = yf_ticker.cashflow
            except:
                financials = balance_sheet = cash_flow = pd.DataFrame()
            
            # Extract fundamental metrics
            fundamental_data = FundamentalData(
                ticker=ticker.upper(),
                report_date=date.today(),  # Use current date as report date
                revenue=self._get_latest_value(financials, 'Total Revenue'),
                net_income=self._get_latest_value(financials, 'Net Income'),
                total_assets=self._get_latest_value(balance_sheet, 'Total Assets'),
                total_debt=self._get_latest_value(balance_sheet, 'Total Debt'),
                shareholders_equity=self._get_latest_value(balance_sheet, 'Stockholders Equity'),
                operating_cash_flow=self._get_latest_value(cash_flow, 'Operating Cash Flow'),
                free_cash_flow=self._get_latest_value(cash_flow, 'Free Cash Flow'),
                price_to_earnings=info.get('trailingPE'),
                price_to_book=info.get('priceToBook'),
                debt_to_equity=info.get('debtToEquity'),
                return_on_equity=info.get('returnOnEquity'),
                return_on_assets=info.get('returnOnAssets'),
                gross_margin=info.get('grossMargins'),
                operating_margin=info.get('operatingMargins'),
                net_margin=info.get('profitMargins'),
                data_source='yfinance'
            )
            
            logger.info(f"Retrieved fundamental data for {ticker}")
            return fundamental_data
            
        except Exception as e:
            logger.error(f"Error getting fundamental data for {ticker}: {e}")
            return None
    
    def _get_latest_value(self, df: pd.DataFrame, key: str) -> Optional[float]:
        """Extract the latest value from a financial statement DataFrame"""
        if df.empty or key not in df.index:
            return None
        
        try:
            # Get the most recent value (first column)
            value = df.loc[key].iloc[0]
            return float(value) if pd.notna(value) else None
        except:
            return None

class PolygonAdapter(DataAdapter):
    """Data adapter for Polygon.io API"""
    
    def __init__(self, api_key: str, rate_limit_delay: float = 0.1):
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.client = RESTClient(api_key=api_key)
    
    def _rate_limit(self):
        """Apply rate limiting"""
        time.sleep(self.rate_limit_delay)
    
    def get_stock_info(self, ticker: str) -> Optional[Stock]:
        """Get basic stock information from Polygon.io"""
        try:
            self._rate_limit()
            ticker_details = self.client.get_ticker_details(ticker.upper())
            
            if not ticker_details:
                logger.warning(f"No info found for ticker {ticker}")
                return None
            
            stock = Stock(
                ticker=ticker.upper(),
                company_name=ticker_details.name,
                sector=getattr(ticker_details, 'sic_description', None),
                industry=getattr(ticker_details, 'sic_description', None),
                market_cap=getattr(ticker_details, 'market_cap', None),
                exchange=getattr(ticker_details, 'primary_exchange', None),
                currency=getattr(ticker_details, 'currency_name', 'USD'),
                country='US',  # Polygon.io focuses on US markets
                active_status=getattr(ticker_details, 'active', True),
                data_source='polygon'
            )
            
            logger.info(f"Retrieved stock info for {ticker}")
            return stock
            
        except Exception as e:
            logger.error(f"Error getting stock info for {ticker}: {e}")
            return None
    
    def get_price_data(self, ticker: str, start_date: date, end_date: date) -> List[PriceData]:
        """Get historical price data from Polygon.io"""
        try:
            self._rate_limit()
            
            # Get aggregates (daily bars)
            aggs = self.client.get_aggs(
                ticker=ticker.upper(),
                multiplier=1,
                timespan="day",
                from_=start_date,
                to=end_date
            )
            
            if not aggs:
                logger.warning(f"No price data found for {ticker}")
                return []
            
            price_data = []
            for agg in aggs:
                # Convert timestamp to date
                price_date = datetime.fromtimestamp(agg.timestamp / 1000).date()
                
                price_data.append(PriceData(
                    ticker=ticker.upper(),
                    date=price_date,
                    open_price=float(agg.open),
                    high_price=float(agg.high),
                    low_price=float(agg.low),
                    close_price=float(agg.close),
                    adjusted_close=float(agg.close),  # Polygon doesn't provide adjusted close directly
                    volume=int(agg.volume),
                    data_source='polygon'
                ))
            
            logger.info(f"Retrieved {len(price_data)} price records for {ticker}")
            return price_data
            
        except Exception as e:
            logger.error(f"Error getting price data for {ticker}: {e}")
            return []
    
    def get_fundamental_data(self, ticker: str) -> Optional[FundamentalData]:
        """Get fundamental data from Polygon.io"""
        try:
            # Note: Polygon.io fundamental data requires higher tier subscription
            # For now, we'll return None and rely on yfinance for fundamentals
            logger.info(f"Fundamental data not available from Polygon.io for {ticker}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting fundamental data for {ticker}: {e}")
            return None

class DataValidator:
    """Data validation and quality control utilities"""
    
    @staticmethod
    def validate_price_data(price_data: List[PriceData]) -> Tuple[List[PriceData], List[str]]:
        """Validate price data and return cleaned data with error messages"""
        valid_data = []
        errors = []
        
        for data in price_data:
            # Check for reasonable price ranges
            if data.open_price <= 0 or data.high_price <= 0 or data.low_price <= 0 or data.close_price <= 0:
                errors.append(f"Invalid price data for {data.ticker} on {data.date}: negative or zero prices")
                continue
            
            # Check price relationships
            if data.high_price < data.low_price:
                errors.append(f"Invalid price data for {data.ticker} on {data.date}: high < low")
                continue
            
            if not (data.low_price <= data.open_price <= data.high_price):
                errors.append(f"Invalid price data for {data.ticker} on {data.date}: open price outside high/low range")
                continue
            
            if not (data.low_price <= data.close_price <= data.high_price):
                errors.append(f"Invalid price data for {data.ticker} on {data.date}: close price outside high/low range")
                continue
            
            # Check for extreme price movements (>50% in a day)
            if data.open_price > 0:
                daily_change = abs(data.close_price - data.open_price) / data.open_price
                if daily_change > 0.5:
                    errors.append(f"Extreme price movement for {data.ticker} on {data.date}: {daily_change:.2%}")
                    # Don't skip, just warn
            
            # Check volume
            if data.volume < 0:
                errors.append(f"Invalid volume for {data.ticker} on {data.date}: negative volume")
                continue
            
            valid_data.append(data)
        
        return valid_data, errors
    
    @staticmethod
    def validate_stock_info(stock: Stock) -> Tuple[bool, List[str]]:
        """Validate stock information"""
        errors = []
        
        if not stock.ticker or len(stock.ticker) < 1:
            errors.append("Invalid ticker symbol")
        
        if not stock.company_name:
            errors.append("Missing company name")
        
        if stock.market_cap is not None and stock.market_cap < 0:
            errors.append("Invalid market cap: negative value")
        
        return len(errors) == 0, errors

class DataCollector:
    """Main data collection orchestrator"""
    
    def __init__(self, polygon_api_key: str):
        self.yfinance_adapter = YFinanceAdapter()
        self.polygon_adapter = PolygonAdapter(polygon_api_key)
        self.data_manager = DataManager()
        self.validator = DataValidator()
    
    def collect_stock_universe(self, tickers: List[str]) -> Dict[str, Stock]:
        """Collect basic information for a universe of stocks"""
        stocks = {}
        
        for ticker in tickers:
            logger.info(f"Collecting stock info for {ticker}")
            
            # Try yfinance first
            stock = self.yfinance_adapter.get_stock_info(ticker)
            
            if stock:
                # Validate the stock info
                is_valid, errors = self.validator.validate_stock_info(stock)
                if is_valid:
                    stocks[ticker.upper()] = stock
                else:
                    logger.warning(f"Invalid stock info for {ticker}: {errors}")
            else:
                logger.warning(f"Could not retrieve stock info for {ticker}")
        
        # Save to CSV
        if stocks:
            self.data_manager.save_stocks(list(stocks.values()))
            logger.info(f"Saved {len(stocks)} stocks to CSV")
        
        return stocks
    
    def collect_price_data(self, ticker: str, start_date: date, end_date: date) -> List[PriceData]:
        """Collect price data for a single ticker"""
        logger.info(f"Collecting price data for {ticker} from {start_date} to {end_date}")
        
        # Collect from yfinance
        price_data = self.yfinance_adapter.get_price_data(ticker, start_date, end_date)
        
        # Validate the data
        if price_data:
            valid_data, errors = self.validator.validate_price_data(price_data)
            for error in errors:
                logger.warning(error)
            
            # Save to CSV
            if valid_data:
                self.data_manager.save_price_data(ticker, valid_data)
                logger.info(f"Saved {len(valid_data)} price records for {ticker}")
            
            return valid_data
        
        return []
    
    def collect_fundamental_data(self, ticker: str) -> Optional[FundamentalData]:
        """Collect fundamental data for a single ticker"""
        logger.info(f"Collecting fundamental data for {ticker}")
        
        # Use yfinance for fundamentals (Polygon requires higher tier)
        fundamental_data = self.yfinance_adapter.get_fundamental_data(ticker)
        
        if fundamental_data:
            # Save to CSV
            self.data_manager.save_fundamental_data(ticker, [fundamental_data])
            logger.info(f"Saved fundamental data for {ticker}")
        
        return fundamental_data

# Example usage and testing
if __name__ == "__main__":
    # Test the data collection system
    POLYGON_API_KEY = "3lKo1IgQ3hXMjMCkmbQACTJySZHkfld7"
    
    collector = DataCollector(POLYGON_API_KEY)
    
    # Test with a few sample tickers
    test_tickers = ['AAPL', 'GOOGL', 'MSFT']
    end_date = date.today()
    start_date = end_date - timedelta(days=30)
    
    print("Testing data collection system...")
    
    # Collect stock universe
    stocks = collector.collect_stock_universe(test_tickers)
    print(f"Collected info for {len(stocks)} stocks")
    
    # Collect price data for one stock
    if stocks:
        test_ticker = list(stocks.keys())[0]
        price_data = collector.collect_price_data(test_ticker, start_date, end_date)
        print(f"Collected {len(price_data)} price records for {test_ticker}")
    
    print("Data collection test completed!")

