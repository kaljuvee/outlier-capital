#!/usr/bin/env python3
"""
Performance Metrics and Backtesting Engine

This module implements comprehensive performance metrics calculation and
backtesting simulation for stock selection strategies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple, Any
import logging
from dataclasses import dataclass
import uuid

# Data models
from data_models import Stock, PriceData, PerformanceMetrics, DataManager
from data_collection_fixed import DataCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """Result of backtesting simulation"""
    strategy_name: str
    start_date: date
    end_date: date
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    volatility: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    information_ratio: float
    var_95: float
    cvar_95: float
    beta: float
    alpha: float
    tracking_error: float
    win_rate: float
    profit_factor: float
    max_consecutive_losses: int
    trades: List[Dict[str, Any]]
    daily_returns: pd.Series
    portfolio_values: pd.Series

@dataclass
class Trade:
    """Individual trade record"""
    ticker: str
    entry_date: date
    exit_date: Optional[date]
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    trade_type: str  # 'long' or 'short'
    pnl: Optional[float]
    return_pct: Optional[float]

class PerformanceCalculator:
    """Calculate comprehensive performance metrics"""
    
    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        """Calculate returns from price series"""
        return prices.pct_change().dropna()
    
    @staticmethod
    def calculate_total_return(start_price: float, end_price: float) -> float:
        """Calculate total return"""
        return (end_price - start_price) / start_price
    
    @staticmethod
    def calculate_annualized_return(total_return: float, days: int) -> float:
        """Calculate annualized return"""
        years = days / 365.25
        if years <= 0:
            return 0
        return (1 + total_return) ** (1 / years) - 1
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, annualized: bool = True) -> float:
        """Calculate volatility (standard deviation of returns)"""
        vol = returns.std()
        if annualized:
            vol *= np.sqrt(252)  # Assuming 252 trading days per year
        return vol
    
    @staticmethod
    def calculate_max_drawdown(prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min()
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if returns.std() == 0:
            return 0
        excess_returns = returns.mean() * 252 - risk_free_rate
        return excess_returns / (returns.std() * np.sqrt(252))
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0
        excess_returns = returns.mean() * 252 - risk_free_rate
        downside_deviation = downside_returns.std() * np.sqrt(252)
        return excess_returns / downside_deviation
    
    @staticmethod
    def calculate_information_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate information ratio"""
        if len(returns) != len(benchmark_returns):
            return 0
        
        excess_returns = returns - benchmark_returns
        if excess_returns.std() == 0:
            return 0
        
        return (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252))
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = PerformanceCalculator.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    @staticmethod
    def calculate_beta(returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta relative to market"""
        if len(returns) != len(market_returns) or market_returns.var() == 0:
            return 0
        
        covariance = np.cov(returns, market_returns)[0, 1]
        market_variance = market_returns.var()
        return covariance / market_variance
    
    @staticmethod
    def calculate_alpha(returns: pd.Series, market_returns: pd.Series, 
                       risk_free_rate: float = 0.02) -> float:
        """Calculate alpha (Jensen's alpha)"""
        if len(returns) != len(market_returns):
            return 0
        
        beta = PerformanceCalculator.calculate_beta(returns, market_returns)
        portfolio_return = returns.mean() * 252
        market_return = market_returns.mean() * 252
        
        expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
        return portfolio_return - expected_return
    
    @staticmethod
    def calculate_tracking_error(returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate tracking error"""
        if len(returns) != len(benchmark_returns):
            return 0
        
        excess_returns = returns - benchmark_returns
        return excess_returns.std() * np.sqrt(252)
    
    @staticmethod
    def calculate_comprehensive_metrics(prices: pd.Series, 
                                      benchmark_prices: Optional[pd.Series] = None,
                                      risk_free_rate: float = 0.02) -> Dict[str, float]:
        """Calculate all performance metrics"""
        returns = PerformanceCalculator.calculate_returns(prices)
        
        if benchmark_prices is not None:
            benchmark_returns = PerformanceCalculator.calculate_returns(benchmark_prices)
        else:
            benchmark_returns = None
        
        start_price = prices.iloc[0]
        end_price = prices.iloc[-1]
        days = len(prices)
        
        total_return = PerformanceCalculator.calculate_total_return(start_price, end_price)
        annualized_return = PerformanceCalculator.calculate_annualized_return(total_return, days)
        volatility = PerformanceCalculator.calculate_volatility(returns)
        max_drawdown = PerformanceCalculator.calculate_max_drawdown(prices)
        sharpe_ratio = PerformanceCalculator.calculate_sharpe_ratio(returns, risk_free_rate)
        sortino_ratio = PerformanceCalculator.calculate_sortino_ratio(returns, risk_free_rate)
        var_95 = PerformanceCalculator.calculate_var(returns, 0.95)
        cvar_95 = PerformanceCalculator.calculate_cvar(returns, 0.95)
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95
        }
        
        # Add benchmark-relative metrics if benchmark is provided
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            information_ratio = PerformanceCalculator.calculate_information_ratio(returns, benchmark_returns)
            beta = PerformanceCalculator.calculate_beta(returns, benchmark_returns)
            alpha = PerformanceCalculator.calculate_alpha(returns, benchmark_returns, risk_free_rate)
            tracking_error = PerformanceCalculator.calculate_tracking_error(returns, benchmark_returns)
            
            metrics.update({
                'information_ratio': information_ratio,
                'beta': beta,
                'alpha': alpha,
                'tracking_error': tracking_error
            })
        
        return metrics

class BacktestEngine:
    """Backtesting engine for strategy simulation"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.data_manager = DataManager()
        self.performance_calculator = PerformanceCalculator()
    
    def run_backtest(self, strategy_config: Dict[str, Any], 
                    start_date: date, end_date: date) -> BacktestResult:
        """Run a complete backtesting simulation"""
        
        strategy_name = strategy_config.get('name', 'Unknown Strategy')
        tickers = strategy_config.get('tickers', [])
        rebalance_frequency = strategy_config.get('rebalance_frequency', 'monthly')  # daily, weekly, monthly, quarterly
        position_sizing = strategy_config.get('position_sizing', 'equal_weight')  # equal_weight, market_cap_weight, custom
        transaction_cost = strategy_config.get('transaction_cost', 0.001)  # 0.1% per trade
        
        logger.info(f"Running backtest for {strategy_name} from {start_date} to {end_date}")
        
        # Load price data for all tickers
        price_data = self._load_price_data(tickers, start_date, end_date)
        
        if not price_data:
            logger.error("No price data available for backtesting")
            return self._create_empty_result(strategy_name, start_date, end_date)
        
        # Create trading calendar
        trading_dates = self._create_trading_calendar(price_data, start_date, end_date)
        rebalance_dates = self._get_rebalance_dates(trading_dates, rebalance_frequency)
        
        # Initialize portfolio
        portfolio = Portfolio(self.initial_capital, transaction_cost)
        trades = []
        daily_values = []
        
        # Run simulation
        for i, current_date in enumerate(trading_dates):
            # Get current prices
            current_prices = self._get_prices_for_date(price_data, current_date)
            
            # Update portfolio value
            portfolio.update_value(current_prices, current_date)
            daily_values.append({
                'date': current_date,
                'value': portfolio.total_value,
                'cash': portfolio.cash,
                'positions_value': portfolio.positions_value
            })
            
            # Check if it's a rebalancing date
            if current_date in rebalance_dates:
                # Generate trading signals (simplified - in practice this would use ML models)
                signals = self._generate_signals(tickers, current_prices, strategy_config)
                
                # Execute trades
                new_trades = portfolio.rebalance(signals, current_prices, current_date)
                trades.extend(new_trades)
        
        # Calculate performance metrics
        portfolio_values = pd.Series([dv['value'] for dv in daily_values], 
                                   index=[dv['date'] for dv in daily_values])
        
        # Load benchmark data (SPY as default)
        benchmark_data = self._load_benchmark_data(start_date, end_date)
        
        # Calculate comprehensive metrics
        metrics = self.performance_calculator.calculate_comprehensive_metrics(
            portfolio_values, benchmark_data
        )
        
        # Calculate additional trading metrics
        trading_metrics = self._calculate_trading_metrics(trades)
        
        # Create result
        result = BacktestResult(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=portfolio_values.iloc[-1],
            total_return=metrics['total_return'],
            annualized_return=metrics['annualized_return'],
            volatility=metrics['volatility'],
            max_drawdown=metrics['max_drawdown'],
            sharpe_ratio=metrics['sharpe_ratio'],
            sortino_ratio=metrics['sortino_ratio'],
            information_ratio=metrics.get('information_ratio', 0),
            var_95=metrics['var_95'],
            cvar_95=metrics['cvar_95'],
            beta=metrics.get('beta', 0),
            alpha=metrics.get('alpha', 0),
            tracking_error=metrics.get('tracking_error', 0),
            win_rate=trading_metrics['win_rate'],
            profit_factor=trading_metrics['profit_factor'],
            max_consecutive_losses=trading_metrics['max_consecutive_losses'],
            trades=[trade.__dict__ for trade in trades],
            daily_returns=portfolio_values.pct_change().dropna(),
            portfolio_values=portfolio_values
        )
        
        logger.info(f"Backtest completed. Total return: {result.total_return:.2%}, "
                   f"Sharpe ratio: {result.sharpe_ratio:.2f}")
        
        return result
    
    def _load_price_data(self, tickers: List[str], start_date: date, end_date: date) -> Dict[str, pd.DataFrame]:
        """Load price data for all tickers"""
        price_data = {}
        
        for ticker in tickers:
            try:
                # Load from CSV files
                price_records = self.data_manager.load_price_data(ticker, start_date, end_date)
                
                if price_records:
                    df = pd.DataFrame([{
                        'date': record.date,
                        'open': record.open_price,
                        'high': record.high_price,
                        'low': record.low_price,
                        'close': record.close_price,
                        'volume': record.volume
                    } for record in price_records])
                    
                    df = df.set_index('date').sort_index()
                    price_data[ticker] = df
                    
            except Exception as e:
                logger.warning(f"Could not load price data for {ticker}: {e}")
        
        return price_data
    
    def _load_benchmark_data(self, start_date: date, end_date: date) -> Optional[pd.Series]:
        """Load benchmark data (SPY)"""
        try:
            spy_records = self.data_manager.load_price_data('SPY', start_date, end_date)
            if spy_records:
                prices = pd.Series([record.close_price for record in spy_records],
                                 index=[record.date for record in spy_records])
                return prices.sort_index()
        except:
            pass
        return None
    
    def _create_trading_calendar(self, price_data: Dict[str, pd.DataFrame], 
                               start_date: date, end_date: date) -> List[date]:
        """Create trading calendar based on available data"""
        all_dates = set()
        for df in price_data.values():
            all_dates.update(df.index)
        
        trading_dates = sorted([d for d in all_dates if start_date <= d <= end_date])
        return trading_dates
    
    def _get_rebalance_dates(self, trading_dates: List[date], frequency: str) -> List[date]:
        """Get rebalancing dates based on frequency"""
        if frequency == 'daily':
            return trading_dates
        elif frequency == 'weekly':
            return [d for d in trading_dates if d.weekday() == 0]  # Mondays
        elif frequency == 'monthly':
            monthly_dates = []
            current_month = None
            for d in trading_dates:
                if current_month != d.month:
                    monthly_dates.append(d)
                    current_month = d.month
            return monthly_dates
        elif frequency == 'quarterly':
            quarterly_dates = []
            current_quarter = None
            for d in trading_dates:
                quarter = (d.month - 1) // 3
                if current_quarter != quarter:
                    quarterly_dates.append(d)
                    current_quarter = quarter
            return quarterly_dates
        else:
            return [trading_dates[0]]  # Just the first date
    
    def _get_prices_for_date(self, price_data: Dict[str, pd.DataFrame], 
                           target_date: date) -> Dict[str, float]:
        """Get closing prices for all tickers on a specific date"""
        prices = {}
        for ticker, df in price_data.items():
            if target_date in df.index:
                prices[ticker] = df.loc[target_date, 'close']
        return prices
    
    def _generate_signals(self, tickers: List[str], current_prices: Dict[str, float],
                         strategy_config: Dict[str, Any]) -> Dict[str, float]:
        """Generate trading signals (simplified implementation)"""
        # This is a simplified implementation
        # In practice, this would use the ML models from stock_analysis.py
        
        position_sizing = strategy_config.get('position_sizing', 'equal_weight')
        
        if position_sizing == 'equal_weight':
            # Equal weight allocation
            weight_per_stock = 1.0 / len(tickers)
            return {ticker: weight_per_stock for ticker in tickers if ticker in current_prices}
        else:
            # Default to equal weight
            weight_per_stock = 1.0 / len(tickers)
            return {ticker: weight_per_stock for ticker in tickers if ticker in current_prices}
    
    def _calculate_trading_metrics(self, trades: List[Trade]) -> Dict[str, float]:
        """Calculate trading-specific metrics"""
        if not trades:
            return {
                'win_rate': 0,
                'profit_factor': 0,
                'max_consecutive_losses': 0
            }
        
        completed_trades = [t for t in trades if t.pnl is not None]
        
        if not completed_trades:
            return {
                'win_rate': 0,
                'profit_factor': 0,
                'max_consecutive_losses': 0
            }
        
        # Win rate
        winning_trades = [t for t in completed_trades if t.pnl > 0]
        win_rate = len(winning_trades) / len(completed_trades)
        
        # Profit factor
        total_profits = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in completed_trades if t.pnl < 0))
        profit_factor = total_profits / total_losses if total_losses > 0 else 0
        
        # Max consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        for trade in completed_trades:
            if trade.pnl < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_consecutive_losses': max_consecutive_losses
        }
    
    def _create_empty_result(self, strategy_name: str, start_date: date, end_date: date) -> BacktestResult:
        """Create empty result when no data is available"""
        return BacktestResult(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=self.initial_capital,
            total_return=0,
            annualized_return=0,
            volatility=0,
            max_drawdown=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            information_ratio=0,
            var_95=0,
            cvar_95=0,
            beta=0,
            alpha=0,
            tracking_error=0,
            win_rate=0,
            profit_factor=0,
            max_consecutive_losses=0,
            trades=[],
            daily_returns=pd.Series(),
            portfolio_values=pd.Series()
        )

class Portfolio:
    """Portfolio management for backtesting"""
    
    def __init__(self, initial_capital: float, transaction_cost: float = 0.001):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # ticker -> quantity
        self.transaction_cost = transaction_cost
        self.total_value = initial_capital
        self.positions_value = 0
    
    def update_value(self, current_prices: Dict[str, float], current_date: date):
        """Update portfolio value based on current prices"""
        self.positions_value = sum(
            quantity * current_prices.get(ticker, 0)
            for ticker, quantity in self.positions.items()
        )
        self.total_value = self.cash + self.positions_value
    
    def rebalance(self, target_weights: Dict[str, float], 
                 current_prices: Dict[str, float], current_date: date) -> List[Trade]:
        """Rebalance portfolio to target weights"""
        trades = []
        
        # Calculate target values
        target_values = {ticker: weight * self.total_value 
                        for ticker, weight in target_weights.items()}
        
        # Calculate current values
        current_values = {ticker: self.positions.get(ticker, 0) * current_prices.get(ticker, 0)
                         for ticker in target_weights.keys()}
        
        # Generate trades
        for ticker in target_weights.keys():
            if ticker not in current_prices:
                continue
                
            current_value = current_values.get(ticker, 0)
            target_value = target_values[ticker]
            price = current_prices[ticker]
            
            # Calculate required trade
            value_diff = target_value - current_value
            
            if abs(value_diff) > self.total_value * 0.01:  # Only trade if difference > 1%
                quantity_diff = int(value_diff / price)
                
                if quantity_diff != 0:
                    # Execute trade
                    trade_value = abs(quantity_diff * price)
                    cost = trade_value * self.transaction_cost
                    
                    if quantity_diff > 0:  # Buy
                        if self.cash >= trade_value + cost:
                            self.cash -= (trade_value + cost)
                            self.positions[ticker] = self.positions.get(ticker, 0) + quantity_diff
                            
                            trade = Trade(
                                ticker=ticker,
                                entry_date=current_date,
                                exit_date=None,
                                entry_price=price,
                                exit_price=None,
                                quantity=quantity_diff,
                                trade_type='long',
                                pnl=None,
                                return_pct=None
                            )
                            trades.append(trade)
                    
                    else:  # Sell
                        current_quantity = self.positions.get(ticker, 0)
                        sell_quantity = min(abs(quantity_diff), current_quantity)
                        
                        if sell_quantity > 0:
                            self.cash += (sell_quantity * price - sell_quantity * price * self.transaction_cost)
                            self.positions[ticker] = current_quantity - sell_quantity
                            
                            trade = Trade(
                                ticker=ticker,
                                entry_date=current_date,
                                exit_date=current_date,
                                entry_price=price,  # Simplified - would need to track actual entry price
                                exit_price=price,
                                quantity=-sell_quantity,
                                trade_type='long',
                                pnl=0,  # Simplified calculation
                                return_pct=0
                            )
                            trades.append(trade)
        
        return trades

# Example usage and testing
if __name__ == "__main__":
    print("Testing performance engine...")
    
    # Test performance calculator
    calc = PerformanceCalculator()
    
    # Create sample price data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
    prices = pd.Series(100 * (1 + returns).cumprod(), index=dates)
    
    print("Testing performance metrics...")
    metrics = calc.calculate_comprehensive_metrics(prices)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Test backtesting engine
    print("\nTesting backtesting engine...")
    engine = BacktestEngine(initial_capital=100000)
    
    strategy_config = {
        'name': 'Test Strategy',
        'tickers': ['AAPL', 'GOOGL', 'MSFT'],
        'rebalance_frequency': 'monthly',
        'position_sizing': 'equal_weight',
        'transaction_cost': 0.001
    }
    
    start_date = date(2024, 1, 1)
    end_date = date(2024, 12, 31)
    
    # Note: This will only work if we have price data saved
    try:
        result = engine.run_backtest(strategy_config, start_date, end_date)
        print(f"Backtest result: {result.total_return:.2%} total return")
        print(f"Sharpe ratio: {result.sharpe_ratio:.2f}")
        print(f"Max drawdown: {result.max_drawdown:.2%}")
    except Exception as e:
        print(f"Backtest failed (expected if no data): {e}")
    
    print("Performance engine testing completed!")

