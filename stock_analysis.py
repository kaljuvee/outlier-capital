#!/usr/bin/env python3
"""
Stock Analysis and Machine Learning Components

This module implements outlier detection, similarity analysis, and predictive
modeling for stock selection and performance analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple, Any
import logging
from dataclasses import dataclass
import uuid

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import scipy.stats as stats

# Data models
from data_models import Stock, PriceData, FundamentalData, AnalysisResult, DataManager
from data_collection_fixed import DataCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OutlierResult:
    """Result of outlier detection analysis"""
    ticker: str
    outlier_score: float
    confidence: float
    rank: int
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    reasons: List[str]

@dataclass
class SimilarityResult:
    """Result of similarity analysis"""
    ticker: str
    similarity_score: float
    distance: float
    common_features: List[str]
    feature_differences: Dict[str, float]

class FeatureEngineer:
    """Feature engineering for stock analysis"""
    
    @staticmethod
    def calculate_technical_indicators(price_data: List[PriceData]) -> pd.DataFrame:
        """Calculate technical indicators from price data"""
        if not price_data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'date': data.date,
            'open': data.open_price,
            'high': data.high_price,
            'low': data.low_price,
            'close': data.close_price,
            'volume': data.volume
        } for data in price_data])
        
        df = df.sort_values('date').reset_index(drop=True)
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_change'] = df['close'] - df['open']
        df['price_range'] = df['high'] - df['low']
        df['body_size'] = abs(df['close'] - df['open'])
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ma_{window}_ratio'] = df['close'] / df[f'ma_{window}']
        
        # Volatility measures
        df['volatility_5'] = df['returns'].rolling(window=5).std()
        df['volatility_20'] = df['returns'].rolling(window=20).std()
        
        # Volume indicators
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']
        
        # Momentum indicators
        df['rsi'] = FeatureEngineer._calculate_rsi(df['close'])
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # Bollinger Bands
        bb_window = 20
        bb_std = 2
        df['bb_middle'] = df['close'].rolling(window=bb_window).mean()
        bb_std_val = df['close'].rolling(window=bb_window).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std_val * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std_val * bb_std)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def create_feature_matrix(stocks_data: Dict[str, Dict[str, Any]], 
                            lookback_days: int = 200) -> pd.DataFrame:
        """Create feature matrix for machine learning"""
        features = []
        
        for ticker, data in stocks_data.items():
            if not data['price_data']:
                continue
            
            # Calculate technical indicators
            tech_df = FeatureEngineer.calculate_technical_indicators(data['price_data'])
            
            if tech_df.empty or len(tech_df) < lookback_days:
                # Use available data if less than lookback_days
                if tech_df.empty or len(tech_df) < 50:  # Minimum 50 days
                    continue
                lookback_days_actual = min(lookback_days, len(tech_df))
            else:
                lookback_days_actual = lookback_days
            
            # Get recent data
            recent_data = tech_df.tail(lookback_days_actual)
            
            # Aggregate features
            feature_row = {
                'ticker': ticker,
                # Price features
                'total_return': (recent_data['close'].iloc[-1] / recent_data['close'].iloc[0]) - 1,
                'volatility': recent_data['returns'].std() * np.sqrt(252),
                'avg_volume': recent_data['volume'].mean(),
                'avg_price_range': recent_data['price_range'].mean(),
                
                # Technical indicators (latest values)
                'rsi': recent_data['rsi'].iloc[-1] if not recent_data['rsi'].isna().all() else 50,
                'ma_5_ratio': recent_data['ma_5_ratio'].iloc[-1] if not recent_data['ma_5_ratio'].isna().all() else 1,
                'ma_20_ratio': recent_data['ma_20_ratio'].iloc[-1] if not recent_data['ma_20_ratio'].isna().all() else 1,
                'bb_position': recent_data['bb_position'].iloc[-1] if not recent_data['bb_position'].isna().all() else 0.5,
                'momentum_20': recent_data['momentum_20'].iloc[-1] if not recent_data['momentum_20'].isna().all() else 0,
                'volume_ratio': recent_data['volume_ratio'].iloc[-1] if not recent_data['volume_ratio'].isna().all() else 1,
                
                # Statistical features
                'skewness': recent_data['returns'].skew(),
                'kurtosis': recent_data['returns'].kurtosis(),
                'max_drawdown': FeatureEngineer._calculate_max_drawdown(recent_data['close']),
                'sharpe_ratio': FeatureEngineer._calculate_sharpe_ratio(recent_data['returns']),
            }
            
            # Add fundamental features if available
            if data.get('fundamental_data'):
                fund_data = data['fundamental_data']
                feature_row.update({
                    'pe_ratio': fund_data.price_to_earnings or 0,
                    'pb_ratio': fund_data.price_to_book or 0,
                    'roe': fund_data.return_on_equity or 0,
                    'debt_to_equity': fund_data.debt_to_equity or 0,
                    'gross_margin': fund_data.gross_margin or 0,
                    'operating_margin': fund_data.operating_margin or 0,
                })
            else:
                # Fill with defaults if no fundamental data
                feature_row.update({
                    'pe_ratio': 0, 'pb_ratio': 0, 'roe': 0,
                    'debt_to_equity': 0, 'gross_margin': 0, 'operating_margin': 0
                })
            
            # Add stock info features
            if data.get('stock_info'):
                stock_info = data['stock_info']
                feature_row.update({
                    'market_cap': np.log(stock_info.market_cap) if stock_info.market_cap else 0,
                    'sector': stock_info.sector or 'Unknown',
                    'industry': stock_info.industry or 'Unknown'
                })
            else:
                feature_row.update({
                    'market_cap': 0, 'sector': 'Unknown', 'industry': 'Unknown'
                })
            
            features.append(feature_row)
        
        return pd.DataFrame(features)
    
    @staticmethod
    def _calculate_max_drawdown(prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min()
    
    @staticmethod
    def _calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if returns.std() == 0:
            return 0
        excess_returns = returns.mean() * 252 - risk_free_rate
        return excess_returns / (returns.std() * np.sqrt(252))

class OutlierDetector:
    """Outlier detection for identifying exceptional stocks"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.feature_importance = {}
    
    def detect_outliers(self, feature_matrix: pd.DataFrame, 
                       top_n: int = 10) -> List[OutlierResult]:
        """Detect outlier stocks based on performance and characteristics"""
        if feature_matrix.empty:
            return []
        
        # Prepare features for analysis
        numeric_features = feature_matrix.select_dtypes(include=[np.number]).drop(['ticker'], axis=1, errors='ignore')
        
        # Handle missing values
        numeric_features = numeric_features.fillna(numeric_features.median())
        
        # Scale features
        scaled_features = self.scaler.fit_transform(numeric_features)
        
        # Statistical outlier detection (Z-score based)
        z_scores = np.abs(stats.zscore(scaled_features, axis=0))
        statistical_outliers = np.mean(z_scores, axis=1)
        
        # Isolation Forest outlier detection
        isolation_scores = self.isolation_forest.fit_predict(scaled_features)
        isolation_anomaly_scores = self.isolation_forest.score_samples(scaled_features)
        
        # Performance-based outlier detection
        performance_scores = self._calculate_performance_scores(feature_matrix)
        
        # Combine scores
        combined_scores = (
            0.3 * statistical_outliers +
            0.3 * (-isolation_anomaly_scores) +  # Negative because lower scores indicate outliers
            0.4 * performance_scores
        )
        
        # Create results
        results = []
        for i, (idx, row) in enumerate(feature_matrix.iterrows()):
            ticker = row['ticker']
            
            # Calculate confidence based on consistency across methods
            confidence = self._calculate_confidence(
                statistical_outliers[i],
                isolation_anomaly_scores[i],
                performance_scores[i]
            )
            
            # Identify reasons for being an outlier
            reasons = self._identify_outlier_reasons(row, numeric_features.iloc[i])
            
            result = OutlierResult(
                ticker=ticker,
                outlier_score=combined_scores[i],
                confidence=confidence,
                rank=0,  # Will be set after sorting
                total_return=row.get('total_return', 0),
                annualized_return=row.get('total_return', 0),  # Simplified
                volatility=row.get('volatility', 0),
                sharpe_ratio=row.get('sharpe_ratio', 0),
                max_drawdown=row.get('max_drawdown', 0),
                reasons=reasons
            )
            results.append(result)
        
        # Sort by outlier score and assign ranks
        results.sort(key=lambda x: x.outlier_score, reverse=True)
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return results[:top_n]
    
    def _calculate_performance_scores(self, feature_matrix: pd.DataFrame) -> np.ndarray:
        """Calculate performance-based outlier scores"""
        # Focus on return, Sharpe ratio, and low drawdown
        returns = feature_matrix['total_return'].fillna(0)
        sharpe = feature_matrix['sharpe_ratio'].fillna(0)
        drawdown = feature_matrix['max_drawdown'].fillna(0)
        
        # Normalize to 0-1 scale
        return_scores = (returns - returns.min()) / (returns.max() - returns.min() + 1e-8)
        sharpe_scores = (sharpe - sharpe.min()) / (sharpe.max() - sharpe.min() + 1e-8)
        drawdown_scores = 1 - ((drawdown - drawdown.min()) / (drawdown.max() - drawdown.min() + 1e-8))
        
        # Combine with weights
        performance_scores = 0.4 * return_scores + 0.4 * sharpe_scores + 0.2 * drawdown_scores
        return performance_scores.values
    
    def _calculate_confidence(self, stat_score: float, iso_score: float, perf_score: float) -> float:
        """Calculate confidence in outlier detection"""
        # Higher confidence when multiple methods agree
        scores = [stat_score, -iso_score, perf_score]  # Normalize isolation score
        normalized_scores = [(s - min(scores)) / (max(scores) - min(scores) + 1e-8) for s in scores]
        
        # Confidence is higher when scores are consistent
        std_dev = np.std(normalized_scores)
        confidence = max(0, 1 - std_dev)
        return confidence
    
    def _identify_outlier_reasons(self, row: pd.Series, numeric_row: pd.Series) -> List[str]:
        """Identify specific reasons why a stock is an outlier"""
        reasons = []
        
        # Performance reasons
        if row.get('total_return', 0) > 0.5:
            reasons.append(f"Exceptional returns: {row['total_return']:.1%}")
        
        if row.get('sharpe_ratio', 0) > 2:
            reasons.append(f"High Sharpe ratio: {row['sharpe_ratio']:.2f}")
        
        if row.get('volatility', 0) < 0.15:
            reasons.append(f"Low volatility: {row['volatility']:.1%}")
        
        # Technical reasons
        if row.get('rsi', 50) > 70:
            reasons.append("Overbought (RSI > 70)")
        elif row.get('rsi', 50) < 30:
            reasons.append("Oversold (RSI < 30)")
        
        if row.get('momentum_20', 0) > 0.2:
            reasons.append("Strong momentum")
        
        # Fundamental reasons
        if row.get('pe_ratio', 0) > 0 and row.get('pe_ratio', 0) < 10:
            reasons.append("Low P/E ratio (value stock)")
        
        if row.get('roe', 0) > 0.2:
            reasons.append("High ROE")
        
        return reasons[:5]  # Limit to top 5 reasons

class SimilarityAnalyzer:
    """Stock similarity analysis using machine learning"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        self.nn_model = NearestNeighbors(n_neighbors=10, metric='euclidean')
        self.feature_names = []
    
    def find_similar_stocks(self, target_ticker: str, feature_matrix: pd.DataFrame,
                          n_similar: int = 5) -> List[SimilarityResult]:
        """Find stocks similar to the target ticker"""
        if feature_matrix.empty or target_ticker not in feature_matrix['ticker'].values:
            return []
        
        # Prepare features
        numeric_features = feature_matrix.select_dtypes(include=[np.number]).drop(['ticker'], axis=1, errors='ignore')
        self.feature_names = numeric_features.columns.tolist()
        
        # Handle missing values
        numeric_features = numeric_features.fillna(numeric_features.median())
        
        # Scale features
        scaled_features = self.scaler.fit_transform(numeric_features)
        
        # Apply PCA for dimensionality reduction
        pca_features = self.pca.fit_transform(scaled_features)
        
        # Fit nearest neighbors model
        self.nn_model.fit(pca_features)
        
        # Find target stock index
        target_idx = feature_matrix[feature_matrix['ticker'] == target_ticker].index[0]
        target_features = pca_features[target_idx].reshape(1, -1)
        
        # Find similar stocks
        distances, indices = self.nn_model.kneighbors(target_features, n_neighbors=n_similar+1)
        
        # Create results (exclude the target stock itself)
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0][1:], indices[0][1:])):
            similar_ticker = feature_matrix.iloc[idx]['ticker']
            
            # Calculate similarity score (inverse of distance)
            similarity_score = 1 / (1 + distance)
            
            # Identify common features and differences
            common_features, feature_diffs = self._analyze_feature_similarity(
                feature_matrix.iloc[target_idx],
                feature_matrix.iloc[idx]
            )
            
            result = SimilarityResult(
                ticker=similar_ticker,
                similarity_score=similarity_score,
                distance=distance,
                common_features=common_features,
                feature_differences=feature_diffs
            )
            results.append(result)
        
        return results
    
    def _analyze_feature_similarity(self, target_row: pd.Series, 
                                  similar_row: pd.Series) -> Tuple[List[str], Dict[str, float]]:
        """Analyze feature similarities and differences"""
        common_features = []
        feature_diffs = {}
        
        # Check categorical features
        if target_row.get('sector') == similar_row.get('sector'):
            common_features.append(f"Same sector: {target_row.get('sector')}")
        
        if target_row.get('industry') == similar_row.get('industry'):
            common_features.append(f"Same industry: {target_row.get('industry')}")
        
        # Check numerical features
        numerical_features = ['total_return', 'volatility', 'sharpe_ratio', 'pe_ratio', 'market_cap']
        
        for feature in numerical_features:
            if feature in target_row and feature in similar_row:
                target_val = target_row[feature]
                similar_val = similar_row[feature]
                
                if pd.notna(target_val) and pd.notna(similar_val):
                    if target_val != 0:
                        diff_pct = abs(target_val - similar_val) / abs(target_val)
                        feature_diffs[feature] = diff_pct
                        
                        if diff_pct < 0.1:  # Less than 10% difference
                            common_features.append(f"Similar {feature}")
        
        return common_features, feature_diffs

class PredictiveModeler:
    """Predictive modeling for stock performance"""
    
    def __init__(self):
        self.regression_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.classification_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_importance = {}
        self.is_trained = False
    
    def train_models(self, feature_matrix: pd.DataFrame) -> Dict[str, float]:
        """Train regression and classification models"""
        if feature_matrix.empty:
            return {}
        
        # Prepare features and targets
        X, y_reg, y_class = self._prepare_training_data(feature_matrix)
        
        if X.empty:
            return {}
        
        # Split data
        X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = train_test_split(
            X, y_reg, y_class, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train regression model (predict returns)
        self.regression_model.fit(X_train_scaled, y_reg_train)
        reg_predictions = self.regression_model.predict(X_test_scaled)
        reg_score = mean_squared_error(y_reg_test, reg_predictions)
        
        # Train classification model (predict performance category)
        self.classification_model.fit(X_train_scaled, y_class_train)
        class_predictions = self.classification_model.predict(X_test_scaled)
        class_score = accuracy_score(y_class_test, class_predictions)
        
        # Store feature importance
        self.feature_importance = {
            'regression': dict(zip(X.columns, self.regression_model.feature_importances_)),
            'classification': dict(zip(X.columns, self.classification_model.feature_importances_))
        }
        
        self.is_trained = True
        
        return {
            'regression_mse': reg_score,
            'classification_accuracy': class_score,
            'n_samples': len(X),
            'n_features': len(X.columns)
        }
    
    def predict_performance(self, feature_matrix: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Predict stock performance using trained models"""
        if not self.is_trained or feature_matrix.empty:
            return {}
        
        # Prepare features
        X = self._prepare_prediction_features(feature_matrix)
        
        if X.empty:
            return {}
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        reg_predictions = self.regression_model.predict(X_scaled)
        class_predictions = self.classification_model.predict(X_scaled)
        class_probabilities = self.classification_model.predict_proba(X_scaled)
        
        # Organize results
        results = {}
        for i, ticker in enumerate(feature_matrix['ticker']):
            results[ticker] = {
                'predicted_return': reg_predictions[i],
                'performance_category': class_predictions[i],
                'category_probabilities': dict(zip(
                    self.classification_model.classes_,
                    class_probabilities[i]
                ))
            }
        
        return results
    
    def _prepare_training_data(self, feature_matrix: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare training data for models"""
        # Select numerical features for training
        feature_cols = [col for col in feature_matrix.columns 
                       if col not in ['ticker', 'sector', 'industry'] and 
                       feature_matrix[col].dtype in ['int64', 'float64']]
        
        X = feature_matrix[feature_cols].fillna(0)
        
        # Regression target: total return
        y_reg = feature_matrix['total_return'].fillna(0)
        
        # Classification target: performance category
        returns = feature_matrix['total_return'].fillna(0)
        y_class = pd.cut(returns, 
                        bins=[-np.inf, -0.1, 0.1, 0.3, np.inf],
                        labels=['Poor', 'Average', 'Good', 'Excellent'])
        
        return X, y_reg, y_class
    
    def _prepare_prediction_features(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction"""
        feature_cols = [col for col in feature_matrix.columns 
                       if col not in ['ticker', 'sector', 'industry'] and 
                       feature_matrix[col].dtype in ['int64', 'float64']]
        
        return feature_matrix[feature_cols].fillna(0)
    
    def get_feature_importance(self, model_type: str = 'both') -> Dict[str, float]:
        """Get feature importance from trained models"""
        if not self.is_trained:
            return {}
        
        if model_type == 'regression':
            return self.feature_importance.get('regression', {})
        elif model_type == 'classification':
            return self.feature_importance.get('classification', {})
        else:
            # Combine both models
            reg_importance = self.feature_importance.get('regression', {})
            class_importance = self.feature_importance.get('classification', {})
            
            combined = {}
            all_features = set(reg_importance.keys()) | set(class_importance.keys())
            
            for feature in all_features:
                reg_imp = reg_importance.get(feature, 0)
                class_imp = class_importance.get(feature, 0)
                combined[feature] = (reg_imp + class_imp) / 2
            
            return combined

# Example usage and testing
if __name__ == "__main__":
    # Test the analysis components
    POLYGON_API_KEY = "3lKo1IgQ3hXMjMCkmbQACTJySZHkfld7"
    
    print("Testing stock analysis components...")
    
    # Initialize components
    collector = DataCollector(POLYGON_API_KEY)
    feature_engineer = FeatureEngineer()
    outlier_detector = OutlierDetector()
    similarity_analyzer = SimilarityAnalyzer()
    predictive_modeler = PredictiveModeler()
    
    # Test with sample data
    test_tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    end_date = date.today()
    start_date = end_date - timedelta(days=365)  # 1 year of data
    
    print(f"Collecting data for {len(test_tickers)} stocks...")
    
    # Collect data for all tickers
    stocks_data = {}
    for ticker in test_tickers:
        print(f"Processing {ticker}...")
        
        stock_info = collector.yfinance_adapter.get_stock_info(ticker)
        price_data = collector.yfinance_adapter.get_price_data(ticker, start_date, end_date)
        fundamental_data = collector.yfinance_adapter.get_fundamental_data(ticker)
        
        stocks_data[ticker] = {
            'stock_info': stock_info,
            'price_data': price_data,
            'fundamental_data': fundamental_data
        }
    
    # Create feature matrix
    print("Creating feature matrix...")
    feature_matrix = feature_engineer.create_feature_matrix(stocks_data)
    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"Features: {list(feature_matrix.columns)}")
    
    if not feature_matrix.empty:
        # Test outlier detection
        print("\nTesting outlier detection...")
        outliers = outlier_detector.detect_outliers(feature_matrix, top_n=3)
        for outlier in outliers:
            print(f"Outlier: {outlier.ticker}, Score: {outlier.outlier_score:.3f}, "
                  f"Return: {outlier.total_return:.2%}, Reasons: {outlier.reasons}")
        
        # Test similarity analysis
        if len(feature_matrix) > 1:
            print("\nTesting similarity analysis...")
            target_ticker = feature_matrix['ticker'].iloc[0]
            similar_stocks = similarity_analyzer.find_similar_stocks(target_ticker, feature_matrix, n_similar=2)
            for similar in similar_stocks:
                print(f"Similar to {target_ticker}: {similar.ticker}, "
                      f"Similarity: {similar.similarity_score:.3f}, "
                      f"Common: {similar.common_features}")
        
        # Test predictive modeling
        print("\nTesting predictive modeling...")
        model_metrics = predictive_modeler.train_models(feature_matrix)
        print(f"Model metrics: {model_metrics}")
        
        if predictive_modeler.is_trained:
            predictions = predictive_modeler.predict_performance(feature_matrix)
            for ticker, pred in predictions.items():
                print(f"Prediction for {ticker}: Return={pred['predicted_return']:.2%}, "
                      f"Category={pred['performance_category']}")
            
            # Feature importance
            importance = predictive_modeler.get_feature_importance()
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"\nTop 5 important features: {top_features}")
    
    print("\nStock analysis testing completed!")

