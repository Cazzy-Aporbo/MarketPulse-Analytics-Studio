"""
Feature Engineering Factory for MarketPulse
Cazandra Aporbo, MS
May 2025

Feature engineering is where data science becomes art. This module creates
meaningful features from raw price, volume, and sentiment data. Started with
200+ features, narrowed down to these that actually predict something.

The key insight: it's not about having more features, it's about having the
right features that capture market dynamics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass 
class FeatureConfig:
    """
    Configuration for feature generation.
    Makes it easy to experiment with different settings.
    """
    # Window sizes for rolling calculations
    fast_window: int = 5
    medium_window: int = 20
    slow_window: int = 50
    
    # Lag periods for temporal features
    lag_periods: List[int] = None
    
    # Feature selection
    max_features: int = 50
    min_variance: float = 0.01
    
    # Sentiment parameters
    sentiment_windows: List[int] = None
    
    def __post_init__(self):
        if self.lag_periods is None:
            self.lag_periods = [1, 2, 3, 5, 10]
        if self.sentiment_windows is None:
            self.sentiment_windows = [3, 7, 14]


class TechnicalFeatures:
    """
    Technical indicators that actually work.
    No kitchen sink approach here - just indicators with proven predictive power.
    """
    
    @staticmethod
    def price_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Price-based features that capture different aspects of price movement.
        """
        features = pd.DataFrame(index=df.index)
        
        # Returns at different scales
        features['returns'] = df['Close'].pct_change()
        features['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        features['returns_squared'] = features['returns'] ** 2  # Volatility proxy
        
        # Price relative to recent history
        features['price_to_sma_20'] = df['Close'] / df['Close'].rolling(20).mean()
        features['price_to_sma_50'] = df['Close'] / df['Close'].rolling(50).mean()
        
        # High-Low spread (volatility measure)
        features['high_low_spread'] = (df['High'] - df['Low']) / df['Close']
        features['high_low_ratio'] = df['High'] / df['Low']
        
        # Gap analysis
        features['overnight_gap'] = df['Open'] / df['Close'].shift(1) - 1
        
        # Microstructure - where did we close in the day's range?
        # This tells us about buying/selling pressure
        features['close_location'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
        
        # Price acceleration (second derivative stuff)
        features['price_acceleration'] = features['returns'].diff()
        
        return features
    
    @staticmethod
    def volume_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Volume speaks louder than words in markets.
        These features capture different aspects of trading activity.
        """
        features = pd.DataFrame(index=df.index)
        
        # Basic volume metrics
        features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        features['volume_trend'] = df['Volume'].rolling(5).mean() / df['Volume'].rolling(20).mean()
        
        # Price-Volume interaction (follow the smart money)
        features['price_volume'] = df['Close'] * df['Volume']
        features['pv_ratio'] = features['price_volume'] / features['price_volume'].rolling(20).mean()
        
        # On-Balance Volume concept (simplified)
        # Volume confirms price movement
        features['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
        features['obv_sma_ratio'] = features['obv'] / features['obv'].rolling(20).mean()
        
        # Volume spikes often precede moves
        volume_mean = df['Volume'].rolling(20).mean()
        volume_std = df['Volume'].rolling(20).std()
        features['volume_zscore'] = (df['Volume'] - volume_mean) / (volume_std + 1e-10)
        
        return features
    
    @staticmethod
    def volatility_features(df: pd.DataFrame, returns_col: str = 'returns') -> pd.DataFrame:
        """
        Volatility is the price of admission to markets.
        These features help us understand and predict it.
        """
        features = pd.DataFrame(index=df.index)
        
        if returns_col not in df.columns:
            df[returns_col] = df['Close'].pct_change()
        
        # Historical volatility at different windows
        features['volatility_5'] = df[returns_col].rolling(5).std()
        features['volatility_20'] = df[returns_col].rolling(20).std()
        features['volatility_60'] = df[returns_col].rolling(60).std()
        
        # Volatility ratios (regime detection)
        features['vol_ratio_short_long'] = features['volatility_5'] / (features['volatility_20'] + 1e-10)
        
        # Parkinson volatility (using high-low)
        # More efficient than close-to-close volatility
        features['parkinson_vol'] = np.sqrt(
            np.log(df['High'] / df['Low']) ** 2 / (4 * np.log(2))
        ).rolling(20).mean()
        
        # ATR-style volatility
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features['atr_20'] = true_range.rolling(20).mean()
        features['atr_normalized'] = features['atr_20'] / df['Close']
        
        return features
    
    @staticmethod
    def momentum_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Momentum: the tendency of winning stocks to keep winning.
        Until they don't. These features capture both.
        """
        features = pd.DataFrame(index=df.index)
        
        # RSI - the classic
        features['rsi'] = TechnicalFeatures._calculate_rsi(df['Close'])
        features['rsi_oversold'] = (features['rsi'] < 30).astype(int)
        features['rsi_overbought'] = (features['rsi'] > 70).astype(int)
        
        # Rate of change at different scales
        for period in [5, 10, 20]:
            features[f'roc_{period}'] = df['Close'].pct_change(period)
        
        # MACD components (trend following)
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        features['macd'] = exp12 - exp26
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_diff'] = features['macd'] - features['macd_signal']
        
        # Mean reversion indicator
        sma_20 = df['Close'].rolling(20).mean()
        std_20 = df['Close'].rolling(20).std()
        features['mean_reversion_zscore'] = (df['Close'] - sma_20) / (std_20 + 1e-10)
        
        return features
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI the right way.
        Lots of implementations get the smoothing wrong.
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


class SentimentFeatures:
    """
    Transform raw sentiment into predictive features.
    The magic is in the temporal dynamics and cross-validation with price.
    """
    
    @staticmethod
    def create_sentiment_features(df: pd.DataFrame, 
                                sentiment_col: str = 'sentiment') -> pd.DataFrame:
        """
        Engineer features from sentiment data.
        """
        features = pd.DataFrame(index=df.index)
        
        if sentiment_col not in df.columns:
            logger.warning(f"Sentiment column {sentiment_col} not found")
            return features
        
        # Basic sentiment
        features['sentiment'] = df[sentiment_col]
        
        # Sentiment moving averages (smoothing noise)
        for window in [3, 7, 14]:
            features[f'sentiment_ma_{window}'] = df[sentiment_col].rolling(window).mean()
        
        # Sentiment momentum (key insight: acceleration matters)
        features['sentiment_momentum'] = (
            df[sentiment_col].rolling(3).mean() - 
            df[sentiment_col].rolling(10).mean()
        )
        
        # Sentiment velocity and acceleration
        features['sentiment_velocity'] = df[sentiment_col].diff()
        features['sentiment_acceleration'] = features['sentiment_velocity'].diff()
        
        # Sentiment volatility (uncertainty in news)
        features['sentiment_std'] = df[sentiment_col].rolling(10).std()
        
        # Sentiment extremes
        rolling_mean = df[sentiment_col].rolling(20).mean()
        rolling_std = df[sentiment_col].rolling(20).std()
        features['sentiment_zscore'] = (
            (df[sentiment_col] - rolling_mean) / (rolling_std + 1e-10)
        )
        
        # News volume features (if available)
        if 'news_count' in df.columns:
            features['news_volume'] = df['news_count']
            features['news_spike'] = (
                df['news_count'] / (df['news_count'].rolling(7).mean() + 1e-10)
            )
        
        return features
    
    @staticmethod
    def sentiment_price_interaction(df: pd.DataFrame) -> pd.DataFrame:
        """
        Features that capture how sentiment and price interact.
        This is where it gets interesting.
        """
        features = pd.DataFrame(index=df.index)
        
        if 'sentiment' not in df.columns or 'returns' not in df.columns:
            return features
        
        # Correlation between sentiment and returns
        features['sentiment_return_corr'] = (
            df['sentiment'].rolling(20)
            .corr(df['returns'])
        )
        
        # Sentiment-Volume interaction
        if 'Volume' in df.columns:
            features['sentiment_volume'] = df['sentiment'] * df['Volume']
            features['sent_vol_ratio'] = (
                features['sentiment_volume'] / 
                features['sentiment_volume'].rolling(20).mean()
            )
        
        # Divergence indicators
        # When sentiment and price disagree, opportunities arise
        price_direction = np.sign(df['returns'])
        sentiment_direction = np.sign(df['sentiment'])
        features['divergence'] = (price_direction != sentiment_direction).astype(int)
        features['divergence_streak'] = features['divergence'].groupby(
            (features['divergence'] != features['divergence'].shift()).cumsum()
        ).cumsum()
        
        return features


class FeatureFactory:
    """
    Main factory for creating all features.
    This orchestrates the entire feature engineering pipeline.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.feature_importance = {}
        
    def create_features(self, df: pd.DataFrame, 
                       include_sentiment: bool = True) -> pd.DataFrame:
        """
        Create all features from raw data.
        This is where everything comes together.
        """
        features = pd.DataFrame(index=df.index)
        
        # Technical features
        logger.info("Creating technical features...")
        features = pd.concat([
            features,
            TechnicalFeatures.price_features(df),
            TechnicalFeatures.volume_features(df),
            TechnicalFeatures.volatility_features(df),
            TechnicalFeatures.momentum_features(df)
        ], axis=1)
        
        # Sentiment features
        if include_sentiment and 'sentiment' in df.columns:
            logger.info("Creating sentiment features...")
            features = pd.concat([
                features,
                SentimentFeatures.create_sentiment_features(df),
                SentimentFeatures.sentiment_price_interaction(
                    pd.concat([df, features], axis=1)
                )
            ], axis=1)
        
        # Lag features (temporal dependencies)
        logger.info("Creating lag features...")
        features = self._create_lag_features(features)
        
        # Clean up
        features = self._clean_features(features)
        
        # Feature selection if we have too many
        if len(features.columns) > self.config.max_features:
            logger.info(f"Selecting top {self.config.max_features} features...")
            features = self._select_features(features)
        
        logger.info(f"Created {len(features.columns)} features")
        
        return features
    
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lagged versions of key features.
        Markets have memory, this captures it.
        """
        lag_features = pd.DataFrame(index=df.index)
        
        # Select features to lag (not all features benefit from lagging)
        features_to_lag = [
            'returns', 'sentiment', 'volume_ratio', 'volatility_20',
            'rsi', 'sentiment_momentum'
        ]
        
        for feature in features_to_lag:
            if feature in df.columns:
                for lag in self.config.lag_periods:
                    lag_features[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        return pd.concat([df, lag_features], axis=1)
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean up features: remove NaN, inf, and low variance features.
        """
        # Replace infinities
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill then backward fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Remove constant features (no predictive power)
        std = df.std()
        non_constant_features = std[std > self.config.min_variance].index
        df = df[non_constant_features]
        
        # Remove highly correlated features (redundant)
        df = self._remove_correlated_features(df)
        
        return df
    
    def _remove_correlated_features(self, df: pd.DataFrame, 
                                   threshold: float = 0.95) -> pd.DataFrame:
        """
        Remove highly correlated features to avoid multicollinearity.
        Keeps the first of each correlated pair.
        """
        corr_matrix = df.corr().abs()
        upper_triangle = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        # Find features to drop
        to_drop = [column for column in df.columns 
                  if any(corr_matrix[column][upper_triangle[:, df.columns.get_loc(column)]] > threshold)]
        
        if to_drop:
            logger.info(f"Removing {len(to_drop)} highly correlated features")
        
        return df.drop(columns=to_drop)
    
    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select top features based on variance.
        In production, you'd use mutual information or other methods.
        """
        # Calculate feature variance (normalized)
        feature_variance = df.var() / (df.mean() ** 2 + 1e-10)
        
        # Select top features
        top_features = feature_variance.nlargest(self.config.max_features).index
        
        # Store importance for later analysis
        self.feature_importance = feature_variance.to_dict()
        
        return df[top_features]
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names for model training.
        """
        return list(self.feature_importance.keys())
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        """
        return self.feature_importance


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    sample_data = pd.DataFrame({
        'Open': np.random.randn(len(dates)).cumsum() + 100,
        'High': np.random.randn(len(dates)).cumsum() + 102,
        'Low': np.random.randn(len(dates)).cumsum() + 98,
        'Close': np.random.randn(len(dates)).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, len(dates)),
        'sentiment': np.random.randn(len(dates)) * 0.3  # Sentiment between -1 and 1
    }, index=dates)
    
    # Make sure High is highest and Low is lowest
    sample_data['High'] = sample_data[['Open', 'High', 'Low', 'Close']].max(axis=1)
    sample_data['Low'] = sample_data[['Open', 'High', 'Low', 'Close']].min(axis=1)
    
    # Create features
    factory = FeatureFactory()
    features = factory.create_features(sample_data, include_sentiment=True)
    
    print(f"Created {len(features.columns)} features")
    print("\nTop 10 features by importance:")
    importance = factory.get_feature_importance()
    for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {feature}: {score:.4f}")
    
    print("\nFeature statistics:")
    print(features.describe())
