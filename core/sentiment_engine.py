"""
Sentiment Analysis Engine for MarketPulse
Cazandra Aporbo, MS
May 2025

This module handles all sentiment analysis operations. I built this to understand
how financial news affects market movements. The approach is straightforward but
effective: weighted lexicon analysis with source credibility scoring.

No pretending to use BERT here. VADER works great for financial text when you
tune it right. Sometimes simple is better.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import defaultdict

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """
    Container for sentiment analysis results.
    Clean, typed, easy to work with.
    """
    text: str
    base_score: float  # -1 to 1
    confidence: float  # 0 to 1
    source: str
    weighted_score: float
    timestamp: datetime
    entities_detected: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'text': self.text,
            'base_score': self.base_score,
            'confidence': self.confidence,
            'source': self.source,
            'weighted_score': self.weighted_score,
            'timestamp': self.timestamp.isoformat(),
            'entities': self.entities_detected
        }


class FinancialLexicon:
    """
    Custom financial lexicon I built from analyzing thousands of headlines.
    These words actually move markets. Took weeks to refine these weights.
    """
    
    def __init__(self):
        # Positive indicators with weights
        self.positive = {
            # Earnings related
            'beat': 0.8, 'beats': 0.8, 'exceeded': 0.7, 'exceeds': 0.7,
            'outperform': 0.8, 'outperformed': 0.8, 'surprise': 0.6,
            
            # Growth indicators  
            'surge': 0.9, 'surged': 0.9, 'soar': 0.9, 'soared': 0.9,
            'rally': 0.8, 'rallied': 0.8, 'gain': 0.6, 'gained': 0.6,
            'rise': 0.5, 'rose': 0.5, 'climb': 0.6, 'climbed': 0.6,
            
            # Business health
            'strong': 0.6, 'robust': 0.7, 'record': 0.8, 'breakthrough': 0.8,
            'innovative': 0.6, 'expansion': 0.6, 'growth': 0.6,
            
            # Analyst actions
            'upgrade': 0.7, 'upgraded': 0.7, 'bullish': 0.8, 'buy': 0.6,
            'accumulate': 0.6, 'overweight': 0.5,
        }
        
        # Negative indicators
        self.negative = {
            # Earnings related
            'miss': -0.8, 'missed': -0.8, 'disappoints': -0.7, 
            'disappointed': -0.7, 'below': -0.6, 'weak': -0.7,
            
            # Decline indicators
            'plunge': -0.9, 'plunged': -0.9, 'crash': -0.95, 'crashed': -0.95,
            'tumble': -0.8, 'tumbled': -0.8, 'fall': -0.6, 'fell': -0.6,
            'drop': -0.7, 'dropped': -0.7, 'decline': -0.6, 'declined': -0.6,
            
            # Business problems
            'lawsuit': -0.6, 'investigation': -0.7, 'probe': -0.6,
            'recall': -0.8, 'bankruptcy': -0.95, 'fraud': -0.9,
            'layoffs': -0.7, 'cuts': -0.6, 'losses': -0.7,
            
            # Analyst actions
            'downgrade': -0.7, 'downgraded': -0.7, 'bearish': -0.8,
            'sell': -0.6, 'underweight': -0.5, 'avoid': -0.6,
        }
        
        # Context modifiers that change sentiment
        self.modifiers = {
            'not': -1.0,  # Reverses sentiment
            'despite': -0.5,  # Weakens following sentiment
            'but': -0.3,  # Indicates contrast
            'however': -0.3,
            'although': -0.2,
            'very': 1.5,  # Amplifies
            'extremely': 2.0,
            'slightly': 0.5,  # Dampens
            'somewhat': 0.6,
        }
        
        # Industry-specific terms
        self.sector_specific = {
            'tech': ['AI', 'cloud', 'software', 'hardware', 'chip', 'semiconductor'],
            'pharma': ['FDA', 'trial', 'approval', 'drug', 'clinical'],
            'finance': ['Fed', 'rates', 'yield', 'bonds', 'treasury'],
            'energy': ['oil', 'gas', 'renewable', 'solar', 'wind'],
        }
    
    def score_text(self, text: str) -> Tuple[float, float]:
        """
        Score text using the lexicon.
        Returns (sentiment_score, confidence)
        """
        text_lower = text.lower()
        words = text_lower.split()
        
        score = 0
        word_count = 0
        confidence_factors = []
        
        # Check for modifier context
        modifier_active = 1.0
        
        for i, word in enumerate(words):
            # Check if this word is a modifier
            if word in self.modifiers:
                modifier_active = self.modifiers[word]
                continue
            
            # Score positive words
            if word in self.positive:
                word_score = self.positive[word] * modifier_active
                score += word_score
                word_count += 1
                confidence_factors.append(abs(word_score))
                modifier_active = 1.0  # Reset modifier
                
            # Score negative words
            elif word in self.negative:
                word_score = self.negative[word] * modifier_active
                score += word_score
                word_count += 1
                confidence_factors.append(abs(word_score))
                modifier_active = 1.0  # Reset modifier
        
        # Normalize score
        if word_count > 0:
            final_score = np.tanh(score / np.sqrt(word_count))  # Bounded -1 to 1
            confidence = min(np.mean(confidence_factors), 1.0)
        else:
            final_score = 0.0
            confidence = 0.1  # Low confidence for neutral
            
        return final_score, confidence


class SourceCredibility:
    """
    Not all news sources are equal. Bloomberg moves markets more than
    random blogs. This class handles source weighting based on historical
    reliability and market impact.
    """
    
    def __init__(self):
        # Base credibility scores
        self.credibility_scores = {
            # Tier 1 - Most reliable financial sources
            'bloomberg': 1.2,
            'reuters': 1.0,
            'wsj': 1.1,
            'ft': 1.1,  # Financial Times
            'cnbc': 0.9,
            
            # Tier 2 - Good but sometimes sensational
            'marketwatch': 0.8,
            'yahoo': 0.7,
            'benzinga': 0.7,
            'seekingalpha': 0.6,
            
            # Tier 3 - Social and alternative
            'reddit': 0.3,
            'twitter': 0.3,
            'stocktwits': 0.2,
            
            # Default for unknown
            'unknown': 0.5
        }
        
        # Track source performance over time
        self.performance_history = defaultdict(list)
        
    def get_weight(self, source: str) -> float:
        """
        Get credibility weight for a source.
        """
        source_lower = source.lower()
        
        # Check for partial matches
        for key, weight in self.credibility_scores.items():
            if key in source_lower:
                return weight
                
        return self.credibility_scores['unknown']
    
    def update_performance(self, source: str, accuracy: float):
        """
        Update source credibility based on prediction accuracy.
        This is where the system learns which sources to trust.
        """
        self.performance_history[source].append(accuracy)
        
        # After 10 predictions, start adjusting weights
        if len(self.performance_history[source]) >= 10:
            avg_accuracy = np.mean(self.performance_history[source][-10:])
            
            # Adjust weight based on performance
            current_weight = self.get_weight(source)
            
            if avg_accuracy > 0.6:  # Better than random
                new_weight = current_weight * 1.05  # Increase trust
            elif avg_accuracy < 0.4:  # Worse than random
                new_weight = current_weight * 0.95  # Decrease trust
            else:
                new_weight = current_weight
                
            # Bound between 0.1 and 2.0
            new_weight = max(0.1, min(2.0, new_weight))
            self.credibility_scores[source.lower()] = new_weight
            
            logger.info(f"Updated {source} credibility: {current_weight:.2f} -> {new_weight:.2f}")


class SentimentEngine:
    """
    Main sentiment analysis engine.
    Combines lexicon analysis with source credibility for weighted sentiment scores.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.lexicon = FinancialLexicon()
        self.credibility = SourceCredibility()
        
        # Sentiment decay rate - older news matters less
        self.decay_rate = self.config.get('decay_rate', 0.95)  # Daily decay
        
        # Cache for efficiency
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
    def analyze(self, text: str, source: str = 'unknown', 
                timestamp: Optional[datetime] = None) -> SentimentResult:
        """
        Analyze sentiment of a piece of text.
        """
        # Check cache
        cache_key = f"{text[:50]}_{source}"
        if cache_key in self.cache:
            cached_result, cache_time = self.cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_ttl:
                return cached_result
        
        # Get base sentiment
        base_score, confidence = self.lexicon.score_text(text)
        
        # Apply source credibility weighting
        source_weight = self.credibility.get_weight(source)
        weighted_score = base_score * source_weight
        
        # Detect entities (simple approach - could use NER here)
        entities = self._detect_entities(text)
        
        # Create result
        result = SentimentResult(
            text=text,
            base_score=base_score,
            confidence=confidence,
            source=source,
            weighted_score=weighted_score,
            timestamp=timestamp or datetime.now(),
            entities_detected=entities
        )
        
        # Cache result
        self.cache[cache_key] = (result, datetime.now())
        
        return result
    
    def analyze_batch(self, articles: List[Dict]) -> pd.DataFrame:
        """
        Analyze multiple articles and return aggregated results.
        This is where the real power comes from - patterns in the noise.
        """
        results = []
        
        for article in articles:
            result = self.analyze(
                text=article.get('headline', ''),
                source=article.get('source', 'unknown'),
                timestamp=article.get('timestamp')
            )
            results.append(result.to_dict())
        
        df = pd.DataFrame(results)
        
        # Add temporal features
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Calculate sentiment momentum (this is key!)
            df['sentiment_ma_3'] = df['weighted_score'].rolling(3).mean()
            df['sentiment_ma_10'] = df['weighted_score'].rolling(10).mean()
            df['sentiment_momentum'] = df['sentiment_ma_3'] - df['sentiment_ma_10']
            
            # Sentiment acceleration (second derivative)
            df['sentiment_velocity'] = df['weighted_score'].diff()
            df['sentiment_acceleration'] = df['sentiment_velocity'].diff()
            
            # Volume of news (attention indicator)
            df['news_volume'] = df.groupby(df['timestamp'].dt.date).transform('count')['text']
            
        return df
    
    def _detect_entities(self, text: str) -> List[str]:
        """
        Simple entity detection. In production, you'd use spaCy or similar.
        For now, just looking for capital sequences and known patterns.
        """
        entities = []
        words = text.split()
        
        for word in words:
            # Ticker symbols (all caps, 1-5 letters)
            if word.isupper() and 1 <= len(word) <= 5 and word.isalpha():
                entities.append(word)
            # Dollar amounts
            elif '$' in word:
                entities.append(word)
                
        return list(set(entities))
    
    def calculate_market_sentiment(self, df: pd.DataFrame, 
                                  lookback_hours: int = 24) -> Dict:
        """
        Calculate overall market sentiment from news flow.
        This gives us the 'mood' of the market.
        """
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        recent_df = df[pd.to_datetime(df['timestamp']) > cutoff_time]
        
        if recent_df.empty:
            return {
                'overall_sentiment': 0,
                'sentiment_std': 0,
                'news_count': 0,
                'top_entities': [],
                'sentiment_trend': 'neutral'
            }
        
        # Apply time decay to older news
        recent_df = recent_df.copy()
        hours_old = (datetime.now() - pd.to_datetime(recent_df['timestamp'])).dt.total_seconds() / 3600
        decay_weights = self.decay_rate ** (hours_old / 24)
        
        # Weighted sentiment
        weighted_sentiment = (recent_df['weighted_score'] * decay_weights).sum() / decay_weights.sum()
        
        # Sentiment volatility (market uncertainty)
        sentiment_std = recent_df['weighted_score'].std()
        
        # News volume
        news_count = len(recent_df)
        
        # Top mentioned entities
        all_entities = []
        for entities in recent_df['entities']:
            if isinstance(entities, list):
                all_entities.extend(entities)
        
        from collections import Counter
        entity_counts = Counter(all_entities)
        top_entities = entity_counts.most_common(5)
        
        # Sentiment trend
        if len(recent_df) >= 2:
            first_half_sentiment = recent_df.iloc[:len(recent_df)//2]['weighted_score'].mean()
            second_half_sentiment = recent_df.iloc[len(recent_df)//2:]['weighted_score'].mean()
            
            if second_half_sentiment > first_half_sentiment + 0.1:
                trend = 'improving'
            elif second_half_sentiment < first_half_sentiment - 0.1:
                trend = 'deteriorating'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'overall_sentiment': float(weighted_sentiment),
            'sentiment_std': float(sentiment_std),
            'news_count': news_count,
            'top_entities': top_entities,
            'sentiment_trend': trend
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize engine
    engine = SentimentEngine()
    
    # Test single headline
    test_headline = "Apple beats earnings expectations, stock soars to record high"
    result = engine.analyze(test_headline, source='bloomberg')
    
    print(f"Headline: {test_headline}")
    print(f"Base Score: {result.base_score:.3f}")
    print(f"Weighted Score: {result.weighted_score:.3f}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Entities: {result.entities_detected}")
    
    # Test batch analysis
    test_articles = [
        {
            'headline': 'Fed raises interest rates by 25 basis points',
            'source': 'reuters',
            'timestamp': datetime.now() - timedelta(hours=2)
        },
        {
            'headline': 'Tech stocks tumble on disappointing earnings',
            'source': 'cnbc',
            'timestamp': datetime.now() - timedelta(hours=1)
        },
        {
            'headline': 'Market rallies on strong jobs report',
            'source': 'bloomberg',
            'timestamp': datetime.now()
        }
    ]
    
    df = engine.analyze_batch(test_articles)
    market_sentiment = engine.calculate_market_sentiment(df)
    
    print("\nMarket Sentiment Analysis:")
    print(f"Overall Sentiment: {market_sentiment['overall_sentiment']:.3f}")
    print(f"Sentiment Trend: {market_sentiment['sentiment_trend']}")
    print(f"News Volume: {market_sentiment['news_count']}")
    print(f"Top Entities: {market_sentiment['top_entities']}")
