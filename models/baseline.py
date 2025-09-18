"""
Baseline Models: The Foundation of Intelligence
Cazandra Aporbo, MS
May 2025

These aren't just simple models. They're the foundation everything else builds on.
I learned the hard way that a well-tuned logistic regression often beats a
poorly configured neural network. Start simple, make it work, then add complexity.

Every model here has a specific purpose. Logistic regression gives me interpretable
baselines. Decision trees show me the rules. Naive Bayes handles text naturally.
Together, they form a diverse committee where each member's weakness is another's strength.

The secret isn't in having fancy models. It's in understanding when to use which one.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, log_loss
)
import warnings
from abc import ABC, abstractmethod
import json
import joblib

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class ModelPersonality:
    """
    Every model has a personality. Some are conservative, some aggressive.
    
    I use this to configure how each model behaves. It's not just hyperparameters,
    it's the model's entire approach to the problem. Think of it as the model's
    trading style.
    """
    
    risk_tolerance: float = 0.5  # 0 = conservative, 1 = aggressive
    confidence_threshold: float = 0.5  # Minimum confidence to make a prediction
    prefers_precision: bool = True  # True = fewer but accurate, False = catch everything
    learning_style: str = "balanced"  # "fast", "balanced", "thorough"
    
    # Performance expectations
    minimum_acceptable_accuracy: float = 0.52  # Better than random
    target_sharpe: float = 1.0  # Risk-adjusted performance target
    
    # Behavioral traits
    adapts_to_regime: bool = False  # Can it handle market regime changes?
    handles_imbalance: bool = False  # Good with imbalanced classes?
    interpretable: bool = True  # Can we understand its decisions?
    
    def to_dict(self) -> Dict:
        """Convert personality to config dict for models."""
        return {
            'class_weight': 'balanced' if self.handles_imbalance else None,
            'probability': True,  # Always want probabilities for ensembling
            'random_state': 42  # Reproducibility matters
        }


class ModelSoul(ABC):
    """
    The abstract soul that all models share.
    
    This is the interface that makes models interchangeable. Every model,
    no matter how complex, must implement these methods. It's like a contract
    that ensures compatibility across the ensemble.
    """
    
    def __init__(self, name: str, personality: Optional[ModelPersonality] = None):
        self.name = name
        self.personality = personality or ModelPersonality()
        self.birth_time = datetime.now()
        self.experience_points = 0  # Grows with training
        self.prediction_history = []
        self.performance_trajectory = []
        
        # Components every model needs
        self.scaler = None
        self.model = None
        self.is_calibrated = False
        
        # Metrics tracking
        self.lifetime_metrics = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'total_training_samples': 0,
            'training_iterations': 0,
            'best_accuracy': 0,
            'worst_accuracy': 1
        }
        
    @abstractmethod
    def awaken(self) -> None:
        """Initialize the model. Every model awakens differently."""
        pass
        
    @abstractmethod
    def meditate(self, X: pd.DataFrame, y: pd.Series) -> 'ModelSoul':
        """Train the model. Meditation leads to understanding."""
        pass
        
    @abstractmethod
    def divine(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions. Divine the future from patterns."""
        pass
        
    def introspect(self) -> Dict[str, Any]:
        """
        Look inward and report on internal state.
        Self-awareness is key to improvement.
        """
        age_days = (datetime.now() - self.birth_time).days
        wisdom_score = min(self.experience_points / 10000, 1.0)
        
        return {
            'name': self.name,
            'age_days': age_days,
            'wisdom_score': wisdom_score,
            'personality': self.personality.learning_style,
            'lifetime_accuracy': (self.lifetime_metrics['correct_predictions'] / 
                                 max(self.lifetime_metrics['total_predictions'], 1)),
            'best_performance': self.lifetime_metrics['best_accuracy'],
            'is_calibrated': self.is_calibrated
        }


class LogisticSage(ModelSoul):
    """
    The Logistic Sage: Master of Linear Wisdom
    
    Don't let the simplicity fool you. This model has predicted more market
    moves than any neural network. It's interpretable, robust, and fast.
    When you need a baseline that actually works, the Sage delivers.
    
    I call it Sage because it speaks in probabilities, not certainties.
    """
    
    def __init__(self, personality: Optional[ModelPersonality] = None):
        super().__init__("Logistic Sage", personality)
        
        # Sage-specific configuration
        self.regularization_path = []  # Track how regularization affects performance
        self.coefficient_memory = []  # Remember how coefficients evolve
        
    def awaken(self) -> None:
        """
        The Sage awakens with perfect balance between bias and variance.
        I use L2 regularization because markets are noisy and overfitting kills.
        """
        
        # Choose regularization based on personality
        if self.personality.risk_tolerance > 0.7:
            # Aggressive = less regularization
            C_value = 10.0
        elif self.personality.risk_tolerance < 0.3:
            # Conservative = more regularization
            C_value = 0.1
        else:
            # Balanced
            C_value = 1.0
            
        self.model = LogisticRegression(
            C=C_value,
            penalty='l2',
            solver='liblinear',  # Works well for small datasets
            max_iter=1000,
            class_weight='balanced' if self.personality.handles_imbalance else None,
            random_state=42
        )
        
        # Robust scaling for financial data (handles outliers better)
        self.scaler = RobustScaler()
        
        logger.info(f"{self.name} awakened with C={C_value}")
        
    def meditate(self, X: pd.DataFrame, y: pd.Series) -> 'LogisticSage':
        """
        Through meditation, the Sage finds the linear path through chaos.
        """
        
        if self.model is None:
            self.awaken()
            
        # Scale features (critical for logistic regression)
        X_scaled = self.scaler.fit_transform(X)
        
        # Train with timing
        start_time = datetime.now()
        self.model.fit(X_scaled, y)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Remember the coefficients (interpretability!)
        self.coefficient_memory.append({
            'timestamp': datetime.now(),
            'coefficients': self.model.coef_[0].tolist(),
            'intercept': self.model.intercept_[0]
        })
        
        # Update experience
        self.experience_points += len(X)
        self.lifetime_metrics['total_training_samples'] += len(X)
        self.lifetime_metrics['training_iterations'] += 1
        
        # Self-evaluation
        train_predictions = self.model.predict(X_scaled)
        train_accuracy = accuracy_score(y, train_predictions)
        
        self.lifetime_metrics['best_accuracy'] = max(
            self.lifetime_metrics['best_accuracy'], 
            train_accuracy
        )
        
        logger.info(
            f"{self.name} meditated for {training_time:.2f}s. "
            f"Training accuracy: {train_accuracy:.3f}"
        )
        
        return self
        
    def divine(self, X: pd.DataFrame) -> np.ndarray:
        """
        The Sage divines probabilities, not certainties.
        Markets are probabilistic, and so are good predictions.
        """
        
        if self.model is None:
            raise ValueError(f"{self.name} must meditate before divining")
            
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities
        probabilities = self.model.predict_proba(X_scaled)
        
        # Apply confidence threshold from personality
        predictions = (probabilities[:, 1] > self.personality.confidence_threshold).astype(int)
        
        # Track predictions for learning
        self.lifetime_metrics['total_predictions'] += len(predictions)
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'count': len(predictions),
            'positive_rate': predictions.mean()
        })
        
        return predictions
        
    def explain_thyself(self, feature_names: List[str]) -> pd.DataFrame:
        """
        The Sage can explain its reasoning. Transparency builds trust.
        """
        
        if not self.coefficient_memory:
            return pd.DataFrame()
            
        latest_coefficients = self.coefficient_memory[-1]['coefficients']
        
        explanation = pd.DataFrame({
            'feature': feature_names[:len(latest_coefficients)],
            'coefficient': latest_coefficients,
            'abs_importance': np.abs(latest_coefficients)
        })
        
        explanation = explanation.sort_values('abs_importance', ascending=False)
        
        # Add interpretation
        explanation['direction'] = explanation['coefficient'].apply(
            lambda x: 'increases' if x > 0 else 'decreases'
        )
        
        return explanation


class DecisionOracle(ModelSoul):
    """
    The Decision Oracle: Seer of Branching Paths
    
    Where the Sage sees lines, the Oracle sees trees. It finds the rules
    that govern markets: "If RSI > 70 AND volume spike, then reversal."
    These rules are gold for understanding market mechanics.
    
    I love decision trees because they think like traders think: in rules.
    """
    
    def __init__(self, personality: Optional[ModelPersonality] = None):
        super().__init__("Decision Oracle", personality)
        
        # Oracle-specific memory
        self.rule_library = []  # Collected trading rules
        self.split_history = []  # How the tree grows over time
        
    def awaken(self) -> None:
        """
        The Oracle awakens with the power to see all possible paths.
        I limit depth to prevent memorizing noise (overfitting is death).
        """
        
        # Depth based on personality
        if self.personality.learning_style == "fast":
            max_depth = 3  # Simple rules only
        elif self.personality.learning_style == "thorough":
            max_depth = 10  # Complex interactions
        else:
            max_depth = 5  # Balanced
            
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=20,  # Need enough samples for reliable splits
            min_samples_leaf=10,  # Avoid memorizing individual samples
            class_weight='balanced' if self.personality.handles_imbalance else None,
            random_state=42
        )
        
        # Trees don't need scaling but I do it for consistency
        self.scaler = StandardScaler()
        
        logger.info(f"{self.name} awakened with max_depth={max_depth}")
        
    def meditate(self, X: pd.DataFrame, y: pd.Series) -> 'DecisionOracle':
        """
        Through meditation, the Oracle sees the branching paths of fate.
        """
        
        if self.model is None:
            self.awaken()
            
        X_scaled = self.scaler.fit_transform(X)
        
        # Train and time
        start_time = datetime.now()
        self.model.fit(X_scaled, y)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Extract the learned rules (this is gold!)
        self._extract_rules(X.columns.tolist())
        
        # Update experience
        self.experience_points += len(X)
        self.lifetime_metrics['total_training_samples'] += len(X)
        self.lifetime_metrics['training_iterations'] += 1
        
        # Self-evaluation
        train_predictions = self.model.predict(X_scaled)
        train_accuracy = accuracy_score(y, train_predictions)
        
        logger.info(
            f"{self.name} found {self.model.get_n_leaves()} rules in {training_time:.2f}s. "
            f"Training accuracy: {train_accuracy:.3f}"
        )
        
        return self
        
    def divine(self, X: pd.DataFrame) -> np.ndarray:
        """
        The Oracle follows the rules to divine the future.
        Each prediction traces a path through the decision tree.
        """
        
        if self.model is None:
            raise ValueError(f"{self.name} must meditate before divining")
            
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Track for learning
        self.lifetime_metrics['total_predictions'] += len(predictions)
        
        # The Oracle also tracks which paths are most traveled
        leaf_indices = self.model.apply(X_scaled)
        unique_leaves, counts = np.unique(leaf_indices, return_counts=True)
        
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'count': len(predictions),
            'unique_paths': len(unique_leaves),
            'most_common_path': unique_leaves[np.argmax(counts)]
        })
        
        return predictions
        
    def _extract_rules(self, feature_names: List[str]) -> None:
        """
        Extract human-readable rules from the tree.
        This is where machine learning becomes trading wisdom.
        """
        
        tree = self.model.tree_
        
        def recurse(node, depth, path):
            """Recursively extract rules from tree nodes."""
            
            if tree.feature[node] != -2:  # Not a leaf
                feature = feature_names[tree.feature[node]]
                threshold = tree.threshold[node]
                
                # Left branch (<=)
                left_path = path + f" AND {feature} <= {threshold:.3f}"
                recurse(tree.children_left[node], depth + 1, left_path)
                
                # Right branch (>)
                right_path = path + f" AND {feature} > {threshold:.3f}"
                recurse(tree.children_right[node], depth + 1, right_path)
                
            else:  # Leaf node
                # Get the prediction for this leaf
                values = tree.value[node][0]
                prediction = np.argmax(values)
                confidence = values[prediction] / values.sum()
                
                if confidence > 0.6:  # Only keep confident rules
                    rule = {
                        'path': path[5:] if path.startswith(" AND ") else path,
                        'prediction': prediction,
                        'confidence': confidence,
                        'samples': int(tree.n_node_samples[node])
                    }
                    self.rule_library.append(rule)
        
        # Extract rules starting from root
        self.rule_library = []  # Clear previous rules
        recurse(0, 0, "")
        
        # Sort by confidence
        self.rule_library.sort(key=lambda x: x['confidence'], reverse=True)
        
    def share_wisdom(self, top_n: int = 5) -> List[Dict]:
        """
        The Oracle shares its most confident trading rules.
        These are the patterns it has divined from the data.
        """
        
        return self.rule_library[:top_n]


class BayesianMystic(ModelSoul):
    """
    The Bayesian Mystic: Master of Probabilistic Reasoning
    
    The Mystic doesn't just predict, it believes. Using Bayes' theorem,
    it updates beliefs as new evidence arrives. Perfect for markets where
    yesterday's truth is today's falsehood.
    
    I use this for sentiment analysis because text is naturally probabilistic.
    """
    
    def __init__(self, personality: Optional[ModelPersonality] = None):
        super().__init__("Bayesian Mystic", personality)
        
        # Mystic-specific attributes
        self.prior_beliefs = {}  # What the Mystic believed before seeing data
        self.posterior_beliefs = {}  # Updated beliefs after evidence
        self.surprise_log = []  # Track when reality defies expectations
        
    def awaken(self) -> None:
        """
        The Mystic awakens with no preconceptions, ready to learn.
        I use Gaussian Naive Bayes because features are roughly normal
        after standardization.
        """
        
        self.model = GaussianNB()
        
        # Mystic uses standard scaling
        self.scaler = StandardScaler()
        
        logger.info(f"{self.name} awakened with empty mind, ready to learn")
        
    def meditate(self, X: pd.DataFrame, y: pd.Series) -> 'BayesianMystic':
        """
        Through meditation, the Mystic updates its beliefs about reality.
        Each sample is evidence that shifts probabilities.
        """
        
        if self.model is None:
            self.awaken()
            
        X_scaled = self.scaler.fit_transform(X)
        
        # Capture prior beliefs (before training)
        if not self.prior_beliefs:
            self.prior_beliefs = {
                'class_balance': y.value_counts(normalize=True).to_dict(),
                'feature_means': X.mean().to_dict(),
                'feature_stds': X.std().to_dict()
            }
        
        # Train and time
        start_time = datetime.now()
        self.model.fit(X_scaled, y)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Capture posterior beliefs (after training)
        self.posterior_beliefs = {
            'class_priors': self.model.class_prior_.tolist(),
            'theta': self.model.theta_.tolist(),  # Feature means per class
            'sigma': self.model.sigma_.tolist()  # Feature variances per class
        }
        
        # Calculate information gain (how much we learned)
        prior_entropy = -sum(p * np.log(p + 1e-10) 
                           for p in self.prior_beliefs['class_balance'].values())
        posterior_entropy = -sum(p * np.log(p + 1e-10) 
                               for p in self.model.class_prior_)
        information_gain = prior_entropy - posterior_entropy
        
        # Update experience
        self.experience_points += int(information_gain * 1000)
        self.lifetime_metrics['total_training_samples'] += len(X)
        
        logger.info(
            f"{self.name} meditated for {training_time:.2f}s. "
            f"Information gain: {information_gain:.3f} bits"
        )
        
        return self
        
    def divine(self, X: pd.DataFrame) -> np.ndarray:
        """
        The Mystic divines by calculating posterior probabilities.
        It's not guessing, it's reasoning under uncertainty.
        """
        
        if self.model is None:
            raise ValueError(f"{self.name} must meditate before divining")
            
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities for decision making
        probabilities = self.model.predict_proba(X_scaled)
        
        # Apply personality-based thresholding
        if self.personality.prefers_precision:
            # Only predict when very confident
            predictions = np.where(
                probabilities.max(axis=1) > 0.7,
                probabilities.argmax(axis=1),
                -1  # Abstain when uncertain
            )
        else:
            # Standard prediction
            predictions = self.model.predict(X_scaled)
        
        # Track surprising predictions (where prior and posterior disagree)
        for i, (pred, prob) in enumerate(zip(predictions, probabilities.max(axis=1))):
            if prob > 0.8 and pred != np.argmax(self.model.class_prior_):
                self.surprise_log.append({
                    'timestamp': datetime.now(),
                    'index': i,
                    'prediction': pred,
                    'confidence': prob,
                    'surprise_factor': abs(prob - self.model.class_prior_[int(pred)])
                })
        
        # Update metrics
        self.lifetime_metrics['total_predictions'] += len(predictions)
        
        return predictions
        
    def contemplate_uncertainty(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        The Mystic contemplates its uncertainty about each prediction.
        Knowing what you don't know is wisdom.
        """
        
        if self.model is None:
            raise ValueError(f"{self.name} must meditate first")
            
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Calculate entropy (uncertainty) for each prediction
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
        
        # Maximum entropy for reference (complete uncertainty)
        max_entropy = -np.log(1.0 / len(self.model.classes_))
        
        uncertainty_df = pd.DataFrame({
            'prediction': probabilities.argmax(axis=1),
            'confidence': probabilities.max(axis=1),
            'entropy': entropy,
            'uncertainty_ratio': entropy / max_entropy,
            'should_abstain': entropy > (max_entropy * 0.8)  # Too uncertain
        })
        
        return uncertainty_df


class VelocityTracker(ModelSoul):
    """
    The Velocity Tracker: Momentum Specialist
    
    Markets have momentum. Trends persist until they don't. This model
    specializes in detecting and riding momentum while being ready to
    bail when the trend breaks.
    
    I built this because pure ML models miss the temporal dynamics.
    This one remembers where we've been and where we're going.
    """
    
    def __init__(self, personality: Optional[ModelPersonality] = None):
        super().__init__("Velocity Tracker", personality)
        
        # Velocity-specific state
        self.momentum_window = 20  # Look back period
        self.acceleration_threshold = 0.02  # When to recognize trend change
        self.regime_memory = []  # Remember market regimes
        
    def awaken(self) -> None:
        """
        The Tracker awakens with eyes on the horizon.
        I use Ridge regression because it handles collinear features well.
        """
        
        # Ridge for continuous predictions, then threshold
        self.model = Ridge(
            alpha=1.0 if self.personality.risk_tolerance < 0.5 else 0.1,
            random_state=42
        )
        
        self.scaler = StandardScaler()
        
        logger.info(f"{self.name} awakened, tracking momentum")
        
    def meditate(self, X: pd.DataFrame, y: pd.Series) -> 'VelocityTracker':
        """
        The Tracker learns the rhythm of the markets.
        It's not about single points, it's about trajectories.
        """
        
        if self.model is None:
            self.awaken()
            
        # Add momentum features if not present
        X_momentum = self._add_momentum_features(X)
        X_scaled = self.scaler.fit_transform(X_momentum)
        
        # Train on continuous targets (returns, not binary)
        # This captures magnitude, not just direction
        start_time = datetime.now()
        self.model.fit(X_scaled, y)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Detect current market regime
        predictions = self.model.predict(X_scaled)
        regime = self._detect_regime(predictions)
        self.regime_memory.append({
            'timestamp': datetime.now(),
            'regime': regime,
            'samples': len(X)
        })
        
        logger.info(
            f"{self.name} detected {regime} regime in {training_time:.2f}s"
        )
        
        return self
        
    def divine(self, X: pd.DataFrame) -> np.ndarray:
        """
        The Tracker divines by following the momentum.
        Where we're going depends on where we've been.
        """
        
        if self.model is None:
            raise ValueError(f"{self.name} must meditate before divining")
            
        X_momentum = self._add_momentum_features(X)
        X_scaled = self.scaler.transform(X_momentum)
        
        # Get continuous predictions
        momentum_scores = self.model.predict(X_scaled)
        
        # Convert to binary based on regime
        current_regime = self.regime_memory[-1]['regime'] if self.regime_memory else 'neutral'
        
        if current_regime == 'trending':
            # In trends, follow momentum
            predictions = (momentum_scores > 0).astype(int)
        elif current_regime == 'mean_reverting':
            # In mean reversion, fade extremes
            predictions = (momentum_scores < 0).astype(int)
        else:
            # Neutral regime, use threshold
            threshold = np.percentile(momentum_scores, 50)
            predictions = (momentum_scores > threshold).astype(int)
            
        return predictions
        
    def _add_momentum_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add velocity and acceleration features.
        Markets care about rate of change, not just level.
        """
        
        X_momentum = X.copy()
        
        # Add momentum indicators if they have price columns
        if 'returns' in X.columns:
            # Velocity (first derivative)
            X_momentum['velocity_5'] = X['returns'].rolling(5).mean()
            X_momentum['velocity_20'] = X['returns'].rolling(20).mean()
            
            # Acceleration (second derivative)
            X_momentum['acceleration'] = X_momentum['velocity_5'].diff()
            
            # Momentum strength
            X_momentum['momentum_strength'] = (
                X_momentum['velocity_5'] - X_momentum['velocity_20']
            )
            
        # Fill NaN values
        X_momentum = X_momentum.fillna(method='ffill').fillna(0)
        
        return X_momentum
        
    def _detect_regime(self, predictions: np.ndarray) -> str:
        """
        Detect if we're in trending, mean-reverting, or choppy markets.
        This changes everything about how to trade.
        """
        
        if len(predictions) < 20:
            return 'neutral'
            
        # Calculate autocorrelation
        autocorr = np.corrcoef(predictions[:-1], predictions[1:])[0, 1]
        
        # Calculate volatility of predictions
        pred_volatility = np.std(predictions)
        
        if autocorr > 0.3 and pred_volatility > 0.1:
            return 'trending'
        elif autocorr < -0.3:
            return 'mean_reverting'
        else:
            return 'choppy'


class ModelZoo:
    """
    The Zoo: Where All Models Live Together
    
    I manage a collection of diverse models, each with their own personality
    and specialty. The Zoo keeper (me) ensures they work together harmoniously,
    leveraging each model's strengths while covering for their weaknesses.
    
    This is where the magic of ensemble learning happens.
    """
    
    def __init__(self):
        self.inhabitants = {}  # The models that live here
        self.performance_ledger = {}  # Track everyone's performance
        self.feeding_schedule = {}  # When each model needs retraining
        
        # Initialize the core inhabitants
        self._populate_zoo()
        
    def _populate_zoo(self):
        """
        Stock the zoo with diverse models.
        Diversity is strength in ensemble learning.
        """
        
        # The wise one
        sage = LogisticSage(
            ModelPersonality(
                risk_tolerance=0.3,
                learning_style="balanced",
                interpretable=True
            )
        )
        self.inhabitants['sage'] = sage
        
        # The rule maker
        oracle = DecisionOracle(
            ModelPersonality(
                risk_tolerance=0.5,
                learning_style="fast",
                interpretable=True
            )
        )
        self.inhabitants['oracle'] = oracle
        
        # The believer
        mystic = BayesianMystic(
            ModelPersonality(
                risk_tolerance=0.4,
                prefers_precision=True,
                handles_imbalance=True
            )
        )
        self.inhabitants['mystic'] = mystic
        
        # The momentum rider
        tracker = VelocityTracker(
            ModelPersonality(
                risk_tolerance=0.7,
                learning_style="thorough",
                adapts_to_regime=True
            )
        )
        self.inhabitants['tracker'] = tracker
        
        logger.info(f"Zoo populated with {len(self.inhabitants)} models")
        
    def train_all(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Training day at the zoo. Everyone learns together.
        """
        
        performance = {}
        
        for name, model in self.inhabitants.items():
            try:
                model.meditate(X, y)
                
                # Evaluate
                predictions = model.divine(X)
                accuracy = accuracy_score(y, predictions[predictions != -1])  # Ignore abstentions
                
                performance[name] = accuracy
                self.performance_ledger[name] = {
                    'accuracy': accuracy,
                    'timestamp': datetime.now(),
                    'samples': len(X)
                }
                
                logger.info(f"{name} achieved {accuracy:.3f} accuracy")
                
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                performance[name] = 0
                
        return performance
        
    def ensemble_predict(self, X: pd.DataFrame, method: str = 'weighted') -> np.ndarray:
        """
        The zoo makes a collective decision.
        Many minds are better than one.
        """
        
        all_predictions = {}
        all_weights = {}
        
        for name, model in self.inhabitants.items():
            try:
                predictions = model.divine(X)
                all_predictions[name] = predictions
                
                # Weight based on recent performance
                weight = self.performance_ledger.get(name, {}).get('accuracy', 0.5)
                all_weights[name] = weight
                
            except Exception as e:
                logger.error(f"{name} failed to predict: {e}")
                
        if not all_predictions:
            raise ValueError("No models made predictions")
            
        # Ensemble methods
        if method == 'weighted':
            # Weighted average
            weighted_sum = np.zeros(len(X))
            weight_sum = 0
            
            for name, predictions in all_predictions.items():
                weight = all_weights[name]
                weighted_sum += predictions * weight
                weight_sum += weight
                
            final_predictions = (weighted_sum / weight_sum > 0.5).astype(int)
            
        elif method == 'majority':
            # Simple majority vote
            prediction_matrix = np.array(list(all_predictions.values()))
            final_predictions = (prediction_matrix.mean(axis=0) > 0.5).astype(int)
            
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
            
        return final_predictions
        
    def get_zoo_report(self) -> pd.DataFrame:
        """
        Generate a report on all inhabitants.
        How is everyone doing?
        """
        
        report_data = []
        
        for name, model in self.inhabitants.items():
            introspection = model.introspect()
            performance = self.performance_ledger.get(name, {})
            
            report_data.append({
                'model': name,
                'type': model.__class__.__name__,
                'personality': introspection['personality'],
                'wisdom_score': introspection['wisdom_score'],
                'recent_accuracy': performance.get('accuracy', None),
                'lifetime_accuracy': introspection['lifetime_accuracy'],
                'age_days': introspection['age_days']
            })
            
        return pd.DataFrame(report_data)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Add some structure so it's learnable
    y = pd.Series(
        ((X['feature_0'] > 0) & (X['feature_1'] < 0.5)).astype(int)
    )
    
    # Initialize the zoo
    print("Initializing Model Zoo...")
    zoo = ModelZoo()
    
    # Train all models
    print("\nTraining all models...")
    performance = zoo.train_all(X, y)
    
    print("\nTraining Performance:")
    for model, accuracy in performance.items():
        print(f"  {model}: {accuracy:.3f}")
        
    # Make ensemble predictions
    print("\nMaking ensemble predictions...")
    predictions = zoo.ensemble_predict(X.head(100))
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Positive predictions: {predictions.mean():.1%}")
    
    # Get zoo report
    print("\nZoo Report:")
    print(zoo.get_zoo_report())
    
    # Get wisdom from the Oracle
    print("\nOracle's Trading Rules:")
    oracle = zoo.inhabitants['oracle']
    for rule in oracle.share_wisdom(top_n=3):
        print(f"  If {rule['path']}")
        print(f"    Then predict {rule['prediction']} (confidence: {rule['confidence']:.2f})")
