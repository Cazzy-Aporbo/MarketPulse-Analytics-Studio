"""
Model Ensemble Orchestrator for MarketPulse
Cazandra Aporbo, MS
May 2025

This module manages multiple ML models working together. Single models are like
solo musicians - good but limited. Ensembles are the full orchestra. Each model
sees the problem differently, and that diversity creates robustness.

Started with grand plans for 10+ models. Reality check: 3-4 well-tuned models
beat 10 mediocre ones every time. Quality over quantity.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
import pickle
from pathlib import Path
from datetime import datetime
import json

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """
    Container for model performance metrics.
    Everything you need to know about how a model is doing.
    """
    accuracy: float
    precision: float
    recall: float
    f1: float
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    training_time: float = 0
    prediction_time: float = 0
    
    def to_dict(self) -> Dict:
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time
        }
    
    def summary(self) -> str:
        return (
            f"Accuracy: {self.accuracy:.3f} | "
            f"Precision: {self.precision:.3f} | "
            f"Recall: {self.recall:.3f} | "
            f"F1: {self.f1:.3f}"
        )


@dataclass
class EnsembleConfig:
    """
    Configuration for the ensemble.
    Tweak these knobs to change behavior.
    """
    # Model weights for voting
    model_weights: Dict[str, float] = field(default_factory=lambda: {
        'logistic': 0.25,
        'random_forest': 0.35,
        'gradient_boost': 0.40
    })
    
    # Training parameters
    test_size: float = 0.2
    n_splits: int = 5  # For time series cross-validation
    random_state: int = 42
    
    # Performance thresholds
    min_accuracy: float = 0.55  # Better than random
    retrain_threshold: float = 0.10  # Retrain if performance drops by 10%
    
    # Model-specific configs
    logistic_config: Dict = field(default_factory=lambda: {
        'max_iter': 1000,
        'C': 1.0,
        'solver': 'lbfgs'
    })
    
    rf_config: Dict = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2
    })


class TimeSeriesValidator:
    """
    Proper validation for time series data.
    Never use random splits on time series - that's cheating (look-ahead bias).
    Walk-forward validation is the way.
    """
    
    def __init__(self, n_splits: int = 5, test_size: float = 0.2):
        self.n_splits = n_splits
        self.test_size = test_size
        
    def split(self, X: pd.DataFrame, y: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create time series splits for validation.
        Each split uses past data to predict future data.
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        splits = []
        
        for train_idx, test_idx in tscv.split(X):
            # Ensure minimum training size
            if len(train_idx) < 50:  # Need at least 50 samples to train
                continue
                
            splits.append((train_idx, test_idx))
            
        return splits
    
    def validate_model(self, model: BaseEstimator, 
                       X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        """
        Validate a model using time series cross-validation.
        Returns averaged metrics across all splits.
        """
        metrics = []
        
        for train_idx, test_idx in self.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Scale features (important!)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train and predict
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            metrics.append({
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            })
        
        # Average metrics across all splits
        avg_metrics = pd.DataFrame(metrics).mean()
        
        return ModelMetrics(
            accuracy=avg_metrics['accuracy'],
            precision=avg_metrics['precision'],
            recall=avg_metrics['recall'],
            f1=avg_metrics['f1'],
            predictions=np.array([]),  # Empty for validation
            probabilities=None
        )


class BaseModel:
    """
    Base class for all models in the ensemble.
    Provides common functionality and interface.
    """
    
    def __init__(self, name: str, config: Dict = None):
        self.name = name
        self.config = config or {}
        self.model = None
        self.scaler = StandardScaler()
        self.metrics = None
        self.is_trained = False
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        """
        Train the model and return metrics.
        """
        start_time = datetime.now()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Calculate training metrics
        y_pred = self.model.predict(X_scaled)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        self.metrics = ModelMetrics(
            accuracy=accuracy_score(y, y_pred),
            precision=precision_score(y, y_pred, average='weighted', zero_division=0),
            recall=recall_score(y, y_pred, average='weighted', zero_division=0),
            f1=f1_score(y, y_pred, average='weighted', zero_division=0),
            predictions=y_pred,
            training_time=training_time
        )
        
        logger.info(f"{self.name} trained: {self.metrics.summary()}")
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        """
        if not self.is_trained:
            raise ValueError(f"{self.name} is not trained yet")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        """
        if not self.is_trained:
            raise ValueError(f"{self.name} is not trained yet")
            
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        else:
            # Fallback for models without probability
            predictions = self.model.predict(X_scaled)
            # Convert to probability-like format
            proba = np.zeros((len(predictions), 2))
            proba[np.arange(len(predictions)), predictions.astype(int)] = 1
            return proba


class LogisticModel(BaseModel):
    """
    Logistic Regression: The reliable baseline.
    Simple, fast, interpretable. Often surprisingly effective.
    """
    
    def __init__(self, config: Dict = None):
        super().__init__("Logistic Regression", config)
        self.model = LogisticRegression(
            max_iter=self.config.get('max_iter', 1000),
            C=self.config.get('C', 1.0),
            solver=self.config.get('solver', 'lbfgs'),
            random_state=42
        )


class RandomForestModel(BaseModel):
    """
    Random Forest: The Swiss Army knife of ML.
    Handles non-linearity, feature interactions, doesn't overfit easily.
    """
    
    def __init__(self, config: Dict = None):
        super().__init__("Random Forest", config)
        self.model = RandomForestClassifier(
            n_estimators=self.config.get('n_estimators', 100),
            max_depth=self.config.get('max_depth', 10),
            min_samples_split=self.config.get('min_samples_split', 5),
            min_samples_leaf=self.config.get('min_samples_leaf', 2),
            random_state=42,
            n_jobs=-1  # Use all cores
        )
    
    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """
        Get feature importance from the random forest.
        This is gold for understanding what drives predictions.
        """
        if not self.is_trained:
            return {}
            
        importance = dict(zip(feature_names, self.model.feature_importances_))
        # Sort by importance
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


class ModelEnsemble:
    """
    The main ensemble orchestrator.
    Manages multiple models, combines predictions, tracks performance.
    """
    
    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        self.models = {}
        self.performance_history = []
        self.validator = TimeSeriesValidator(
            n_splits=self.config.n_splits,
            test_size=self.config.test_size
        )
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """
        Initialize all models in the ensemble.
        """
        self.models['logistic'] = LogisticModel(self.config.logistic_config)
        self.models['random_forest'] = RandomForestModel(self.config.rf_config)
        
        logger.info(f"Initialized {len(self.models)} models")
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              validate: bool = True) -> Dict[str, ModelMetrics]:
        """
        Train all models in the ensemble.
        """
        logger.info("Training ensemble...")
        metrics = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            if validate:
                # Use time series validation
                model_metrics = self.validator.validate_model(
                    model.model, X, y
                )
                # Now train on full data
                model.train(X, y)
                model.metrics = model_metrics
            else:
                # Just train without validation
                model_metrics = model.train(X, y)
            
            metrics[name] = model_metrics
            
            # Check if model meets minimum performance
            if model_metrics.accuracy < self.config.min_accuracy:
                logger.warning(
                    f"{name} accuracy {model_metrics.accuracy:.3f} "
                    f"below minimum {self.config.min_accuracy}"
                )
        
        # Store performance for monitoring
        self.performance_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })
        
        return metrics
    
    def predict(self, X: pd.DataFrame, method: str = 'weighted') -> np.ndarray:
        """
        Make ensemble predictions.
        Methods: 'weighted', 'voting', 'stacking'
        """
        if not all(model.is_trained for model in self.models.values()):
            raise ValueError("Not all models are trained")
        
        if method == 'weighted':
            return self._weighted_predict(X)
        elif method == 'voting':
            return self._voting_predict(X)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _weighted_predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Weighted average of model predictions.
        Weights based on individual model performance.
        """
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            pred_proba = model.predict_proba(X)[:, 1]  # Probability of positive class
            predictions.append(pred_proba)
            
            # Use configured weights or performance-based weights
            weight = self.config.model_weights.get(name, 1.0)
            if model.metrics:
                # Adjust weight based on performance
                weight *= model.metrics.accuracy
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Weighted average
        weighted_proba = np.average(predictions, axis=0, weights=weights)
        
        # Convert to binary predictions
        return (weighted_proba > 0.5).astype(int)
    
    def _voting_predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Simple majority voting.
        Each model gets one vote.
        """
        predictions = []
        
        for model in self.models.values():
            pred = model.predict(X)
            predictions.append(pred)
        
        # Majority vote
        predictions = np.array(predictions)
        return np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), 
            axis=0, 
            arr=predictions
        )
    
    def get_model_weights(self) -> Dict[str, float]:
        """
        Get current model weights based on performance.
        """
        weights = {}
        
        for name, model in self.models.items():
            base_weight = self.config.model_weights.get(name, 1.0)
            
            if model.metrics:
                # Adjust based on accuracy
                performance_weight = model.metrics.accuracy
                weights[name] = base_weight * performance_weight
            else:
                weights[name] = base_weight
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
            
        return weights
    
    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """
        Get aggregated feature importance from all models.
        Only Random Forest provides this, but we could add more.
        """
        if 'random_forest' in self.models:
            return self.models['random_forest'].get_feature_importance(feature_names)
        return {}
    
    def save(self, path: Path):
        """
        Save the ensemble to disk.
        """
        save_dict = {
            'config': self.config,
            'models': self.models,
            'performance_history': self.performance_history
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
            
        logger.info(f"Ensemble saved to {path}")
    
    def load(self, path: Path):
        """
        Load the ensemble from disk.
        """
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
            
        self.config = save_dict['config']
        self.models = save_dict['models']
        self.performance_history = save_dict['performance_history']
        
        logger.info(f"Ensemble loaded from {path}")
    
    def get_performance_summary(self) -> pd.DataFrame:
        """
        Get a summary of model performance over time.
        """
        if not self.performance_history:
            return pd.DataFrame()
        
        summary_data = []
        
        for record in self.performance_history:
            timestamp = record['timestamp']
            for model_name, metrics in record['metrics'].items():
                summary_data.append({
                    'timestamp': timestamp,
                    'model': model_name,
                    'accuracy': metrics.accuracy,
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                    'f1': metrics.f1
                })
        
        return pd.DataFrame(summary_data)


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Create target with some signal
    y = pd.Series(
        (X['feature_0'] + 0.5 * X['feature_1'] - 0.3 * X['feature_2'] + 
         np.random.randn(n_samples) * 0.5 > 0).astype(int)
    )
    
    # Train ensemble
    ensemble = ModelEnsemble()
    metrics = ensemble.train(X, y, validate=True)
    
    print("\nModel Performance:")
    print("-" * 50)
    for model_name, model_metrics in metrics.items():
        print(f"{model_name}: {model_metrics.summary()}")
    
    # Make predictions
    predictions = ensemble.predict(X.iloc[-100:], method='weighted')
    
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Prediction distribution: {pd.Series(predictions).value_counts()}")
    
    # Get model weights
    weights = ensemble.get_model_weights()
    print("\nModel Weights:")
    for name, weight in weights.items():
        print(f"  {name}: {weight:.3f}")
    
    # Get feature importance
    importance = ensemble.get_feature_importance(X.columns.tolist())
    print("\nTop 5 Important Features:")
    for feature, score in list(importance.items())[:5]:
        print(f"  {feature}: {score:.4f}")
