from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
import pandas as pd
from joblib import load, dump
import json
from datetime import datetime, timedelta
import os

class ModelAdjuster:
    def __init__(self, models_dir='exportedModels', history_file='model_history.json'):
        self.models_dir = models_dir
        self.history_file = os.path.join(models_dir, history_file)
        self.performance_window = 50  # Number of matches to consider for recent performance
        self.adjustment_threshold = 0.1  # Minimum difference in performance to trigger adjustment
        self.load_history()
    
    def load_history(self):
        """Load or initialize model performance history"""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = {
                'predictions': [],
                'model_versions': [],
                'performance_metrics': {
                    'accuracy': [],
                    'f1_score': [],
                    'confidence_calibration': []
                }
            }
    
    def save_history(self):
        """Save model performance history"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f)
    
    def record_prediction(self, match_data, predictions, actual_result):
        """Record a prediction and its outcome"""
        prediction_record = {
            'date': match_data['Date'].strftime('%Y-%m-%d'),
            'match_id': f"{match_data['HomeTeam']}_{match_data['AwayTeam']}_{match_data['Date']}",
            'home_team': match_data['HomeTeam'],
            'away_team': match_data['AwayTeam'],
            'predictions': {
                'lr': predictions['lr_pred'],
                'nb': predictions['nb_pred'],
                'rf': predictions['rf_pred'],
                'ensemble': predictions['ensemble_pred']
            },
            'probabilities': {
                'lr': predictions['lr_proba'].tolist() if isinstance(predictions['lr_proba'], np.ndarray) else predictions['lr_proba'],
                'nb': predictions['nb_proba'].tolist() if isinstance(predictions['nb_proba'], np.ndarray) else predictions['nb_proba'],
                'rf': predictions['rf_proba'].tolist() if isinstance(predictions['rf_proba'], np.ndarray) else predictions['rf_proba'],
                'ensemble': predictions['ensemble_proba'].tolist() if isinstance(predictions['ensemble_proba'], np.ndarray) else predictions['ensemble_proba']
            },
            'confidence_scores': {
                'lr': predictions['lr_confidence'],
                'nb': predictions['nb_confidence'],
                'rf': predictions['rf_confidence'],
                'ensemble': predictions['ensemble_confidence']
            },
            'actual_result': actual_result
        }
        
        self.history['predictions'].append(prediction_record)
        self.save_history()
    
    def calculate_recent_performance(self):
        """Calculate performance metrics for recent predictions"""
        if len(self.history['predictions']) == 0:
            return None
        
        recent_predictions = self.history['predictions'][-self.performance_window:]
        
        metrics = {
            'accuracy': {},
            'f1_score': {},
            'confidence_calibration': {},
            'prediction_bias': {}
        }
        
        for model in ['lr', 'nb', 'rf', 'ensemble']:
            y_true = [p['actual_result'] for p in recent_predictions]
            y_pred = [p['predictions'][model] for p in recent_predictions]
            probas = np.array([p['probabilities'][model] for p in recent_predictions])
            confidences = [p['confidence_scores'][model] for p in recent_predictions]
            
            metrics['accuracy'][model] = accuracy_score(y_true, y_pred)
            metrics['f1_score'][model] = f1_score(y_true, y_pred, average='weighted')
            
            # Calculate confidence calibration (how well confidence scores match actual performance)
            correct_predictions = [1 if true == pred else 0 
                                for true, pred in zip(y_true, y_pred)]
            metrics['confidence_calibration'][model] = np.mean(
                [abs(conf - corr) for conf, corr in zip(confidences, correct_predictions)]
            )
            
            # Calculate prediction bias (systematic over/under-prediction)
            actual_outcomes = pd.get_dummies(y_true)
            metrics['prediction_bias'][model] = np.mean(probas - actual_outcomes.values, axis=0)
        
        return metrics
    
    def adjust_model_weights(self, current_weights):
        """Adjust ensemble model weights based on recent performance"""
        metrics = self.calculate_recent_performance()
        if not metrics:
            return current_weights
        
        # Calculate new weights based on recent performance
        performance_scores = {
            model: (
                0.4 * metrics['accuracy'][model] +
                0.3 * metrics['f1_score'][model] +
                0.3 * (1 - metrics['confidence_calibration'][model])
            )
            for model in ['lr', 'nb', 'rf']
        }
        
        # Normalize weights
        total_score = sum(performance_scores.values())
        new_weights = {
            model: score / total_score
            for model, score in performance_scores.items()
        }
        
        # Only adjust if the change is significant
        if any(abs(new_weights[model] - current_weights[model]) > self.adjustment_threshold
               for model in new_weights):
            return new_weights
        
        return current_weights
    
    def adjust_confidence_thresholds(self, current_scores=None):
        """Adjust confidence thresholds based on historical performance"""
        metrics = self.calculate_recent_performance()
        if not metrics:
            return None
        
        thresholds = {}
        for model in ['lr', 'nb', 'rf', 'ensemble']:
            # Calculate optimal threshold based on confidence calibration
            recent_predictions = self.history['predictions'][-self.performance_window:]
            confidences = [p['confidence_scores'][model] for p in recent_predictions]
            correct_predictions = [1 if p['predictions'][model] == p['actual_result'] else 0 
                                for p in recent_predictions]
            
            # Find the confidence threshold that best separates correct from incorrect predictions
            sorted_pairs = sorted(zip(confidences, correct_predictions))
            best_threshold = 0.5
            best_separation = 0
            
            for threshold in np.arange(0.3, 0.9, 0.05):
                high_conf_correct = sum(1 for c, p in sorted_pairs if c >= threshold and p == 1)
                high_conf_total = sum(1 for c, _ in sorted_pairs if c >= threshold)
                low_conf_incorrect = sum(1 for c, p in sorted_pairs if c < threshold and p == 0)
                low_conf_total = sum(1 for c, _ in sorted_pairs if c < threshold)
                
                if high_conf_total > 0 and low_conf_total > 0:
                    separation = (high_conf_correct / high_conf_total +
                                low_conf_incorrect / low_conf_total) / 2
                    if separation > best_separation:
                        best_separation = separation
                        best_threshold = threshold
            
            thresholds[model] = best_threshold
        
        # If current scores are provided, adjust them based on thresholds
        if current_scores:
            adjusted_scores = {}
            for model in ['lr', 'nb', 'rf', 'ensemble']:
                if model in current_scores and model in thresholds:
                    # Scale the confidence score based on the threshold
                    adjusted_scores[model] = current_scores[model] * (thresholds[model] / 0.5)
            return adjusted_scores
            
        return thresholds
    
    def get_model_insights(self):
        """Generate insights about model performance and adjustments"""
        metrics = self.calculate_recent_performance()
        if not metrics:
            return "Insufficient data for insights"
        
        insights = []
        
        # Analyze overall performance trends
        for model in ['lr', 'nb', 'rf', 'ensemble']:
            accuracy = metrics['accuracy'][model]
            bias = metrics['prediction_bias'][model]
            calibration = metrics['confidence_calibration'][model]
            
            insights.append(f"{model.upper()} Model Performance:")
            insights.append(f"- Accuracy: {accuracy:.3f}")
            insights.append(f"- Confidence Calibration Error: {calibration:.3f}")
            insights.append("- Prediction Bias:")
            insights.append(f"  Home: {bias[0]:.3f}, Draw: {bias[1]:.3f}, Away: {bias[2]:.3f}")
            insights.append("")
        
        # Add specific recommendations
        if any(metrics['confidence_calibration'].values()):
            worst_calibrated = max(metrics['confidence_calibration'].items(),
                                 key=lambda x: x[1])[0]
            insights.append(f"Recommendation: Adjust confidence calculation for {worst_calibrated} model")
        
        return "\n".join(insights) 