"""
Drift detection algorithms for RAG pipeline monitoring.

This module provides comprehensive drift detection capabilities for monitoring
performance changes in Retrieval-Augmented Generation systems over time.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class DriftDetector:
    """Detect drift in RAG pipeline performance over time."""
    
    def __init__(self, 
                 baseline_metrics: Dict,
                 threshold_method: str = "z_score",
                 z_threshold: float = 2.0,
                 window_size: int = 5):
        """
        Initialize the drift detector.
        
        Args:
            baseline_metrics: Baseline performance metrics
            threshold_method: Method for drift detection ("z_score", "percentile", "absolute")
            z_threshold: Z-score threshold for drift detection
            window_size: Size of sliding window for trend analysis
        """
        self.baseline_metrics = baseline_metrics
        self.threshold_method = threshold_method
        self.z_threshold = z_threshold
        self.window_size = window_size
        self.metrics_history = []
        self.drift_alerts = []
        
    def add_metrics(self, metrics: Dict) -> None:
        """
        Add new metrics to the history.
        
        Args:
            metrics: New metrics dictionary
        """
        self.metrics_history.append(metrics)
    
    def compute_z_score(self, current_value: float, baseline_value: float, baseline_std: float) -> float:
        """
        Compute z-score for drift detection.
        
        Args:
            current_value: Current metric value
            baseline_value: Baseline metric value
            baseline_std: Baseline standard deviation
            
        Returns:
            Z-score
        """
        if baseline_std == 0:
            return 0.0
        
        return abs(current_value - baseline_value) / baseline_std
    
    def detect_threshold_drift(self, current_metrics: Dict) -> Dict:
        """
        Detect drift using threshold-based methods.
        
        Args:
            current_metrics: Current performance metrics
            
        Returns:
            Dictionary with drift detection results
        """
        drift_results = {
            'retrieval_drift': False,
            'generation_drift': False,
            'overall_drift': False,
            'drift_scores': {},
            'drift_reasons': []
        }
        
        # Check retrieval metrics
        baseline_retrieval = self.baseline_metrics['retrieval']
        current_retrieval = current_metrics['retrieval']
        
        # Cosine similarity drift
        cos_sim_z = self.compute_z_score(
            current_retrieval['avg_cosine_similarity'],
            baseline_retrieval['avg_cosine_similarity'],
            baseline_retrieval.get('std_cosine_similarity', 0.1)
        )
        
        # Recall@K drift
        recall_z = self.compute_z_score(
            current_retrieval['recall_at_k'],
            baseline_retrieval['recall_at_k'],
            0.1  # Assume 10% standard deviation
        )
        
        # Check generation metrics
        baseline_generation = self.baseline_metrics['generation']
        current_generation = current_metrics['generation']
        
        # F1 score drift
        f1_z = self.compute_z_score(
            current_generation['avg_f1'],
            baseline_generation['avg_f1'],
            0.1  # Assume 10% standard deviation
        )
        
        # Exact match drift
        em_z = self.compute_z_score(
            current_generation['exact_match'],
            baseline_generation['exact_match'],
            0.1  # Assume 10% standard deviation
        )
        
        # Store drift scores
        drift_results['drift_scores'] = {
            'cosine_similarity_z': cos_sim_z,
            'recall_at_k_z': recall_z,
            'f1_z': f1_z,
            'exact_match_z': em_z
        }
        
        # Determine drift flags
        retrieval_drift = cos_sim_z > self.z_threshold or recall_z > self.z_threshold
        generation_drift = f1_z > self.z_threshold or em_z > self.z_threshold
        
        drift_results['retrieval_drift'] = retrieval_drift
        drift_results['generation_drift'] = generation_drift
        drift_results['overall_drift'] = retrieval_drift or generation_drift
        
        # Add drift reasons
        if cos_sim_z > self.z_threshold:
            drift_results['drift_reasons'].append(f"Cosine similarity dropped (z={cos_sim_z:.2f})")
        if recall_z > self.z_threshold:
            drift_results['drift_reasons'].append(f"Recall@K dropped (z={recall_z:.2f})")
        if f1_z > self.z_threshold:
            drift_results['drift_reasons'].append(f"F1 score dropped (z={f1_z:.2f})")
        if em_z > self.z_threshold:
            drift_results['drift_reasons'].append(f"Exact match dropped (z={em_z:.2f})")
        
        return drift_results
    
    def detect_trend_drift(self) -> Dict:
        """
        Detect drift using trend analysis.
        
        Returns:
            Dictionary with trend analysis results
        """
        if len(self.metrics_history) < self.window_size:
            return {'trend_drift': False, 'trend_direction': 'insufficient_data'}
        
        # Extract recent metrics for trend analysis
        recent_metrics = self.metrics_history[-self.window_size:]
        
        # Analyze trends for key metrics
        trends = {}
        
        # Cosine similarity trend
        cos_sim_values = [m['retrieval']['avg_cosine_similarity'] for m in recent_metrics]
        cos_sim_trend = self.compute_trend(cos_sim_values)
        trends['cosine_similarity'] = cos_sim_trend
        
        # F1 score trend
        f1_values = [m['generation']['avg_f1'] for m in recent_metrics]
        f1_trend = self.compute_trend(f1_values)
        trends['f1_score'] = f1_trend
        
        # Recall@K trend
        recall_values = [m['retrieval']['recall_at_k'] for m in recent_metrics]
        recall_trend = self.compute_trend(recall_values)
        trends['recall_at_k'] = recall_trend
        
        # Determine overall trend
        negative_trends = sum(1 for trend in trends.values() if trend < -0.1)
        positive_trends = sum(1 for trend in trends.values() if trend > 0.1)
        
        if negative_trends >= 2:
            trend_direction = 'declining'
            trend_drift = True
        elif positive_trends >= 2:
            trend_direction = 'improving'
            trend_drift = False
        else:
            trend_direction = 'stable'
            trend_drift = False
        
        return {
            'trend_drift': trend_drift,
            'trend_direction': trend_direction,
            'trends': trends
        }
    
    def compute_trend(self, values: List[float]) -> float:
        """
        Compute trend slope using linear regression.
        
        Args:
            values: List of metric values over time
            
        Returns:
            Trend slope
        """
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope, _, _, _, _ = stats.linregress(x, values)
        
        return slope
    
    def detect_statistical_drift(self, current_metrics: Dict) -> Dict:
        """
        Detect drift using statistical tests.
        
        Args:
            current_metrics: Current performance metrics
            
        Returns:
            Dictionary with statistical drift results
        """
        if len(self.metrics_history) < 10:
            return {'statistical_drift': False, 'p_values': {}}
        
        # Extract historical values for statistical comparison
        historical_cos_sim = [m['retrieval']['avg_cosine_similarity'] for m in self.metrics_history[:-1]]
        historical_f1 = [m['generation']['avg_f1'] for m in self.metrics_history[:-1]]
        
        current_cos_sim = current_metrics['retrieval']['avg_cosine_similarity']
        current_f1 = current_metrics['generation']['avg_f1']
        
        # Perform t-tests
        try:
            cos_sim_stat, cos_sim_p = stats.ttest_1samp(historical_cos_sim, current_cos_sim)
            f1_stat, f1_p = stats.ttest_1samp(historical_f1, current_f1)
        except:
            cos_sim_p = 1.0
            f1_p = 1.0
        
        # Determine statistical drift
        alpha = 0.05
        statistical_drift = cos_sim_p < alpha or f1_p < alpha
        
        return {
            'statistical_drift': statistical_drift,
            'p_values': {
                'cosine_similarity_p': cos_sim_p,
                'f1_p': f1_p
            }
        }
    
    def detect_drift(self, current_metrics: Dict) -> Dict:
        """
        Comprehensive drift detection using multiple methods.
        
        Args:
            current_metrics: Current performance metrics
            
        Returns:
            Dictionary with comprehensive drift analysis
        """
        # Add to history
        self.add_metrics(current_metrics)
        
        # Perform different drift detection methods
        threshold_results = self.detect_threshold_drift(current_metrics)
        trend_results = self.detect_trend_drift()
        statistical_results = self.detect_statistical_drift(current_metrics)
        
        # Combine results
        drift_detected = (
            threshold_results['overall_drift'] or 
            trend_results['trend_drift'] or 
            statistical_results['statistical_drift']
        )
        
        # Create alert if drift detected
        if drift_detected:
            alert = {
                'timestamp': len(self.metrics_history),
                'drift_type': [],
                'severity': 'medium',
                'details': {}
            }
            
            if threshold_results['overall_drift']:
                alert['drift_type'].append('threshold')
                alert['details']['threshold'] = threshold_results
                
            if trend_results['trend_drift']:
                alert['drift_type'].append('trend')
                alert['details']['trend'] = trend_results
                
            if statistical_results['statistical_drift']:
                alert['drift_type'].append('statistical')
                alert['details']['statistical'] = statistical_results
            
            # Determine severity based on z-scores
            max_z_score = max(threshold_results['drift_scores'].values())
            if max_z_score > 3.0:
                alert['severity'] = 'high'
            elif max_z_score > 2.0:
                alert['severity'] = 'medium'
            else:
                alert['severity'] = 'low'
            
            self.drift_alerts.append(alert)
        
        return {
            'drift_detected': drift_detected,
            'threshold_results': threshold_results,
            'trend_results': trend_results,
            'statistical_results': statistical_results,
            'current_metrics': current_metrics,
            'baseline_metrics': self.baseline_metrics
        }
    
    def get_drift_summary(self) -> pd.DataFrame:
        """
        Get summary of drift detection results.
        
        Returns:
            DataFrame with drift summary
        """
        if not self.metrics_history:
            return pd.DataFrame()
        
        summary_data = []
        
        for i, metrics in enumerate(self.metrics_history):
            row = {
                'week': i + 1,
                'avg_cosine_similarity': metrics['retrieval']['avg_cosine_similarity'],
                'recall_at_k': metrics['retrieval']['recall_at_k'],
                'exact_match': metrics['generation']['exact_match'],
                'avg_f1': metrics['generation']['avg_f1'],
                'avg_bert_score': metrics['generation']['avg_bert_score']
            }
            
            # Add drift flags if available
            if i > 0:  # Skip baseline
                # Compute drift scores
                baseline_retrieval = self.baseline_metrics['retrieval']
                baseline_generation = self.baseline_metrics['generation']
                
                cos_sim_z = self.compute_z_score(
                    row['avg_cosine_similarity'],
                    baseline_retrieval['avg_cosine_similarity'],
                    baseline_retrieval.get('std_cosine_similarity', 0.1)
                )
                
                f1_z = self.compute_z_score(
                    row['avg_f1'],
                    baseline_generation['avg_f1'],
                    0.1
                )
                
                # Determine drift flag
                if cos_sim_z > self.z_threshold or f1_z > self.z_threshold:
                    drift_flag = 'High'
                elif cos_sim_z > self.z_threshold * 0.5 or f1_z > self.z_threshold * 0.5:
                    drift_flag = 'Slight'
                else:
                    drift_flag = 'No'
                
                row['drift_flag'] = drift_flag
                row['cosine_sim_z'] = cos_sim_z
                row['f1_z'] = f1_z
            else:
                row['drift_flag'] = 'Baseline'
                row['cosine_sim_z'] = 0.0
                row['f1_z'] = 0.0
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def get_alerts(self) -> List[Dict]:
        """Get list of drift alerts."""
        return self.drift_alerts


def create_drift_detector_from_baseline(baseline_metrics: Dict, **kwargs) -> DriftDetector:
    """
    Create a drift detector with baseline metrics.
    
    Args:
        baseline_metrics: Baseline performance metrics
        **kwargs: Additional arguments for DriftDetector
        
    Returns:
        Initialized DriftDetector
    """
    return DriftDetector(baseline_metrics, **kwargs)


if __name__ == "__main__":
    # Example usage
    baseline_metrics = {
        'retrieval': {
            'avg_cosine_similarity': 0.85,
            'std_cosine_similarity': 0.1,
            'recall_at_k': 0.78,
            'precision_at_k': 0.72
        },
        'generation': {
            'exact_match': 0.65,
            'avg_f1': 0.72,
            'avg_bert_score': 0.81
        }
    }
    
    detector = DriftDetector(baseline_metrics)
    
    # Simulate some metrics
    current_metrics = {
        'retrieval': {
            'avg_cosine_similarity': 0.75,  # Dropped
            'std_cosine_similarity': 0.12,
            'recall_at_k': 0.70,  # Dropped
            'precision_at_k': 0.65
        },
        'generation': {
            'exact_match': 0.55,  # Dropped
            'avg_f1': 0.65,  # Dropped
            'avg_bert_score': 0.75
        }
    }
    
    drift_results = detector.detect_drift(current_metrics)
    print("Drift detected:", drift_results['drift_detected'])
    print("Drift reasons:", drift_results['threshold_results']['drift_reasons'])
