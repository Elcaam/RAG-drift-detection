"""
Evaluation metrics for RAG pipeline performance.

This module provides comprehensive evaluation metrics for assessing the performance
of Retrieval-Augmented Generation systems, including retrieval and generation metrics.
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from bert_score import score as bert_score
from sklearn.metrics import f1_score, precision_score, recall_score
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class RAGMetrics:
    """Compute various metrics for RAG pipeline evaluation."""
    
    def __init__(self):
        """Initialize the metrics calculator with stop words."""
        self.stop_words = set(stopwords.words('english'))
    
    def normalize_answer(self, text: str) -> str:
        """
        Normalize answer text for comparison.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text (lowercase, no punctuation, trimmed)
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def exact_match(self, prediction: str, ground_truth: str) -> bool:
        """
        Compute exact match between prediction and ground truth.
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            True if exact match, False otherwise
        """
        pred_norm = self.normalize_answer(prediction)
        gt_norm = self.normalize_answer(ground_truth)
        
        return pred_norm == gt_norm
    
    def f1_score(self, prediction: str, ground_truth: str) -> float:
        """
        Compute F1 score between prediction and ground truth.
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            F1 score (harmonic mean of precision and recall)
        """
        pred_tokens = set(word_tokenize(self.normalize_answer(prediction)))
        gt_tokens = set(word_tokenize(self.normalize_answer(ground_truth)))
        
        # Remove stop words for better comparison
        pred_tokens = pred_tokens - self.stop_words
        gt_tokens = gt_tokens - self.stop_words
        
        if not gt_tokens:
            return 1.0 if not pred_tokens else 0.0
        
        common = pred_tokens & gt_tokens
        precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
        recall = len(common) / len(gt_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def bert_score_similarity(self, predictions: List[str], references: List[str]) -> Tuple[List[float], float]:
        """
        Compute BERTScore similarity between predictions and references.
        
        Args:
            predictions: List of predicted answers
            references: List of reference answers
            
        Returns:
            Tuple of (individual scores, average score)
        """
        try:
            P, R, F1 = bert_score(predictions, references, lang='en', verbose=True)
            return F1.tolist(), F1.mean().item()
        except Exception as e:
            print(f"BERTScore computation failed: {e}")
            # Fallback to simple similarity
            return self.simple_similarity(predictions, references)
    
    def simple_similarity(self, predictions: List[str], references: List[str]) -> Tuple[List[float], float]:
        """
        Compute simple text similarity as fallback when BERTScore fails.
        
        Args:
            predictions: List of predicted answers
            references: List of reference answers
            
        Returns:
            Tuple of (individual scores, average score)
        """
        similarities = []
        
        for pred, ref in zip(predictions, references):
            pred_norm = self.normalize_answer(pred)
            ref_norm = self.normalize_answer(ref)
            
            pred_tokens = set(word_tokenize(pred_norm))
            ref_tokens = set(word_tokenize(ref_norm))
            
            if not ref_tokens:
                similarity = 1.0 if not pred_tokens else 0.0
            else:
                intersection = len(pred_tokens & ref_tokens)
                union = len(pred_tokens | ref_tokens)
                similarity = intersection / union if union > 0 else 0.0
            
            similarities.append(similarity)
        
        return similarities, np.mean(similarities)
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def recall_at_k(self, retrieved_answers: List[str], ground_truth: str, k: int = 5) -> bool:
        """
        Check if ground truth is in top-k retrieved answers.
        
        Args:
            retrieved_answers: List of retrieved answers
            ground_truth: Ground truth answer
            k: Number of top answers to consider
            
        Returns:
            True if ground truth is in top-k, False otherwise
        """
        top_k_answers = retrieved_answers[:k]
        gt_norm = self.normalize_answer(ground_truth)
        
        for answer in top_k_answers:
            if self.normalize_answer(answer) == gt_norm:
                return True
        
        return False
    
    def precision_at_k(self, retrieved_answers: List[str], ground_truth: str, k: int = 5) -> float:
        """
        Compute precision at k.
        
        Args:
            retrieved_answers: List of retrieved answers
            ground_truth: Ground truth answer
            k: Number of top answers to consider
            
        Returns:
            Precision at k
        """
        top_k_answers = retrieved_answers[:k]
        gt_norm = self.normalize_answer(ground_truth)
        
        correct = 0
        for answer in top_k_answers:
            if self.normalize_answer(answer) == gt_norm:
                correct += 1
        
        return correct / k if k > 0 else 0.0
    
    def compute_retrieval_metrics(self, 
                                similarities: List[float],
                                recall_at_k_results: List[bool],
                                precision_at_k_results: List[float]) -> Dict:
        """
        Compute comprehensive retrieval metrics.
        
        Args:
            similarities: List of cosine similarities
            recall_at_k_results: List of recall@k results
            precision_at_k_results: List of precision@k results
            
        Returns:
            Dictionary with retrieval metrics
        """
        metrics = {
            'avg_cosine_similarity': np.mean(similarities),
            'std_cosine_similarity': np.std(similarities),
            'min_cosine_similarity': np.min(similarities),
            'max_cosine_similarity': np.max(similarities),
            'recall_at_k': np.mean(recall_at_k_results),
            'precision_at_k': np.mean(precision_at_k_results),
            'num_queries': len(similarities)
        }
        
        return metrics
    
    def compute_generation_metrics(self, 
                                 predictions: List[str],
                                 references: List[str]) -> Dict:
        """
        Compute comprehensive generation metrics.
        
        Args:
            predictions: List of predicted answers
            references: List of reference answers
            
        Returns:
            Dictionary with generation metrics
        """
        # Compute exact match
        em_scores = [self.exact_match(pred, ref) for pred, ref in zip(predictions, references)]
        em_accuracy = np.mean(em_scores)
        
        # Compute F1 scores
        f1_scores = [self.f1_score(pred, ref) for pred, ref in zip(predictions, references)]
        avg_f1 = np.mean(f1_scores)
        
        # Compute BERTScore
        bert_scores, avg_bert_score = self.bert_score_similarity(predictions, references)
        
        metrics = {
            'exact_match': em_accuracy,
            'avg_f1': avg_f1,
            'avg_bert_score': avg_bert_score,
            'f1_scores': f1_scores,
            'bert_scores': bert_scores,
            'num_samples': len(predictions)
        }
        
        return metrics
    
    def compute_comprehensive_metrics(self, 
                                    results: List[Dict]) -> Dict:
        """
        Compute comprehensive metrics for RAG pipeline results.
        
        Args:
            results: List of result dictionaries from RAG pipeline
            
        Returns:
            Dictionary with all metrics (retrieval, generation, overall)
        """
        similarities = []
        recall_at_k_results = []
        precision_at_k_results = []
        predictions = []
        references = []
        
        for result in results:
            # Extract retrieval metrics
            similarities.append(result['best_passage_score'])
            
            retrieved_answers = [p['metadata']['answer'] for p in result['passages']]
            ground_truth = result['ground_truth']
            
            recall_at_k_results.append(self.recall_at_k(retrieved_answers, ground_truth))
            precision_at_k_results.append(self.precision_at_k(retrieved_answers, ground_truth))
            
            # Extract generation metrics
            predictions.append(result['answer'])
            references.append(ground_truth)
        
        # Compute metrics
        retrieval_metrics = self.compute_retrieval_metrics(
            similarities, recall_at_k_results, precision_at_k_results
        )
        
        generation_metrics = self.compute_generation_metrics(predictions, references)
        
        # Combine all metrics
        all_metrics = {
            'retrieval': retrieval_metrics,
            'generation': generation_metrics,
            'overall': {
                'num_queries': len(results),
                'avg_retrieval_score': retrieval_metrics['avg_cosine_similarity'],
                'avg_generation_score': generation_metrics['avg_f1']
            }
        }
        
        return all_metrics


def create_metrics_summary(metrics_history: List[Dict]) -> pd.DataFrame:
    """
    Create a summary DataFrame from metrics history.
    
    Args:
        metrics_history: List of metrics dictionaries over time
        
    Returns:
        DataFrame with metrics summary
    """
    summary_data = []
    
    for i, metrics in enumerate(metrics_history):
        row = {
            'week': i + 1,
            'avg_cosine_similarity': metrics['retrieval']['avg_cosine_similarity'],
            'recall_at_k': metrics['retrieval']['recall_at_k'],
            'precision_at_k': metrics['retrieval']['precision_at_k'],
            'exact_match': metrics['generation']['exact_match'],
            'avg_f1': metrics['generation']['avg_f1'],
            'avg_bert_score': metrics['generation']['avg_bert_score']
        }
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)


if __name__ == "__main__":
    # Example usage
    metrics = RAGMetrics()
    
    # Test metrics
    prediction = "The quick brown fox"
    ground_truth = "The quick brown fox jumps"
    
    print(f"Exact Match: {metrics.exact_match(prediction, ground_truth)}")
    print(f"F1 Score: {metrics.f1_score(prediction, ground_truth)}")
    
    # Test BERTScore
    predictions = ["The quick brown fox", "A fast red dog"]
    references = ["The quick brown fox jumps", "A fast red dog runs"]
    
    bert_scores, avg_bert = metrics.bert_score_similarity(predictions, references)
    print(f"BERTScore: {avg_bert}")
