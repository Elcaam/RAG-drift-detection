#!/usr/bin/env python3
"""
Demonstration of drift detection with more dramatic changes to trigger drift.

This script demonstrates how the drift detection system works by creating
synthetic scenarios that cause significant performance degradation in a RAG system.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import pandas as pd
import numpy as np
from src.rag_pipeline import RAGPipeline
from src.metrics import RAGMetrics
from src.drift_detector import DriftDetector
from src.visualization import DriftVisualizer


def create_drift_demo_data():
    """Create synthetic data for drift detection demonstration."""
    print("Creating drift demonstration data...")
    
    # Create baseline data with consistent quality
    baseline_data = []
    for i in range(100):
        baseline_data.append({
            'id': f'baseline_{i}',
            'title': f'Document_{i//10}',
            'context': f'This is a comprehensive context about topic {i}. It contains detailed information that can be used to answer questions about this subject. The topic {i} is well-documented and thoroughly explained.',
            'question': f'What is topic {i}?',
            'answer_text': f'Topic {i} is a comprehensive subject with detailed information.',
            'answer_start': 0,
            'is_impossible': False
        })
    
    return pd.DataFrame(baseline_data)


def create_drift_scenarios(baseline_data):
    """Create scenarios that will cause drift by degrading data quality."""
    scenarios = []
    
    # Scenario 1: Major content degradation
    degraded_data = baseline_data.copy()
    degraded_data['context'] = degraded_data['context'].apply(
        lambda x: x.replace('comprehensive', 'basic').replace('detailed', 'minimal')
    )
    degraded_data['answer_text'] = degraded_data['answer_text'].apply(
        lambda x: x.replace('comprehensive', 'basic').replace('detailed', 'minimal')
    )
    scenarios.append(("Content Degradation", degraded_data))
    
    # Scenario 2: Add irrelevant content
    irrelevant_data = baseline_data.copy()
    irrelevant_data['context'] = irrelevant_data['context'].apply(
        lambda x: x + " This is completely irrelevant information that has nothing to do with the topic. Random facts and unrelated content that will confuse the system."
    )
    scenarios.append(("Irrelevant Content", irrelevant_data))
    
    # Scenario 3: Remove key information
    removed_data = baseline_data.copy()
    removed_data['context'] = removed_data['context'].apply(
        lambda x: x.split('.')[0] + "."  # Keep only first sentence
    )
    scenarios.append(("Information Removal", removed_data))
    
    return scenarios


def main():
    """Main demonstration function showing drift detection workflow."""
    print("=" * 60)
    print("DRIFT DETECTION DEMONSTRATION")
    print("=" * 60)
    
    # Step 1: Create baseline data
    print("\n1. Creating baseline data...")
    baseline_data = create_drift_demo_data()
    test_data = baseline_data.sample(n=20, random_state=42).reset_index(drop=True)
    
    print(f"Baseline data: {len(baseline_data)} samples")
    print(f"Test data: {len(test_data)} samples")
    
    # Step 2: Build baseline RAG pipeline
    print("\n2. Building baseline RAG pipeline...")
    pipeline = RAGPipeline(
        embedding_model_name="all-MiniLM-L6-v2",
        chunk_size=500,
        chunk_overlap=50,
        top_k=5
    )
    
    pipeline.build_vectorstore(baseline_data)
    
    # Step 3: Evaluate baseline performance
    print("\n3. Evaluating baseline performance...")
    metrics_calculator = RAGMetrics()
    
    baseline_results = []
    for _, row in test_data.iterrows():
        result = pipeline.answer_question(row['question'], row['answer_text'])
        baseline_results.append(result)
    
    baseline_metrics = metrics_calculator.compute_comprehensive_metrics(baseline_results)
    
    print("Baseline Performance:")
    print(f"  Cosine Similarity: {baseline_metrics['retrieval']['avg_cosine_similarity']:.3f}")
    print(f"  Recall@K: {baseline_metrics['retrieval']['recall_at_k']:.3f}")
    print(f"  Exact Match: {baseline_metrics['generation']['exact_match']:.3f}")
    print(f"  F1 Score: {baseline_metrics['generation']['avg_f1']:.3f}")
    
    # Step 4: Create drift detector
    print("\n4. Creating drift detector...")
    drift_detector = DriftDetector(baseline_metrics, z_threshold=1.5)  # Lower threshold for demo
    
    # Step 5: Create drift scenarios
    print("\n5. Creating drift scenarios...")
    scenarios = create_drift_scenarios(baseline_data)
    
    # Step 6: Test drift detection
    print("\n6. Testing drift detection...")
    
    all_metrics = [baseline_metrics]
    drift_results = []
    
    for scenario_name, scenario_data in scenarios:
        print(f"\nTesting scenario: {scenario_name}")
        
        # Update pipeline with degraded data
        pipeline.update_knowledge_base(scenario_data)
        
        # Evaluate performance on degraded data
        scenario_results = []
        for _, row in test_data.iterrows():
            result = pipeline.answer_question(row['question'], row['answer_text'])
            scenario_results.append(result)
        
        scenario_metrics = metrics_calculator.compute_comprehensive_metrics(scenario_results)
        
        # Detect drift by comparing to baseline
        drift_result = drift_detector.detect_drift(scenario_metrics)
        
        all_metrics.append(scenario_metrics)
        drift_results.append(drift_result)
        
        print(f"  Drift detected: {drift_result['drift_detected']}")
        if drift_result['drift_detected']:
            print(f"  Drift reasons: {drift_result['threshold_results']['drift_reasons']}")
        
        # Show performance changes
        cos_sim_change = ((scenario_metrics['retrieval']['avg_cosine_similarity'] - 
                          baseline_metrics['retrieval']['avg_cosine_similarity']) / 
                         baseline_metrics['retrieval']['avg_cosine_similarity']) * 100
        f1_change = ((scenario_metrics['generation']['avg_f1'] - 
                     baseline_metrics['generation']['avg_f1']) / 
                    baseline_metrics['generation']['avg_f1']) * 100
        
        print(f"  Cosine Similarity change: {cos_sim_change:+.1f}%")
        print(f"  F1 Score change: {f1_change:+.1f}%")
    
    # Step 7: Create visualizations
    print("\n7. Creating visualizations...")
    
    # Create metrics summary with drift flags
    summary_data = []
    for i, metrics in enumerate(all_metrics):
        row = {
            'week': i,
            'avg_cosine_similarity': metrics['retrieval']['avg_cosine_similarity'],
            'recall_at_k': metrics['retrieval']['recall_at_k'],
            'exact_match': metrics['generation']['exact_match'],
            'avg_f1': metrics['generation']['avg_f1'],
            'avg_bert_score': metrics['generation']['avg_bert_score']
        }
        
        if i > 0:  # Skip baseline
            drift_result = drift_results[i-1]
            row['drift_detected'] = drift_result['drift_detected']
            row['drift_reasons'] = '; '.join(drift_result['threshold_results']['drift_reasons'])
            
            # Add drift flag for visualization
            if drift_result['drift_detected']:
                row['drift_flag'] = 'High'
            else:
                row['drift_flag'] = 'No'
        else:
            row['drift_detected'] = False
            row['drift_reasons'] = 'Baseline'
            row['drift_flag'] = 'Baseline'
        
        summary_data.append(row)
    
    metrics_summary = pd.DataFrame(summary_data)
    
    # Create visualizations
    visualizer = DriftVisualizer()
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Plot metrics over time
    visualizer.plot_metrics_over_time(metrics_summary, save_path="results/drift_demo_metrics.png")
    
    # Create drift heatmap
    visualizer.plot_drift_heatmap(metrics_summary, save_path="results/drift_demo_heatmap.png")
    
    # Create interactive dashboard
    visualizer.plot_interactive_dashboard(metrics_summary, save_path="results/drift_demo_dashboard.html")
    
    # Step 8: Print summary
    print("\n" + "=" * 60)
    print("DRIFT DEMONSTRATION COMPLETED")
    print("=" * 60)
    
    print(f"\nResults:")
    print(f"  - Total scenarios tested: {len(scenarios)}")
    print(f"  - Scenarios with drift: {sum(1 for r in drift_results if r['drift_detected'])}")
    print(f"  - Drift rate: {sum(1 for r in drift_results if r['drift_detected']) / len(scenarios) * 100:.1f}%")
    
    print(f"\nVisualizations saved to:")
    print(f"  - results/drift_demo_metrics.png")
    print(f"  - results/drift_demo_heatmap.png")
    print(f"  - results/drift_demo_dashboard.html")
    
    print(f"\nKey findings:")
    for i, (scenario_name, _) in enumerate(scenarios):
        drift_detected = drift_results[i]['drift_detected']
        print(f"  - {scenario_name}: {'DRIFT DETECTED' if drift_detected else 'No drift'}")
    
    return baseline_metrics, drift_detector, all_metrics, drift_results


if __name__ == "__main__":
    # Run the drift demonstration
    baseline_metrics, drift_detector, all_metrics, drift_results = main()
    
    print("\nDrift demonstration completed successfully!")
