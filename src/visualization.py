"""
Visualization tools for RAG drift detection analysis.

This module provides comprehensive visualization capabilities for analyzing
drift detection results in Retrieval-Augmented Generation systems.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class DriftVisualizer:
    """Visualization tools for drift detection analysis."""
    
    def __init__(self, figsize: tuple = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size for matplotlib plots
        """
        self.figsize = figsize
        
    def plot_metrics_over_time(self, 
                              metrics_summary: pd.DataFrame,
                              save_path: Optional[str] = None) -> None:
        """
        Plot metrics over time with drift indicators.
        
        Args:
            metrics_summary: DataFrame with metrics over time
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RAG Pipeline Metrics Over Time', fontsize=16, fontweight='bold')
        
        # Plot 1: Cosine Similarity
        ax1 = axes[0, 0]
        ax1.plot(metrics_summary['week'], metrics_summary['avg_cosine_similarity'], 
                marker='o', linewidth=2, markersize=6)
        ax1.set_title('Average Cosine Similarity')
        ax1.set_xlabel('Week')
        ax1.set_ylabel('Cosine Similarity')
        ax1.grid(True, alpha=0.3)
        
        # Add drift indicators
        if 'drift_flag' in metrics_summary.columns:
            for i, drift_flag in enumerate(metrics_summary['drift_flag']):
                if drift_flag == 'High':
                    ax1.scatter(metrics_summary.iloc[i]['week'], 
                               metrics_summary.iloc[i]['avg_cosine_similarity'],
                               color='red', s=100, zorder=5, marker='x')
                elif drift_flag == 'Slight':
                    ax1.scatter(metrics_summary.iloc[i]['week'], 
                               metrics_summary.iloc[i]['avg_cosine_similarity'],
                               color='orange', s=80, zorder=5, marker='^')
        
        # Plot 2: Recall@K
        ax2 = axes[0, 1]
        ax2.plot(metrics_summary['week'], metrics_summary['recall_at_k'], 
                marker='s', linewidth=2, markersize=6, color='green')
        ax2.set_title('Recall@K')
        ax2.set_xlabel('Week')
        ax2.set_ylabel('Recall@K')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Exact Match
        ax3 = axes[1, 0]
        ax3.plot(metrics_summary['week'], metrics_summary['exact_match'], 
                marker='d', linewidth=2, markersize=6, color='purple')
        ax3.set_title('Exact Match')
        ax3.set_xlabel('Week')
        ax3.set_ylabel('Exact Match')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: F1 Score
        ax4 = axes[1, 1]
        ax4.plot(metrics_summary['week'], metrics_summary['avg_f1'], 
                marker='*', linewidth=2, markersize=8, color='brown')
        ax4.set_title('Average F1 Score')
        ax4.set_xlabel('Week')
        ax4.set_ylabel('F1 Score')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_drift_heatmap(self, 
                          metrics_summary: pd.DataFrame,
                          save_path: Optional[str] = None) -> None:
        """
        Create a heatmap showing drift severity over time.
        
        Args:
            metrics_summary: DataFrame with metrics over time
            save_path: Optional path to save the plot
        """
        # Create drift severity matrix
        metrics = ['avg_cosine_similarity', 'recall_at_k', 'exact_match', 'avg_f1']
        drift_matrix = []
        
        for _, row in metrics_summary.iterrows():
            week_drift = []
            for metric in metrics:
                if metric in ['cosine_sim_z', 'f1_z'] and metric in row:
                    # Use z-scores if available
                    z_score = row[metric]
                    if z_score > 2.0:
                        severity = 3  # High
                    elif z_score > 1.0:
                        severity = 2  # Medium
                    else:
                        severity = 1  # Low
                    week_drift.append(severity)
                else:
                    # For other metrics, use relative performance
                    baseline = metrics_summary.iloc[0][metric]
                    current = row[metric]
                    if current < baseline * 0.9:
                        severity = 3
                    elif current < baseline * 0.95:
                        severity = 2
                    else:
                        severity = 1
                    week_drift.append(severity)
            drift_matrix.append(week_drift)
        
        drift_df = pd.DataFrame(drift_matrix, 
                               columns=['Cosine Sim', 'Recall@K', 'Exact Match', 'F1 Score'],
                               index=metrics_summary['week'])
        
        # Create heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(drift_df.T, annot=True, cmap='RdYlGn_r', 
                   cbar_kws={'label': 'Drift Severity (1=Low, 2=Medium, 3=High)'})
        plt.title('Drift Severity Heatmap Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Week')
        plt.ylabel('Metrics')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_interactive_dashboard(self, 
                                  metrics_summary: pd.DataFrame,
                                  save_path: Optional[str] = None) -> None:
        """
        Create an interactive dashboard using Plotly.
        
        Args:
            metrics_summary: DataFrame with metrics over time
            save_path: Optional path to save the HTML file
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cosine Similarity', 'Recall@K', 'Exact Match', 'F1 Score'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add traces
        colors = ['blue', 'green', 'purple', 'brown']
        metrics = ['avg_cosine_similarity', 'recall_at_k', 'exact_match', 'avg_f1']
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            # Main metric line
            fig.add_trace(
                go.Scatter(
                    x=metrics_summary['week'],
                    y=metrics_summary[metric],
                    mode='lines+markers',
                    name=metric.replace('_', ' ').title(),
                    line=dict(color=color, width=3),
                    marker=dict(size=8)
                ),
                row=row, col=col
            )
            
            # Add drift indicators
            if 'drift_flag' in metrics_summary.columns:
                high_drift = metrics_summary[metrics_summary['drift_flag'] == 'High']
                slight_drift = metrics_summary[metrics_summary['drift_flag'] == 'Slight']
            else:
                high_drift = pd.DataFrame()
                slight_drift = pd.DataFrame()
            
            if not high_drift.empty:
                fig.add_trace(
                    go.Scatter(
                        x=high_drift['week'],
                        y=high_drift[metric],
                        mode='markers',
                        name='High Drift',
                        marker=dict(color='red', size=12, symbol='x'),
                        showlegend=False
                    ),
                    row=row, col=col
                )
            
            if not slight_drift.empty:
                fig.add_trace(
                    go.Scatter(
                        x=slight_drift['week'],
                        y=slight_drift[metric],
                        mode='markers',
                        name='Slight Drift',
                        marker=dict(color='orange', size=10, symbol='triangle-up'),
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        # Update layout
        fig.update_layout(
            title_text="RAG Pipeline Drift Detection Dashboard",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_xaxes(title_text="Week", row=i, col=j)
                fig.update_yaxes(title_text="Score", row=i, col=j)
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def plot_z_score_analysis(self, 
                             metrics_summary: pd.DataFrame,
                             save_path: Optional[str] = None) -> None:
        """
        Plot Z-score analysis for drift detection.
        
        Args:
            metrics_summary: DataFrame with metrics over time
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle('Z-Score Analysis for Drift Detection', fontsize=16, fontweight='bold')
        
        # Plot Z-scores
        ax1 = axes[0]
        ax1.plot(metrics_summary['week'], metrics_summary['cosine_sim_z'], 
                marker='o', linewidth=2, label='Cosine Similarity Z-Score', color='blue')
        ax1.plot(metrics_summary['week'], metrics_summary['f1_z'], 
                marker='s', linewidth=2, label='F1 Score Z-Score', color='red')
        
        # Add threshold lines
        ax1.axhline(y=2.0, color='orange', linestyle='--', alpha=0.7, label='High Drift Threshold (z=2.0)')
        ax1.axhline(y=1.0, color='yellow', linestyle='--', alpha=0.7, label='Slight Drift Threshold (z=1.0)')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax1.set_title('Z-Scores Over Time')
        ax1.set_xlabel('Week')
        ax1.set_ylabel('Z-Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot drift flags
        ax2 = axes[1]
        drift_colors = {'No': 'green', 'Slight': 'orange', 'High': 'red', 'Baseline': 'gray'}
        colors = [drift_colors[flag] for flag in metrics_summary['drift_flag']]
        
        bars = ax2.bar(metrics_summary['week'], 
                      [1 if flag != 'Baseline' else 0 for flag in metrics_summary['drift_flag']],
                      color=colors, alpha=0.7)
        
        ax2.set_title('Drift Flags Over Time')
        ax2.set_xlabel('Week')
        ax2.set_ylabel('Drift Detected')
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['No', 'Yes'])
        ax2.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=flag) 
                          for flag, color in drift_colors.items()]
        ax2.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_metrics_distribution(self, 
                                 metrics_summary: pd.DataFrame,
                                 save_path: Optional[str] = None) -> None:
        """
        Plot distribution of metrics to understand baseline vs current performance.
        
        Args:
            metrics_summary: DataFrame with metrics over time
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Metrics Distribution Analysis', fontsize=16, fontweight='bold')
        
        metrics = ['avg_cosine_similarity', 'recall_at_k', 'exact_match', 'avg_f1']
        titles = ['Cosine Similarity', 'Recall@K', 'Exact Match', 'F1 Score']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            # Plot histogram
            ax.hist(metrics_summary[metric], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            
            # Add vertical line for baseline
            baseline = metrics_summary.iloc[0][metric]
            ax.axvline(baseline, color='red', linestyle='--', linewidth=2, 
                      label=f'Baseline: {baseline:.3f}')
            
            # Add vertical line for mean
            mean_val = metrics_summary[metric].mean()
            ax.axvline(mean_val, color='green', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_val:.3f}')
            
            ax.set_title(title)
            ax.set_xlabel('Score')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_drift_report(self, 
                           metrics_summary: pd.DataFrame,
                           drift_alerts: List[Dict],
                           save_path: Optional[str] = None) -> str:
        """
        Create a comprehensive drift detection report.
        
        Args:
            metrics_summary: DataFrame with metrics over time
            drift_alerts: List of drift alerts
            save_path: Optional path to save the report
            
        Returns:
            Report text
        """
        report = []
        report.append("=" * 60)
        report.append("RAG PIPELINE DRIFT DETECTION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary statistics
        report.append("SUMMARY STATISTICS:")
        report.append("-" * 30)
        baseline = metrics_summary.iloc[0]
        latest = metrics_summary.iloc[-1]
        
        metrics = ['avg_cosine_similarity', 'recall_at_k', 'exact_match', 'avg_f1']
        metric_names = ['Cosine Similarity', 'Recall@K', 'Exact Match', 'F1 Score']
        
        for metric, name in zip(metrics, metric_names):
            baseline_val = baseline[metric]
            latest_val = latest[metric]
            change = ((latest_val - baseline_val) / baseline_val) * 100
            
            report.append(f"{name}:")
            report.append(f"  Baseline: {baseline_val:.3f}")
            report.append(f"  Latest: {latest_val:.3f}")
            report.append(f"  Change: {change:+.2f}%")
            report.append("")
        
        # Drift analysis
        report.append("DRIFT ANALYSIS:")
        report.append("-" * 30)
        
        high_drift_weeks = metrics_summary[metrics_summary['drift_flag'] == 'High']['week'].tolist()
        slight_drift_weeks = metrics_summary[metrics_summary['drift_flag'] == 'Slight']['week'].tolist()
        
        report.append(f"High Drift Detected: {len(high_drift_weeks)} weeks")
        if high_drift_weeks:
            report.append(f"  Weeks: {high_drift_weeks}")
        
        report.append(f"Slight Drift Detected: {len(slight_drift_weeks)} weeks")
        if slight_drift_weeks:
            report.append(f"  Weeks: {slight_drift_weeks}")
        
        report.append("")
        
        # Alerts summary
        report.append("DRIFT ALERTS:")
        report.append("-" * 30)
        
        if drift_alerts:
            for i, alert in enumerate(drift_alerts, 1):
                report.append(f"Alert {i}:")
                report.append(f"  Week: {alert['timestamp']}")
                report.append(f"  Severity: {alert['severity']}")
                report.append(f"  Types: {', '.join(alert['drift_type'])}")
                report.append("")
        else:
            report.append("No drift alerts generated.")
        
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text


def create_comprehensive_visualization(metrics_summary: pd.DataFrame,
                                     drift_alerts: List[Dict],
                                     output_dir: str = "results") -> None:
    """
    Create comprehensive visualization suite.
    
    Args:
        metrics_summary: DataFrame with metrics over time
        drift_alerts: List of drift alerts
        output_dir: Directory to save visualizations
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = DriftVisualizer()
    
    # Create all visualizations
    visualizer.plot_metrics_over_time(metrics_summary, 
                                     save_path=f"{output_dir}/metrics_over_time.png")
    
    visualizer.plot_drift_heatmap(metrics_summary, 
                                 save_path=f"{output_dir}/drift_heatmap.png")
    
    visualizer.plot_interactive_dashboard(metrics_summary, 
                                        save_path=f"{output_dir}/interactive_dashboard.html")
    
    visualizer.plot_z_score_analysis(metrics_summary, 
                                   save_path=f"{output_dir}/z_score_analysis.png")
    
    visualizer.plot_metrics_distribution(metrics_summary, 
                                       save_path=f"{output_dir}/metrics_distribution.png")
    
    # Create report
    report = visualizer.create_drift_report(metrics_summary, drift_alerts,
                                          save_path=f"{output_dir}/drift_report.txt")
    
    print(f"All visualizations saved to {output_dir}/")
    print("\nDrift Report:")
    print(report)


if __name__ == "__main__":
    # Example usage
    # Create sample data
    sample_data = pd.DataFrame({
        'week': range(1, 9),
        'avg_cosine_similarity': [0.89, 0.85, 0.82, 0.78, 0.75, 0.72, 0.70, 0.68],
        'recall_at_k': [0.78, 0.76, 0.74, 0.71, 0.68, 0.65, 0.62, 0.60],
        'exact_match': [0.65, 0.63, 0.60, 0.58, 0.55, 0.52, 0.50, 0.48],
        'avg_f1': [0.72, 0.70, 0.68, 0.65, 0.62, 0.59, 0.56, 0.54],
        'avg_bert_score': [0.81, 0.79, 0.77, 0.75, 0.73, 0.71, 0.69, 0.67],
        'drift_flag': ['Baseline', 'No', 'Slight', 'Slight', 'High', 'High', 'High', 'High'],
        'cosine_sim_z': [0.0, 0.4, 0.7, 1.1, 1.4, 1.7, 1.9, 2.1],
        'f1_z': [0.0, 0.2, 0.4, 0.7, 1.0, 1.3, 1.6, 1.8]
    })
    
    # Create visualizations
    visualizer = DriftVisualizer()
    visualizer.plot_metrics_over_time(sample_data)
    visualizer.plot_drift_heatmap(sample_data)
    visualizer.plot_z_score_analysis(sample_data)
