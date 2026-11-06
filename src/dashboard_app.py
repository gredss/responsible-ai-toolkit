"""
Dashboard Application for IndoBERT Clickbait Detection System

This module implements an interactive Streamlit dashboard with four main modules:
1. Single-Text Prediction: Real-time prediction with perturbation testing
2. Domain Shift Matrix: 5x5 heatmap visualization of cross-domain performance
3. Robustness Analysis: Performance degradation curves across perturbation levels
4. Reliability Summary: Automated report generation with statistical insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from typing import Dict, List, Optional, Any, Tuple
import logging

# Import project modules
from config import config as _app_config
from model_trainer import ModelTrainer
from perturbation_engine import PerturbationEngine
from evaluation_engine import MetricsCalculator
from statistical_analyzer import StatisticalAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DashboardConfig:
    """
    Dashboard-level constants derived from the central :class:`config.Config`
    singleton.  Read values from ``config.dashboard`` and ``config.data``
    instead of duplicating them here, so there is a single source of truth.
    """

    DOMAINS: list = _app_config.data.DOMAINS
    # The two perturbation mechanisms and their three intensity levels.
    # 'clean' is the shared unperturbed baseline (shown alongside both).
    PERTURBATION_TYPES: list = ['semantic', 'typo']
    PERTURBATION_LEVELS: list = ['low', 'medium', 'high']
    # Full ordered sequence used when a chart shows clean -> increasing noise.
    PERTURBATION_SEQUENCE: list = ['clean', 'low', 'medium', 'high']
    PERTURBATION_TYPE_LABELS: dict = {
        'semantic': 'Semantic (synonym substitution)',
        'typo': 'Typo (character noise)',
    }
    MODEL_VARIANTS: list = [
        'indobert-base-p1', 'indobert-large-p1', 'indobert-lite-base-p1'
    ]

    # Visualisation — delegate to config.dashboard
    HEATMAP_COLORSCALE: str         = _app_config.dashboard.HEATMAP_COLORSCALE
    DEGRADATION_COLORS: dict        = _app_config.dashboard.DEGRADATION_COLORS

    # Performance thresholds — delegate to config.evaluation
    GOOD_PERFORMANCE_THRESHOLD: float       = _app_config.evaluation.GOOD_PERFORMANCE_THRESHOLD
    ACCEPTABLE_PERFORMANCE_THRESHOLD: float = _app_config.evaluation.ACCEPTABLE_PERFORMANCE_THRESHOLD


class ModelCache:
    """Manages model loading and caching for performance."""
    
    def __init__(self):
        """Initialize model cache."""
        self.models = {}
        self.tokenizers = {}
        logger.info("ModelCache initialized")
    
    @st.cache_resource
    def load_model(_self, model_path: str, model_name: str) -> ModelTrainer:
        """
        Load and cache a trained model.
        
        Args:
            model_path: Path to model checkpoint
            model_name: Name/identifier for the model
            
        Returns:
            ModelTrainer instance with loaded model
        """
        if model_name not in _self.models:
            logger.info(f"Loading model: {model_name}")
            trainer = ModelTrainer(model_name=model_name)
            trainer.load_checkpoint(model_path)
            _self.models[model_name] = trainer
            logger.info(f"Model {model_name} loaded and cached")
        
        return _self.models[model_name]
    
    @st.cache_data
    def load_evaluation_results(_self, results_path: str) -> Dict[str, Any]:
        """
        Load and cache evaluation results.
        
        Args:
            results_path: Path to evaluation results JSON
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Loading evaluation results from {results_path}")
        with open(results_path, 'r') as f:
            results = json.load(f)
        return results


class SingleTextPredictor:
    """Module 1: Single-text prediction with perturbation testing."""
    
    def __init__(self, model_trainer: ModelTrainer, perturbation_engine: PerturbationEngine):
        """
        Initialize single-text predictor.
        
        Args:
            model_trainer: Trained model for predictions
            perturbation_engine: Engine for text perturbations
        """
        self.model_trainer = model_trainer
        self.perturbation_engine = perturbation_engine
        self.metrics_calculator = MetricsCalculator()
    
    def render(self):
        """Render the single-text prediction interface."""
        st.header("📝 Single-Text Prediction")
        st.markdown("Test clickbait detection on custom text with optional perturbations.")
        
        # Text input
        col1, col2 = st.columns([2, 1])
        
        with col1:
            input_text = st.text_area(
                "Enter Indonesian text to analyze:",
                height=150,
                placeholder="Masukkan teks berita dalam Bahasa Indonesia..."
            )
        
        with col2:
            st.markdown("**Perturbation Settings**")
            apply_perturbation = st.checkbox("Apply perturbation", value=False)
            
            if apply_perturbation:
                # Pick the mechanism first, then the intensity level.
                perturbation_type = st.selectbox(
                    "Perturbation type:",
                    options=['semantic', 'typo'],
                    index=0,
                    format_func=lambda t: (
                        "Semantic (synonym)" if t == "semantic"
                        else "Typo (character noise)"
                    ),
                )
                perturbation_level = st.selectbox(
                    "Perturbation level:",
                    options=['low', 'medium', 'high'],
                    index=0,
                )
            else:
                perturbation_type = None
                perturbation_level = None
        
        # Predict button
        if st.button("🔍 Analyze Text", type="primary"):
            if not input_text.strip():
                st.warning("Please enter some text to analyze.")
                return
            
            with st.spinner("Analyzing..."):
                # Clean prediction
                clean_pred, clean_proba = self.model_trainer.predict([input_text])
                clean_label = int(clean_pred[0])
                clean_confidence = float(clean_proba[0][clean_label])
                
                # Display clean results
                st.subheader("Clean Text Analysis")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Prediction",
                        "Clickbait" if clean_label == 1 else "Non-Clickbait",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "Confidence",
                        f"{clean_confidence:.2%}",
                        delta=None
                    )
                
                with col3:
                    reliability = "High" if clean_confidence > 0.8 else "Medium" if clean_confidence > 0.6 else "Low"
                    st.metric("Reliability", reliability)
                
                # Probability distribution
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Non-Clickbait', 'Clickbait'],
                        y=[clean_proba[0][0], clean_proba[0][1]],
                        marker_color=['#2ecc71', '#e74c3c']
                    )
                ])
                fig.update_layout(
                    title="Prediction Probabilities",
                    yaxis_title="Probability",
                    showlegend=False,
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Perturbed prediction
                if apply_perturbation and perturbation_level:
                    # The engine uses distinct level names per mechanism:
                    #   semantic -> "low"/"medium"/"high"
                    #   typo     -> "typo_low"/"typo_medium"/"typo_high"
                    if perturbation_type == "typo":
                        engine_level = f"typo_{perturbation_level}"
                    else:
                        engine_level = perturbation_level

                    st.subheader(
                        f"Perturbed Text Analysis "
                        f"({perturbation_type.capitalize()} — "
                        f"{perturbation_level.capitalize()} Level)"
                    )

                    perturbed_text = self.perturbation_engine.apply_perturbation(
                        input_text,
                        engine_level
                    )
                    
                    # Show perturbed text
                    with st.expander("View Perturbed Text"):
                        st.text_area("Perturbed version:", perturbed_text, height=100, disabled=True)
                    
                    # Predict on perturbed
                    pert_pred, pert_proba = self.model_trainer.predict([perturbed_text])
                    pert_label = int(pert_pred[0])
                    pert_confidence = float(pert_proba[0][pert_label])
                    
                    # Display perturbed results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        prediction_changed = clean_label != pert_label
                        st.metric(
                            "Prediction",
                            "Clickbait" if pert_label == 1 else "Non-Clickbait",
                            delta="Changed" if prediction_changed else "Stable",
                            delta_color="inverse" if prediction_changed else "normal"
                        )
                    
                    with col2:
                        confidence_drop = clean_confidence - pert_confidence
                        st.metric(
                            "Confidence",
                            f"{pert_confidence:.2%}",
                            delta=f"{confidence_drop:.2%}",
                            delta_color="inverse"
                        )
                    
                    with col3:
                        robustness = "Robust" if abs(confidence_drop) < 0.1 else "Moderate" if abs(confidence_drop) < 0.2 else "Vulnerable"
                        st.metric("Robustness", robustness)
                    
                    # Side-by-side comparison
                    comparison_fig = go.Figure(data=[
                        go.Bar(
                            name='Clean',
                            x=['Non-Clickbait', 'Clickbait'],
                            y=[clean_proba[0][0], clean_proba[0][1]],
                            marker_color='#3498db'
                        ),
                        go.Bar(
                            name='Perturbed',
                            x=['Non-Clickbait', 'Clickbait'],
                            y=[pert_proba[0][0], pert_proba[0][1]],
                            marker_color='#e67e22'
                        )
                    ])
                    comparison_fig.update_layout(
                        title="Clean vs Perturbed Comparison",
                        yaxis_title="Probability",
                        barmode='group',
                        height=300
                    )
                    st.plotly_chart(comparison_fig, use_container_width=True)


class DomainShiftMatrix:
    """Module 2: Domain shift matrix visualization."""
    
    def __init__(self, evaluation_results: Dict[str, Any]):
        """
        Initialize domain shift matrix visualizer.
        
        Args:
            evaluation_results: Results from evaluation engine
        """
        self.evaluation_results = evaluation_results
    
    def render(self):
        """Render the domain shift matrix visualization."""
        st.header("🔄 Domain Shift Matrix")
        st.markdown("Cross-domain performance analysis showing how models perform across different domains.")
        
        # Metric selection
        metric = st.selectbox(
            "Select metric to visualize:",
            options=['f1', 'accuracy', 'precision', 'recall'],
            index=0,
            format_func=lambda x: x.upper() if x == 'f1' else x.capitalize()
        )
        
        # Extract cross-domain results
        if 'cross_domain' not in self.evaluation_results:
            st.warning("No cross-domain evaluation results available.")
            return
        
        # Create performance matrix
        matrix_data = self._create_performance_matrix(metric)
        
        if matrix_data is None:
            st.error("Unable to create performance matrix.")
            return
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix_data.values,
            x=matrix_data.columns,
            y=matrix_data.index,
            colorscale=DashboardConfig.HEATMAP_COLORSCALE,
            text=matrix_data.values,
            texttemplate='%{text:.3f}',
            textfont={"size": 10},
            colorbar=dict(title=metric.upper()),
            hoverongaps=False,
            hovertemplate='Source: %{y}<br>Target: %{x}<br>' + metric.upper() + ': %{z:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Cross-Domain Performance Matrix ({metric.upper()})",
            xaxis_title="Target Domain",
            yaxis_title="Source Domain (Trained On)",
            height=500,
            width=700
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.subheader("Matrix Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Performance", f"{matrix_data.values.mean():.4f}")
        
        with col2:
            diagonal = np.diag(matrix_data.values)
            st.metric("In-Domain Avg", f"{diagonal.mean():.4f}")
        
        with col3:
            off_diagonal = matrix_data.values[~np.eye(len(matrix_data), dtype=bool)]
            st.metric("Cross-Domain Avg", f"{off_diagonal.mean():.4f}")
        
        with col4:
            domain_shift = diagonal.mean() - off_diagonal.mean()
            st.metric("Avg Domain Shift", f"{domain_shift:.4f}", delta_color="inverse")
        
        # Detailed table
        with st.expander("View Detailed Matrix"):
            st.dataframe(matrix_data.style.background_gradient(cmap='RdYlGn', axis=None))
    
    def _create_performance_matrix(self, metric: str) -> Optional[pd.DataFrame]:
        """
        Create performance matrix from evaluation results.
        
        Args:
            metric: Metric to extract
            
        Returns:
            DataFrame with performance matrix
        """
        cross_domain_results = self.evaluation_results.get('cross_domain', {})
        
        if not cross_domain_results:
            return None
        
        # Extract domains
        domains = sorted(set(
            [k[0] for k in cross_domain_results.keys()] +
            [k[1] for k in cross_domain_results.keys()]
        ))
        
        # Create matrix
        matrix = pd.DataFrame(index=domains, columns=domains, dtype=float)
        
        for (source, target), results in cross_domain_results.items():
            if metric in results.get('metrics', {}):
                matrix.loc[source, target] = results['metrics'][metric]
        
        return matrix


class RobustnessAnalyzer:
    """Module 3: Robustness analysis with degradation curves."""
    
    def __init__(self, evaluation_results: Dict[str, Any]):
        """
        Initialize robustness analyzer.
        
        Args:
            evaluation_results: Results from evaluation engine
        """
        self.evaluation_results = evaluation_results
    
    def render(self):
        """Render the robustness analysis interface."""
        st.header("📊 Robustness Analysis")
        st.markdown("Performance degradation analysis across perturbation levels.")
        
        if 'perturbation' not in self.evaluation_results:
            st.warning("No perturbation evaluation results available.")
            return
        
        # Domain selection
        available_domains = list(self.evaluation_results['perturbation'].keys())
        selected_domain = st.selectbox(
            "Select domain:",
            options=available_domains,
            index=0
        )
        
        # Metric selection
        metric = st.selectbox(
            "Select metric:",
            options=['f1', 'accuracy', 'precision', 'recall'],
            index=0,
            format_func=lambda x: x.upper() if x == 'f1' else x.capitalize()
        )
        
        # Extract perturbation data for the selected domain.
        # Nested shape: {'clean': {...}, 'semantic': {low,medium,high}, 'typo': {...}}
        pert_data = self.evaluation_results['perturbation'][selected_domain]

        # Shared clean baseline (same unperturbed data for both mechanisms).
        clean_score = self._clean_score(pert_data, metric)

        # Plot ONE degradation line per mechanism, each starting from clean.
        fig = go.Figure()
        for ptype in DashboardConfig.PERTURBATION_TYPES:
            scores = [clean_score] + [
                self._level_score(pert_data, ptype, level, metric)
                for level in DashboardConfig.PERTURBATION_LEVELS
            ]
            fig.add_trace(go.Scatter(
                x=DashboardConfig.PERTURBATION_SEQUENCE,
                y=scores,
                mode='lines+markers',
                name=DashboardConfig.PERTURBATION_TYPE_LABELS[ptype],
                line=dict(width=3),
                marker=dict(size=10)
            ))

        # Add threshold lines
        fig.add_hline(
            y=DashboardConfig.GOOD_PERFORMANCE_THRESHOLD,
            line_dash="dash",
            line_color="green",
            annotation_text="Good Performance"
        )
        fig.add_hline(
            y=DashboardConfig.ACCEPTABLE_PERFORMANCE_THRESHOLD,
            line_dash="dash",
            line_color="orange",
            annotation_text="Acceptable Performance"
        )
        fig.update_layout(
            title=f"Performance Degradation Curve - {selected_domain} ({metric.upper()})",
            xaxis_title="Perturbation Level",
            yaxis_title=metric.upper() + " Score",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Degradation statistics — reported PER mechanism so the two are not
        # conflated. "High" is looked up explicitly (no fixed list index).
        st.subheader("Degradation Statistics (Clean → High)")
        st.metric("Clean Performance", f"{clean_score:.4f}")

        for ptype in DashboardConfig.PERTURBATION_TYPES:
            high_score = self._level_score(pert_data, ptype, 'high', metric)
            abs_drop = clean_score - high_score
            rel_drop = (abs_drop / clean_score * 100) if clean_score > 0 else 0

            st.markdown(f"**{DashboardConfig.PERTURBATION_TYPE_LABELS[ptype]}**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("High Pert. Performance", f"{high_score:.4f}")
            with col2:
                st.metric("Absolute Drop", f"{abs_drop:.4f}", delta_color="inverse")
            with col3:
                st.metric("Relative Drop", f"{rel_drop:.2f}%", delta_color="inverse")

        # Multi-domain comparison
        if st.checkbox("Compare across all domains"):
            self._render_multi_domain_comparison(metric)

    @staticmethod
    def _clean_score(pert_data: Dict[str, Any], metric: str) -> float:
        """Shared clean-baseline metric for a domain (0 if missing)."""
        return (
            pert_data.get('clean', {})
            .get('metrics', {})
            .get(metric, 0)
        )

    @staticmethod
    def _level_score(pert_data: Dict[str, Any], ptype: str,
                     level: str, metric: str) -> float:
        """One perturbed metric from the nested structure (0 if missing)."""
        return (
            pert_data.get(ptype, {})
            .get(level, {})
            .get('metrics', {})
            .get(metric, 0)
        )
    
    def _render_multi_domain_comparison(self, metric: str):
        """Render multi-domain comparison plot."""
        st.subheader("Multi-Domain Comparison")

        pert_results = self.evaluation_results['perturbation']

        # One chart PER mechanism so the two are not visually conflated.
        # Each chart shows all domains across clean -> low -> medium -> high,
        # with the shared clean baseline as the first point.
        for ptype in DashboardConfig.PERTURBATION_TYPES:
            fig = go.Figure()
            for domain, pert_data in pert_results.items():
                scores = [self._clean_score(pert_data, metric)] + [
                    self._level_score(pert_data, ptype, level, metric)
                    for level in DashboardConfig.PERTURBATION_LEVELS
                ]
                fig.add_trace(go.Scatter(
                    x=DashboardConfig.PERTURBATION_SEQUENCE,
                    y=scores,
                    mode='lines+markers',
                    name=domain,
                    line=dict(width=2),
                    marker=dict(size=8)
                ))

            fig.update_layout(
                title=(
                    f"Multi-Domain Robustness — "
                    f"{DashboardConfig.PERTURBATION_TYPE_LABELS[ptype]} "
                    f"({metric.upper()})"
                ),
                xaxis_title="Perturbation Level",
                yaxis_title=metric.upper() + " Score",
                height=450,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)


class ReliabilitySummary:
    """Module 4: Reliability summary and report generation."""
    
    def __init__(
        self,
        evaluation_results: Dict[str, Any],
        statistical_results: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize reliability summary generator.
        
        Args:
            evaluation_results: Results from evaluation engine
            statistical_results: Optional statistical analysis results
        """
        self.evaluation_results = evaluation_results
        self.statistical_results = statistical_results
    
    def render(self):
        """Render the reliability summary interface."""
        st.header("📋 Reliability Summary")
        st.markdown("Comprehensive reliability assessment with statistical insights.")
        
        # Overall performance summary
        st.subheader("Overall Performance Summary")
        self._render_performance_summary()
        
        # Best and worst case scenarios
        st.subheader("Best & Worst Case Scenarios")
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_best_case()
        
        with col2:
            self._render_worst_case()
        
        # Recommendations
        st.subheader("Recommendations")
        self._render_recommendations()
        
        # Export options
        st.subheader("Export Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Generate Text Report"):
                report = self._generate_text_report()
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name="reliability_summary.txt",
                    mime="text/plain"
                )
        
        with col2:
            if st.button("📊 Export Summary Table"):
                summary_df = self._create_summary_table()
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="summary_table.csv",
                    mime="text/csv"
                )
    
    def _render_performance_summary(self):
        """Render overall performance summary."""
        if 'in_domain' not in self.evaluation_results:
            st.warning("No in-domain results available.")
            return
        
        in_domain_results = self.evaluation_results['in_domain']
        
        # Calculate aggregate metrics
        f1_scores = [results['metrics']['f1'] for results in in_domain_results.values()]
        acc_scores = [results['metrics']['accuracy'] for results in in_domain_results.values()]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Avg F1-Score", f"{np.mean(f1_scores):.4f}")
        
        with col2:
            st.metric("Avg Accuracy", f"{np.mean(acc_scores):.4f}")
        
        with col3:
            st.metric("Best F1", f"{np.max(f1_scores):.4f}")
        
        with col4:
            st.metric("Worst F1", f"{np.min(f1_scores):.4f}")
    
    def _render_best_case(self):
        """Render best case scenario."""
        st.markdown("**🌟 Best Performance**")
        
        if 'in_domain' in self.evaluation_results:
            in_domain_results = self.evaluation_results['in_domain']
            
            best_domain = max(
                in_domain_results.items(),
                key=lambda x: x[1]['metrics']['f1']
            )
            
            domain_name = best_domain[0]
            metrics = best_domain[1]['metrics']
            
            st.success(f"**Domain:** {domain_name}")
            st.write(f"- F1-Score: {metrics['f1']:.4f}")
            st.write(f"- Accuracy: {metrics['accuracy']:.4f}")
            st.write(f"- Precision: {metrics['precision']:.4f}")
            st.write(f"- Recall: {metrics['recall']:.4f}")
    
    def _render_worst_case(self):
        """Render worst case scenario."""
        st.markdown("**⚠️ Worst Performance**")
        
        if 'in_domain' in self.evaluation_results:
            in_domain_results = self.evaluation_results['in_domain']
            
            worst_domain = min(
                in_domain_results.items(),
                key=lambda x: x[1]['metrics']['f1']
            )
            
            domain_name = worst_domain[0]
            metrics = worst_domain[1]['metrics']
            
            st.warning(f"**Domain:** {domain_name}")
            st.write(f"- F1-Score: {metrics['f1']:.4f}")
            st.write(f"- Accuracy: {metrics['accuracy']:.4f}")
            st.write(f"- Precision: {metrics['precision']:.4f}")
            st.write(f"- Recall: {metrics['recall']:.4f}")
    
    def _render_recommendations(self):
        """Render recommendations based on analysis."""
        recommendations = []
        
        # Analyze in-domain performance
        if 'in_domain' in self.evaluation_results:
            in_domain_results = self.evaluation_results['in_domain']
            f1_scores = [results['metrics']['f1'] for results in in_domain_results.values()]
            avg_f1 = np.mean(f1_scores)
            
            if avg_f1 >= 0.85:
                recommendations.append("✅ Excellent overall performance. Model is production-ready.")
            elif avg_f1 >= 0.75:
                recommendations.append("✓ Good performance. Consider fine-tuning for specific domains.")
            else:
                recommendations.append("⚠️ Performance below target. Additional training recommended.")
        
        # Analyze robustness
        if 'perturbation' in self.evaluation_results:
            pert_results = self.evaluation_results['perturbation']

            # Report robustness PER mechanism (semantic / typo) so a
            # recommendation is not averaged across two different phenomena.
            # "High" degradation = (clean F1 - high-level F1) / clean F1.
            for ptype in DashboardConfig.PERTURBATION_TYPES:
                high_degradations = []
                for domain, pert_data in pert_results.items():
                    clean_block = pert_data.get('clean', {})
                    high_block = pert_data.get(ptype, {}).get('high', {})
                    if not clean_block or not high_block:
                        continue
                    clean_f1 = clean_block['metrics']['f1']
                    high_f1 = high_block['metrics']['f1']
                    degradation = (clean_f1 - high_f1) / clean_f1 if clean_f1 > 0 else 0
                    high_degradations.append(degradation)

                if not high_degradations:
                    continue

                avg_degradation = np.mean(high_degradations)
                label = DashboardConfig.PERTURBATION_TYPE_LABELS[ptype]
                if avg_degradation < 0.15:
                    recommendations.append(
                        f"✅ Excellent robustness to {label} perturbations."
                    )
                elif avg_degradation < 0.30:
                    recommendations.append(
                        f"✓ Moderate robustness to {label}. Consider adversarial training."
                    )
                else:
                    recommendations.append(
                        f"⚠️ High vulnerability to {label} perturbations. "
                        "Robustness improvement needed."
                    )
        
        # Analyze cross-domain performance
        if 'cross_domain' in self.evaluation_results:
            cross_domain_results = self.evaluation_results['cross_domain']
            
            cross_f1_scores = []
            for (source, target), results in cross_domain_results.items():
                if source != target:
                    cross_f1_scores.append(results['metrics']['f1'])
            
            if cross_f1_scores:
                avg_cross_f1 = np.mean(cross_f1_scores)
                
                if avg_cross_f1 >= 0.75:
                    recommendations.append("✅ Good cross-domain generalization.")
                elif avg_cross_f1 >= 0.65:
                    recommendations.append("✓ Acceptable cross-domain performance.")
                else:
                    recommendations.append("⚠️ Poor cross-domain generalization. Domain adaptation recommended.")
        
        # Display recommendations
        for rec in recommendations:
            st.markdown(f"- {rec}")
    
    def _generate_text_report(self) -> str:
        """Generate comprehensive text report."""
        lines = [
            "=" * 80,
            "RELIABILITY SUMMARY REPORT",
            "IndoBERT Clickbait Detection System",
            "=" * 80,
            ""
        ]
        
        # In-domain performance
        if 'in_domain' in self.evaluation_results:
            lines.append("IN-DOMAIN PERFORMANCE:")
            lines.append("-" * 80)
            
            for domain, results in self.evaluation_results['in_domain'].items():
                metrics = results['metrics']
                lines.append(f"\n{domain}:")
                lines.append(f"  F1-Score:  {metrics['f1']:.4f}")
                lines.append(f"  Accuracy:  {metrics['accuracy']:.4f}")
                lines.append(f"  Precision: {metrics['precision']:.4f}")
                lines.append(f"  Recall:    {metrics['recall']:.4f}")
            
            lines.append("")
        
        # Robustness summary
        if 'perturbation' in self.evaluation_results:
            lines.append("ROBUSTNESS SUMMARY:")
            lines.append("-" * 80)
            
            for domain, pert_data in self.evaluation_results['perturbation'].items():
                lines.append(f"\n{domain}:")

                # Shared clean baseline first.
                clean_block = pert_data.get('clean', {})
                if clean_block:
                    clean_f1 = clean_block['metrics']['f1']
                    lines.append(f"  {'Clean':8s}: F1 = {clean_f1:.4f}")

                # Then each mechanism with its three levels.
                for ptype in DashboardConfig.PERTURBATION_TYPES:
                    type_block = pert_data.get(ptype, {})
                    if not type_block:
                        continue
                    label = DashboardConfig.PERTURBATION_TYPE_LABELS[ptype]
                    lines.append(f"  [{label}]")
                    for level in DashboardConfig.PERTURBATION_LEVELS:
                        if level in type_block:
                            f1 = type_block[level]['metrics']['f1']
                            lines.append(f"    {level.capitalize():8s}: F1 = {f1:.4f}")

            lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _create_summary_table(self) -> pd.DataFrame:
        """Create summary table for export."""
        rows = []
        
        # In-domain results
        if 'in_domain' in self.evaluation_results:
            for domain, results in self.evaluation_results['in_domain'].items():
                metrics = results['metrics']
                rows.append({
                    'Domain': domain,
                    'Evaluation Type': 'In-Domain',
                    'Perturbation Type': 'None',
                    'Perturbation Level': 'Clean',
                    'F1-Score': metrics['f1'],
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall']
                })

        # Perturbation results — one row per (domain, mechanism, level).
        # The 'Perturbation Type' column keeps semantic and typo distinct.
        if 'perturbation' in self.evaluation_results:
            for domain, pert_data in self.evaluation_results['perturbation'].items():
                for ptype in DashboardConfig.PERTURBATION_TYPES:
                    type_block = pert_data.get(ptype, {})
                    for level in DashboardConfig.PERTURBATION_LEVELS:
                        if level in type_block:
                            metrics = type_block[level]['metrics']
                            rows.append({
                                'Domain': domain,
                                'Evaluation Type': 'Perturbation',
                                'Perturbation Type': ptype.capitalize(),
                                'Perturbation Level': level.capitalize(),
                                'F1-Score': metrics['f1'],
                                'Accuracy': metrics['accuracy'],
                                'Precision': metrics['precision'],
                                'Recall': metrics['recall']
                            })

        return pd.DataFrame(rows)


class DashboardApp:
    """Main dashboard application."""
    
    def __init__(self):
        """Initialize dashboard application."""
        self.config = DashboardConfig()
        self.model_cache = ModelCache()
        
        # Initialize session state
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.current_model = None
            st.session_state.evaluation_results = None
    
    def run(self):
        """Run the dashboard application."""
        # Page configuration
        st.set_page_config(
            page_title="IndoBERT Clickbait Detection Dashboard",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Title
        st.title("🔍 IndoBERT Clickbait Detection Dashboard")
        st.markdown("Comprehensive robustness evaluation system for Indonesian clickbait detection")
        
        # Sidebar
        self._render_sidebar()
        
        # Main content
        if st.session_state.evaluation_results is None:
            st.info("👈 Please load evaluation results from the sidebar to begin.")
            return
        
        # Navigation
        tab1, tab2, tab3, tab4 = st.tabs([
            "📝 Single-Text Prediction",
            "🔄 Domain Shift Matrix",
            "📊 Robustness Analysis",
            "📋 Reliability Summary"
        ])
        
        with tab1:
            if st.session_state.current_model is not None:
                predictor = SingleTextPredictor(
                    st.session_state.current_model,
                    PerturbationEngine()
                )
                predictor.render()
            else:
                st.warning("Please load a model from the sidebar.")
        
        with tab2:
            matrix_viz = DomainShiftMatrix(st.session_state.evaluation_results)
            matrix_viz.render()
        
        with tab3:
            robustness_viz = RobustnessAnalyzer(st.session_state.evaluation_results)
            robustness_viz.render()
        
        with tab4:
            summary = ReliabilitySummary(st.session_state.evaluation_results)
            summary.render()
    
    def _render_sidebar(self):
        """Render sidebar with configuration options."""
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Model selection
            st.subheader("Model Selection")
            model_variant = st.selectbox(
                "Select model variant:",
                options=self.config.MODEL_VARIANTS,
                index=0
            )
            
            model_path = st.text_input(
                "Model checkpoint path:",
                value=f"checkpoints/{model_variant}/best_model.pt"
            )
            
            if st.button("Load Model"):
                try:
                    with st.spinner("Loading model..."):
                        st.session_state.current_model = self.model_cache.load_model(
                            model_path,
                            model_variant
                        )
                    st.success("Model loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
            
            st.divider()
            
            # Evaluation results
            st.subheader("Evaluation Results")
            results_path = st.text_input(
                "Results JSON path:",
                value="evaluation_results/evaluation_results.json"
            )
            
            if st.button("Load Results"):
                try:
                    with st.spinner("Loading evaluation results..."):
                        st.session_state.evaluation_results = self.model_cache.load_evaluation_results(
                            results_path
                        )
                    st.success("Results loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading results: {str(e)}")
            
            st.divider()
            
            # About
            st.subheader("About")
            st.markdown("""
            **IndoBERT Clickbait Detection**
            
            This dashboard provides comprehensive evaluation of clickbait detection models:
            - Single-text prediction with perturbation testing
            - Cross-domain performance analysis
            - Robustness evaluation
            - Statistical insights
            
            Developed for Indonesian language clickbait detection research.
            """)


def main():
    """Main entry point for the dashboard application."""
    app = DashboardApp()
    app.run()


if __name__ == "__main__":
    main()
