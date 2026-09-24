"""
Statistical Analyzer Module for IndoBERT Clickbait Detection System

This module implements Bayesian statistical tests, ROPE analysis, significance testing,
comparative statistics, and result interpretation for robustness evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu
import logging
import json
import os

from debug_logger import (
    dbg_stats_input,
    dbg_stats_bayesian,
    dbg_stats_rope,
    dbg_stats_bootstrap,
    dbg_stats_significance,
    dbg_stats_robustness_input,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BayesianTester:
    """
    Implements Bayesian statistical tests for model comparison.
    """
    
    def __init__(self, rope_threshold: float = 0.01, random_seed: int = 42):
        """
        Initialize Bayesian tester.
        
        Args:
            rope_threshold: Region of Practical Equivalence threshold (default: 0.01 = 1%)
            random_seed: Random seed for reproducibility
        """
        self.rope_threshold = rope_threshold
        self.random_seed = random_seed
        # Use a per-instance RNG instead of mutating the global numpy seed.
        self._rng = np.random.default_rng(random_seed)

        logger.info(f"BayesianTester initialized with ROPE threshold={rope_threshold}")
    
    def bayesian_signed_rank_test( #MARK
        self,
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        rope: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Perform Bayesian signed-rank test for paired comparisons.
        
        This test is used to compare two models on the same test set,
        determining if one significantly outperforms the other.
        
        Args:
            scores_a: Performance scores from model A
            scores_b: Performance scores from model B
            rope: ROPE threshold (uses default if None)
            
        Returns:
            Dictionary with test results and interpretation
        """
        if rope is None:
            rope = self.rope_threshold
        
        logger.info("Performing Bayesian signed-rank test")
        
        # Calculate differences
        differences = scores_a - scores_b
        
        # Wilcoxon signed-rank test
        if len(differences) < 10:
            logger.warning(f"Small sample size ({len(differences)}). Results may be unreliable.")
        
        # Remove zeros (ties)
        non_zero_diffs = differences[differences != 0]
        
        if len(non_zero_diffs) == 0:
            return {
                'test_type': 'bayesian_signed_rank',
                'n_samples': len(differences),
                'mean_difference': 0.0,
                'median_difference': 0.0,
                'std_difference': 0.0,
                'statistic': None,
                'p_value': 1.0,
                'rope_threshold': rope,
                'interpretation': 'equivalent',
                'conclusion': 'Models are equivalent (no differences detected)'
            }
        
        # ── debug: raw score arrays going into the test ────────────────────
        dbg_stats_input(
            context="bayesian_signed_rank",
            scores_a=scores_a,
            scores_b=scores_b,
        )

        # Perform Wilcoxon test
        statistic, p_value = wilcoxon(non_zero_diffs, alternative='two-sided')
        
        # Calculate effect size (r = Z / sqrt(N))
        z_score = stats.norm.ppf(1 - p_value / 2)
        effect_size = abs(z_score) / np.sqrt(len(non_zero_diffs))
        
        # Bayesian interpretation using ROPE
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)
        
        # Calculate credible interval (95%)
        ci_lower = mean_diff - 1.96 * (std_diff / np.sqrt(len(differences)))
        ci_upper = mean_diff + 1.96 * (std_diff / np.sqrt(len(differences)))
        
        # ROPE decision
        if ci_lower > rope:
            interpretation = 'model_a_better'
            conclusion = f'Model A significantly outperforms Model B (mean diff: {mean_diff:.4f})'
        elif ci_upper < -rope:
            interpretation = 'model_b_better'
            conclusion = f'Model B significantly outperforms Model A (mean diff: {mean_diff:.4f})'
        elif ci_lower > -rope and ci_upper < rope:
            interpretation = 'equivalent'
            conclusion = f'Models are practically equivalent (within ROPE: ±{rope})'
        else:
            interpretation = 'inconclusive'
            conclusion = 'Results are inconclusive (credible interval overlaps ROPE)'
        
        results = {
            'test_type': 'bayesian_signed_rank',
            'n_samples': len(differences),
            'mean_difference': float(mean_diff),
            'median_difference': float(np.median(differences)),
            'std_difference': float(std_diff),
            'statistic': float(statistic),
            'p_value': float(p_value),
            'effect_size': float(effect_size),
            'credible_interval_95': {
                'lower': float(ci_lower),
                'upper': float(ci_upper)
            },
            'rope_threshold': rope,
            'interpretation': interpretation,
            'conclusion': conclusion
        }
        
        logger.info(f"Test complete: {interpretation}")

        # ── debug: full test results ───────────────────────────────────────
        dbg_stats_bayesian(results)
        
        return results
    
class ROPEAnalyzer:
    """
    Implements Region of Practical Equivalence (ROPE) analysis.
    """
    
    def __init__(self, rope_threshold: float = 0.01):
        """
        Initialize ROPE analyzer.
        
        Args:
            rope_threshold: ROPE threshold (default: 0.01 = 1%)
        """
        self.rope_threshold = rope_threshold
        logger.info(f"ROPEAnalyzer initialized with threshold={rope_threshold}")
    
    def analyze_rope(
        self,
        differences: np.ndarray,
        rope: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Analyze differences using ROPE framework.
        
        Args:
            differences: Array of performance differences
            rope: ROPE threshold (uses default if None)
            
        Returns:
            Dictionary with ROPE analysis results
        """
        if rope is None:
            rope = self.rope_threshold
        
        logger.info("Performing ROPE analysis")

        # ── debug: input differences ───────────────────────────────────────
        dbg_stats_input(
            context="rope_analysis",
            scores_a=differences,
            scores_b=np.zeros_like(differences),
            label_a="differences",
            label_b="zero_baseline",
        )
        
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)
        n = len(differences)
        
        # Calculate credible intervals at different levels
        credible_intervals = {}
        for level in [0.90, 0.95, 0.99]:
            z_score = stats.norm.ppf((1 + level) / 2)
            margin = z_score * (std_diff / np.sqrt(n))
            credible_intervals[f'{int(level*100)}%'] = {
                'lower': float(mean_diff - margin),
                'upper': float(mean_diff + margin)
            }
        
        # ROPE decision for 95% CI
        ci_95 = credible_intervals['95%']
        
        if ci_95['lower'] > rope:
            decision = 'reject_equivalence_positive'
            interpretation = 'Significant positive difference (outside ROPE)'
        elif ci_95['upper'] < -rope:
            decision = 'reject_equivalence_negative'
            interpretation = 'Significant negative difference (outside ROPE)'
        elif ci_95['lower'] > -rope and ci_95['upper'] < rope:
            decision = 'accept_equivalence'
            interpretation = 'Practically equivalent (within ROPE)'
        else:
            decision = 'undecided'
            interpretation = 'Inconclusive (CI overlaps ROPE boundaries)'
        
        # Calculate probability of being in ROPE (approximation)
        prob_in_rope = self._calculate_rope_probability(mean_diff, std_diff, n, rope)
        
        results = {
            'mean_difference': float(mean_diff),
            'std_difference': float(std_diff),
            'n_samples': int(n),
            'rope_threshold': rope,
            'rope_bounds': {'lower': -rope, 'upper': rope},
            'credible_intervals': credible_intervals,
            'decision': decision,
            'interpretation': interpretation,
            'probability_in_rope': float(prob_in_rope)
        }
        
        logger.info(f"ROPE analysis complete: {decision}")

        # ── debug: ROPE decision ───────────────────────────────────────────
        dbg_stats_rope(results)
        
        return results
    
    def _calculate_rope_probability(
        self,
        mean: float,
        std: float,
        n: int,
        rope: float
    ) -> float:
        """
        Calculate approximate probability that true difference is within ROPE.
        
        Args:
            mean: Mean difference
            std: Standard deviation
            n: Sample size
            rope: ROPE threshold
            
        Returns:
            Probability (0-1)
        """
        se = std / np.sqrt(n)
        
        if se == 0:
            return 1.0 if abs(mean) <= rope else 0.0

        # Calculate z-scores for ROPE boundaries
        z_lower = (-rope - mean) / se
        z_upper = (rope - mean) / se
        
        # Calculate probability
        prob = stats.norm.cdf(z_upper) - stats.norm.cdf(z_lower)
        
        return max(0.0, min(1.0, prob))
    
class SignificanceTester:
    """
    Implements various significance testing utilities.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize significance tester.
        
        Args:
            alpha: Significance level (default: 0.05)
        """
        self.alpha = alpha
        logger.info(f"SignificanceTester initialized with alpha={alpha}")
    
    def paired_t_test(
        self,
        scores_a: np.ndarray,
        scores_b: np.ndarray
    ) -> Dict[str, Any]:
        """
        Perform paired t-test.
        
        Args:
            scores_a: Scores from model A
            scores_b: Scores from model B
            
        Returns:
            Dictionary with test results
        """
        logger.info("Performing paired t-test")

        # ── debug: raw input scores ────────────────────────────────────────
        dbg_stats_input(
            context="paired_t_test",
            scores_a=scores_a,
            scores_b=scores_b,
        )
        
        statistic, p_value = stats.ttest_rel(scores_a, scores_b)
        
        # Calculate effect size (Cohen's d)
        differences = scores_a - scores_b
        cohens_d = np.mean(differences) / np.std(differences, ddof=1)
        
        is_significant = p_value < self.alpha
        
        results = {
            'test_type': 'paired_t_test',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'alpha': self.alpha,
            'is_significant': bool(is_significant),
            'cohens_d': float(cohens_d),
            'mean_difference': float(np.mean(differences)),
            'interpretation': 'significant' if is_significant else 'not_significant'
        }
        
        logger.info(f"Paired t-test: p={p_value:.4f}, significant={is_significant}")

        # ── debug: test output ─────────────────────────────────────────────
        dbg_stats_significance("paired_t_test", results)
        
        return results
    
    def mann_whitney_u_test(
        self,
        scores_a: np.ndarray,
        scores_b: np.ndarray
    ) -> Dict[str, Any]:
        """
        Perform Mann-Whitney U test (non-parametric).
        
        Args:
            scores_a: Scores from model A
            scores_b: Scores from model B
            
        Returns:
            Dictionary with test results
        """
        logger.info("Performing Mann-Whitney U test")

        # ── debug: raw input scores ────────────────────────────────────────
        dbg_stats_input(
            context="mann_whitney_u",
            scores_a=scores_a,
            scores_b=scores_b,
        )
        
        statistic, p_value = mannwhitneyu(
            scores_a,
            scores_b,
            alternative='two-sided'
        )
        
        is_significant = p_value < self.alpha
        
        # Calculate effect size (rank-biserial correlation)
        n_a = len(scores_a)
        n_b = len(scores_b)
        r = 1 - (2 * statistic) / (n_a * n_b)
        
        results = {
            'test_type': 'mann_whitney_u',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'alpha': self.alpha,
            'is_significant': bool(is_significant),
            'effect_size_r': float(r),
            'interpretation': 'significant' if is_significant else 'not_significant'
        }
        
        logger.info(f"Mann-Whitney U test: p={p_value:.4f}, significant={is_significant}")

        # ── debug: test output ─────────────────────────────────────────────
        dbg_stats_significance("mann_whitney_u", results)
        
        return results
    
    def bootstrap_confidence_interval(
        self,
        data: np.ndarray,
        statistic_func: callable = np.mean,
        n_bootstrap: int = 10000,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Calculate bootstrap confidence interval.
        
        Args:
            data: Data array
            statistic_func: Function to calculate statistic (default: mean)
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (default: 0.95)
            
        Returns:
            Dictionary with bootstrap results
        """
        logger.info(f"Calculating bootstrap CI with {n_bootstrap} samples")

        # ── debug: input data distribution ────────────────────────────────
        dbg_stats_input(
            context="bootstrap_ci",
            scores_a=data,
            scores_b=np.zeros_like(data),
            label_a="data",
            label_b="zero_baseline",
        )
        
        bootstrap_statistics = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_statistics.append(statistic_func(sample))
        
        bootstrap_statistics = np.array(bootstrap_statistics)
        
        # Calculate percentile confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_statistics, lower_percentile)
        ci_upper = np.percentile(bootstrap_statistics, upper_percentile)
        
        observed_statistic = statistic_func(data)
        
        results = {
            'observed_statistic': float(observed_statistic),
            'confidence_level': confidence_level,
            'confidence_interval': {
                'lower': float(ci_lower),
                'upper': float(ci_upper)
            },
            'bootstrap_mean': float(np.mean(bootstrap_statistics)),
            'bootstrap_std': float(np.std(bootstrap_statistics)),
            'n_bootstrap': n_bootstrap
        }
        
        logger.info(f"Bootstrap CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

        # ── debug: bootstrap output ────────────────────────────────────────
        dbg_stats_bootstrap(results)
        
        return results


class ComparativeStatistics:
    """
    Implements comparative statistics for model evaluation.
    """
    
    def __init__(self):
        """Initialize comparative statistics analyzer."""
        logger.info("ComparativeStatistics initialized")
    
    def calculate_degradation_statistics(
        self,
        clean_scores: np.ndarray,
        perturbed_scores: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate degradation statistics.
        
        Args:
            clean_scores: Scores on clean data
            perturbed_scores: Scores on perturbed data
            
        Returns:
            Dictionary with degradation statistics
        """
        degradation = clean_scores - perturbed_scores
        
        stats_dict = {
            'mean_degradation': float(np.mean(degradation)),
            'median_degradation': float(np.median(degradation)),
            'std_degradation': float(np.std(degradation)),
            'min_degradation': float(np.min(degradation)),
            'max_degradation': float(np.max(degradation)),
            'mean_relative_degradation': float(np.mean(degradation / clean_scores) * 100),
            'samples_degraded': int(np.sum(degradation > 0)),
            'samples_improved': int(np.sum(degradation < 0)),
            'samples_unchanged': int(np.sum(degradation == 0))
        }
        
        return stats_dict
    
    def rank_models_by_robustness(
        self,
        model_degradations: Dict[str, np.ndarray]
    ) -> List[Tuple[str, float]]:
        """
        Rank models by robustness (lower degradation is better).
        
        Args:
            model_degradations: Dictionary mapping model names to degradation arrays
            
        Returns:
            List of (model_name, mean_degradation) tuples, sorted by robustness
        """
        logger.info(f"Ranking {len(model_degradations)} models by robustness")
        
        rankings = []
        
        for name, degradations in model_degradations.items():
            mean_deg = np.mean(degradations)
            rankings.append((name, float(mean_deg)))
        
        # Sort by degradation (ascending - lower is better)
        rankings.sort(key=lambda x: x[1])
        
        logger.info(f"Most robust model: {rankings[0][0]} (degradation: {rankings[0][1]:.4f})")
        
        return rankings
    
    def calculate_domain_shift_statistics(
        self,
        in_domain_scores: np.ndarray,
        cross_domain_scores: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate domain shift statistics.
        
        Args:
            in_domain_scores: Scores on in-domain data
            cross_domain_scores: Scores on cross-domain data
            
        Returns:
            Dictionary with domain shift statistics
        """
        shift = in_domain_scores - cross_domain_scores
        
        stats_dict = {
            'mean_shift': float(np.mean(shift)),
            'median_shift': float(np.median(shift)),
            'std_shift': float(np.std(shift)),
            'mean_relative_shift': float(np.mean(shift / in_domain_scores) * 100),
            'in_domain_mean': float(np.mean(in_domain_scores)),
            'cross_domain_mean': float(np.mean(cross_domain_scores))
        }
        
        return stats_dict


class StatisticalAnalyzer:
    """
    Main statistical analyzer orchestrating all statistical tests and analyses.
    """
    
    def __init__(
        self,
        rope_threshold: float = 0.01,
        alpha: float = 0.05,
        random_seed: int = 42,
        output_dir: str = "statistical_analysis"
    ):
        """
        Initialize statistical analyzer.
        
        Args:
            rope_threshold: ROPE threshold (default: 0.01 = 1%)
            alpha: Significance level (default: 0.05)
            random_seed: Random seed for reproducibility
            output_dir: Directory to save analysis results
        """
        self.rope_threshold = rope_threshold
        self.alpha = alpha
        self.random_seed = random_seed
        self.output_dir = output_dir
        
        # Initialize sub-analyzers
        self.bayesian_tester = BayesianTester(rope_threshold, random_seed)
        self.rope_analyzer = ROPEAnalyzer(rope_threshold)
        self.significance_tester = SignificanceTester(alpha)
        self.comparative_stats = ComparativeStatistics()
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"StatisticalAnalyzer initialized. Results will be saved to {output_dir}")
    
    def analyze_model_comparison(
        self,
        model_a_scores: np.ndarray,
        model_b_scores: np.ndarray,
        model_a_name: str = "Model A",
        model_b_name: str = "Model B"
    ) -> Dict[str, Any]:
        """
        Comprehensive comparison of two models.
        
        Args:
            model_a_scores: Scores from model A
            model_b_scores: Scores from model B
            model_a_name: Name of model A
            model_b_name: Name of model B
            
        Returns:
            Dictionary with comprehensive comparison results
        """
        logger.info(f"Analyzing comparison: {model_a_name} vs {model_b_name}")
        
        results = {
            'model_a_name': model_a_name,
            'model_b_name': model_b_name,
            'bayesian_test': self.bayesian_tester.bayesian_signed_rank_test(
                model_a_scores,
                model_b_scores
            ),
            'rope_analysis': self.rope_analyzer.analyze_rope(
                model_a_scores - model_b_scores
            ),
            'paired_t_test': self.significance_tester.paired_t_test(
                model_a_scores,
                model_b_scores
            ),
            'mann_whitney_test': self.significance_tester.mann_whitney_u_test(
                model_a_scores,
                model_b_scores
            )
        }
        
        # Generate interpretation
        results['overall_interpretation'] = self._interpret_comparison(results)
        
        return results
    
    def analyze_robustness(
        self,
        clean_scores: np.ndarray,
        perturbed_scores_dict: Dict[str, np.ndarray],
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Analyze model robustness across perturbation levels.
        
        Args:
            clean_scores: Scores on clean data
            perturbed_scores_dict: Dictionary mapping perturbation levels to scores
            model_name: Name of the model
            
        Returns:
            Dictionary with robustness analysis results
        """
        logger.info(f"Analyzing robustness for {model_name}")

        # ── debug: input score distributions ──────────────────────────────
        dbg_stats_robustness_input(
            model_name=model_name,
            clean_scores=clean_scores,
            perturbed_dict=perturbed_scores_dict,
        )
        
        results = {
            'model_name': model_name,
            'perturbation_levels': list(perturbed_scores_dict.keys()),
            'degradation_analysis': {}
        }
        
        for level, perturbed_scores in perturbed_scores_dict.items():
            logger.info(f"Analyzing {level} perturbation level")
            
            # Degradation statistics
            deg_stats = self.comparative_stats.calculate_degradation_statistics(
                clean_scores,
                perturbed_scores
            )
            
            # Statistical tests
            bayesian_test = self.bayesian_tester.bayesian_signed_rank_test(
                clean_scores,
                perturbed_scores
            )
            
            rope_analysis = self.rope_analyzer.analyze_rope(
                clean_scores - perturbed_scores
            )
            
            results['degradation_analysis'][level] = {
                'statistics': deg_stats,
                'bayesian_test': bayesian_test,
                'rope_analysis': rope_analysis
            }
        
        # Overall robustness ranking
        degradations = {
            level: clean_scores - perturbed_scores_dict[level]
            for level in perturbed_scores_dict.keys()
        }
        
        results['robustness_ranking'] = self.comparative_stats.rank_models_by_robustness(
            degradations
        )
        
        return results
    
    def analyze_cross_domain_robustness(
        self,
        in_domain_scores: np.ndarray,
        cross_domain_scores_dict: Dict[str, np.ndarray],
        source_domain: str
    ) -> Dict[str, Any]:
        """
        Analyze cross-domain robustness.
        
        Args:
            in_domain_scores: Scores on in-domain data
            cross_domain_scores_dict: Dictionary mapping target domains to scores
            source_domain: Source domain name
            
        Returns:
            Dictionary with cross-domain analysis results
        """
        logger.info(f"Analyzing cross-domain robustness for {source_domain}")
        
        results = {
            'source_domain': source_domain,
            'target_domains': list(cross_domain_scores_dict.keys()),
            'domain_shift_analysis': {}
        }
        
        for target_domain, target_scores in cross_domain_scores_dict.items():
            logger.info(f"Analyzing shift to {target_domain}")
            
            # Domain shift statistics
            shift_stats = self.comparative_stats.calculate_domain_shift_statistics(
                in_domain_scores,
                target_scores
            )
            
            # Statistical tests
            bayesian_test = self.bayesian_tester.bayesian_signed_rank_test(
                in_domain_scores,
                target_scores
            )
            
            results['domain_shift_analysis'][target_domain] = {
                'statistics': shift_stats,
                'bayesian_test': bayesian_test
            }
        
        return results
    
    def _interpret_comparison(self, results: Dict[str, Any]) -> str:
        """
        Generate overall interpretation from comparison results.
        
        Args:
            results: Comparison results dictionary
            
        Returns:
            Interpretation string
        """
        bayesian = results['bayesian_test']['interpretation']
        rope = results['rope_analysis']['decision']
        
        if bayesian == 'model_a_better' and 'positive' in rope:
            return f"{results['model_a_name']} significantly outperforms {results['model_b_name']}"
        elif bayesian == 'model_b_better' and 'negative' in rope:
            return f"{results['model_b_name']} significantly outperforms {results['model_a_name']}"
        elif bayesian == 'equivalent' or 'equivalence' in rope:
            return f"{results['model_a_name']} and {results['model_b_name']} are practically equivalent"
        else:
            return "Results are inconclusive - further investigation needed"
    
    def generate_analysis_report(
        self,
        results: Dict[str, Any],
        report_type: str = "comparison"
    ) -> str:
        """
        Generate human-readable analysis report.
        
        Args:
            results: Analysis results dictionary
            report_type: Type of report ('comparison', 'robustness', 'cross_domain')
            
        Returns:
            Report as string
        """
        report_lines = [
            "=" * 80,
            f"STATISTICAL ANALYSIS REPORT - {report_type.upper()}",
            "=" * 80,
            ""
        ]
        
        if report_type == "comparison":
            report_lines.extend(self._format_comparison_report(results))
        elif report_type == "robustness":
            report_lines.extend(self._format_robustness_report(results))
        elif report_type == "cross_domain":
            report_lines.extend(self._format_cross_domain_report(results))
        
        report_lines.append("=" * 80)
        
        report = "\n".join(report_lines)
        
        # Save report
        report_path = os.path.join(
            self.output_dir,
            f"{report_type}_analysis_report.txt"
        )
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to {report_path}")
        
        return report
    
    def _format_comparison_report(self, results: Dict[str, Any]) -> List[str]:
        """Format comparison report."""
        lines = [
            f"Model A: {results['model_a_name']}",
            f"Model B: {results['model_b_name']}",
            "",
            "BAYESIAN SIGNED-RANK TEST:",
            f"  Mean Difference: {results['bayesian_test']['mean_difference']:.4f}",
            f"  P-value: {results['bayesian_test']['p_value']:.4f}",
            f"  Interpretation: {results['bayesian_test']['interpretation']}",
            f"  Conclusion: {results['bayesian_test']['conclusion']}",
            "",
            "ROPE ANALYSIS:",
            f"  ROPE Threshold: ±{results['rope_analysis']['rope_threshold']}",
            f"  Decision: {results['rope_analysis']['decision']}",
            f"  Interpretation: {results['rope_analysis']['interpretation']}",
            "",
            "OVERALL INTERPRETATION:",
            f"  {results['overall_interpretation']}",
            ""
        ]
        return lines
    
    def _format_robustness_report(self, results: Dict[str, Any]) -> List[str]:
        """Format robustness report."""
        lines = [
            f"Model: {results['model_name']}",
            f"Perturbation Levels: {', '.join(results['perturbation_levels'])}",
            ""
        ]
        
        for level, analysis in results['degradation_analysis'].items():
            stats = analysis['statistics']
            lines.extend([
                f"{level.upper()} PERTURBATION:",
                f"  Mean Degradation: {stats['mean_degradation']:.4f}",
                f"  Relative Degradation: {stats['mean_relative_degradation']:.2f}%",
                f"  Samples Degraded: {stats['samples_degraded']}",
                f"  Interpretation: {analysis['bayesian_test']['interpretation']}",
                ""
            ])
        
        return lines
    
    def _format_cross_domain_report(self, results: Dict[str, Any]) -> List[str]:
        """Format cross-domain report."""
        lines = [
            f"Source Domain: {results['source_domain']}",
            f"Target Domains: {', '.join(results['target_domains'])}",
            ""
        ]
        
        for target, analysis in results['domain_shift_analysis'].items():
            stats = analysis['statistics']
            lines.extend([
                f"SHIFT TO {target.upper()}:",
                f"  Mean Shift: {stats['mean_shift']:.4f}",
                f"  Relative Shift: {stats['mean_relative_shift']:.2f}%",
                f"  In-domain Mean: {stats['in_domain_mean']:.4f}",
                f"  Cross-domain Mean: {stats['cross_domain_mean']:.4f}",
                ""
            ])
        
        return lines
