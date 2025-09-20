"""
Utilities package for TVC system

Provides configuration, metrics, visualization, and helpers.

This __init__ avoids eager submodule imports to prevent circular imports.
It exposes a lazy attribute loader so consumers can continue using
`from src.utils import create_config_object`, etc.
"""

from importlib import import_module
from typing import Any

__all__ = [
    # Configuration
    'ExperimentConfig',
    'ConfigObject',
    'create_config_object',
    'create_default_config',
    'save_config',
    'load_config',
    'create_experiment_configs',

    # Metrics
    'EpisodeMetrics',
    'ExperimentMetrics',
    'MetricsCollector',
    'MetricsAnalyzer',

    # Visualization
    'TVCVisualizer',
]

_EXPORT_MAP = {
    # config exports
    'ExperimentConfig': ('src.utils.config', 'ExperimentConfig'),
    'ConfigObject': ('src.utils.config', 'ConfigObject'),
    'create_config_object': ('src.utils.config', 'create_config_object'),
    'create_default_config': ('src.utils.config', 'create_default_config'),
    'save_config': ('src.utils.config', 'save_config'),
    'load_config': ('src.utils.config', 'load_config'),
    'create_experiment_configs': ('src.utils.config', 'create_experiment_configs'),

    # metrics exports
    'EpisodeMetrics': ('src.utils.metrics', 'EpisodeMetrics'),
    'ExperimentMetrics': ('src.utils.metrics', 'ExperimentMetrics'),
    'MetricsCollector': ('src.utils.metrics', 'MetricsCollector'),
    'MetricsAnalyzer': ('src.utils.metrics', 'MetricsAnalyzer'),

    # visualization exports
    'TVCVisualizer': ('src.utils.visualization', 'TVCVisualizer'),
}


def __getattr__(name: str) -> Any:
    target = _EXPORT_MAP.get(name)
    if target is None:
        raise AttributeError(f"module 'src.utils' has no attribute '{name}'")
    module_name, attr_name = target
    module = import_module(module_name)
    return getattr(module, attr_name)


def __dir__():
    return sorted(list(globals().keys()) + __all__)