"""Expose the task-generation package from a source checkout without installation."""
from pathlib import Path
__path__ = [str(Path(__file__).resolve().parents[1] / 'v5/tada_taskgen')]
__version__ = '5.0.0'
