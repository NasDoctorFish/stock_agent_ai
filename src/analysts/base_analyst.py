# analysts/base_analyst.py
from abc import ABC, abstractmethod

class BaseAnalyst(ABC):
    def __init__(self, logger=None):
        self.logger = logger

    @abstractmethod
    def analyze(self, *args, **kwargs):
        """Run analysis and return a dict-like signal"""
        pass
