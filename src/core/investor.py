"""
Investor
--------
Top-level orchestrator that aggregates analyst outputs.
"""
from __future__ import annotations

from typing import Dict, List
from analysts.base_analyst import AnalystResult, BaseAnalyst


class Investor:
    def __init__(self, analysts: List[BaseAnalyst], logger=None):
        self.analysts = analysts
        self.logger = logger

    def gather_results(self) -> Dict[str, AnalystResult]:
        results: Dict[str, AnalystResult] = {}
        for analyst in self.analysts:
            try:
                result = analyst.run()
            except Exception as e:
                name = getattr(analyst, "name", analyst.__class__.__name__)
                if self.logger:
                    self.logger.error(f"Analyst {name} failed: {e}")
                result = AnalystResult(name=name, success=False, errors=[str(e)])
            results[result.name] = result
        return results

    def aggregate(self) -> Dict[str, dict]:
        results = self.gather_results()
        aggregated = {name: res.to_dict() for name, res in results.items()}
        return aggregated
