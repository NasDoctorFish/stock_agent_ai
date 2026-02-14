from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging


@dataclass
class AnalystResult:
    name: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "success": self.success,
            "data": self.data,
            "summary": self.summary,
            "warnings": self.warnings,
            "errors": self.errors,
            "artifacts": self.artifacts,
        }


class BaseAnalyst(ABC):
    name: str = "base"

    def __init__(self, db=None, logger=None):
        self.db = db
        if logger is None:
            logger = logging.getLogger(self.__class__.__name__)
            if not logger.handlers:
                logger.addHandler(logging.NullHandler())
        self.logger = logger
        self.last_result: Optional[AnalystResult] = None

    @abstractmethod
    def analyze(self) -> AnalystResult:
        """Run analysis and return a structured AnalystResult."""
        raise NotImplementedError

    def run(self, force: bool = False) -> AnalystResult:
        if not force and self.last_result is not None:
            return self.last_result
        result = self.analyze()
        self.last_result = result
        return result

    def _result(
        self,
        success: bool,
        data: Optional[Dict[str, Any]] = None,
        summary: str = "",
        warnings: Optional[List[str]] = None,
        errors: Optional[List[str]] = None,
        artifacts: Optional[Dict[str, Any]] = None,
    ) -> AnalystResult:
        return AnalystResult(
            name=self.name,
            success=success,
            data=data or {},
            summary=summary,
            warnings=warnings or [],
            errors=errors or [],
            artifacts=artifacts or {},
        )
