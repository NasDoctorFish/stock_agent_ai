"""
Investor
--------
Top-level orchestrator that uses the DecisionMaker and Analysts to act.
"""
from core.status_logger import StatusLogger

class Investor:
    def __init__(self, decision_maker, logger: StatusLogger):
        self.brain = decision_maker
        self.logger = logger

    def act(self, market_state):
        decision = self.brain.make_decision(market_state)
        self.logger.info(f"Decision: {decision}")
        return decision
