"""
DecisionMaker
-------------
Integrates multiple analysts and produces actionable investment decisions.
"""
class DecisionMaker:
    def __init__(self, analysts: list):
        self.analysts = analysts

    def make_decision(self, market_state):
        results = {a.__class__.__name__: a.analyze(market_state) for a in self.analysts}
        # Basic rule-based example
        sentiment = results.get("NewsMediaAnalyst", {}).get("sentiment", 0)
        macro = results.get("MacroAnalyst", {}).get("macro_trend", "neutral")

        if macro == "bullish" and sentiment > 0.5:
            return "BUY"
        elif macro == "bearish" and sentiment < -0.5:
            return "SELL"
        else:
            return "HOLD"
    def run(self):
        print("Just ran")
