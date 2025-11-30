from analysts.base_analyst import BaseAnalyst

class StockDataAnalyst(BaseAnalyst):
    def analyze(self, data):
        return {"momentum": 0.65, "volatility": 0.22}
