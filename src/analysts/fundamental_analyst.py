from analysts.base_analyst import BaseAnalyst

class FundamentalAnalyst(BaseAnalyst):
    def analyze(self, data):
        return {"valuation": "undervalued", "pe_ratio": 12.5}
