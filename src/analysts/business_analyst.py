from analysts.base_analyst import BaseAnalyst

class BusinessAnalyst(BaseAnalyst):
    def analyze(self, data):
        return {"industry_outlook": "positive", "competition": "moderate"}
