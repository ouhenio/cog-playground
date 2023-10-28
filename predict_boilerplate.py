from typing import Any
from cog import BasePredictor, Input

class Predictor(BasePredictor):
    def setup(self):
        self.model = None

    def predict(self, x: str = Input(description="Model input")) -> Any:
        self.model.predict(x)
        return self.model.predict(x)