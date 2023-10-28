from typing import Any
from cog import BasePredictor
from transformers import pipeline

class Predictor(BasePredictor):
    def setup(self):
        self.model = pipeline('text-generation', model='gpt2')

    def predict(self, prompt: str) -> Any:
        output = self.model(prompt, max_length=60)
        return output