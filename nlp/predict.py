from typing import Any
from cog import BasePredictor, Input
from transformers import pipeline

class Predictor(BasePredictor):
    def setup(self):
        # This is inefficient, weights are downloaded each time
        self.model = pipeline('text-generation', model='gpt2')

    def predict(
            self,
            prompt: str = Input(description="GPT2 prompt"),
            max_length: int = Input(
                description="Max tokens to generate",
                default=60
            )
        ) -> Any:
        output = self.model(prompt, max_length=max_length)
        return output