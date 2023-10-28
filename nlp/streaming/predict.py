from typing import Any
from cog import BasePredictor, Input
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

class Predictor(BasePredictor):
    def setup(self):
        # This is inefficient, weights are downloaded each time
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")

    def predict(
            self,
            prompt: str = Input(description="GPT2 prompt"),
            max_length: int = Input(
                description="Max tokens to generate",
                default=60
            )
        ) -> Any:
        input = self.tokenizer([prompt], return_tensors="pt")
        streamer = TextStreamer(self.tokenizer)
        return self.model.generate(
            **input,
            streamer=streamer,
            max_new_tokens=max_length
        )
