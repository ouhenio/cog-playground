from cog import BasePredictor, Input, ConcatenateIterator
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
        ) -> ConcatenateIterator[str]:
        input = self.tokenizer([prompt], return_tensors="pt")
        streamer = TextStreamer(self.tokenizer)

        for token in self.model.generate(
            **input,
            streamer=streamer,
            max_new_tokens=max_length
        ):
            yield token
