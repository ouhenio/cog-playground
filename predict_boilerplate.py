from typing import Any
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = None

    # Define the arguments and types the model takes as input
    def predict(self, image: Path = Input(description="Image to classify")) -> Any:
        """Run a single prediction on the model"""
        # Run the prediction
        preds = self.model.predict(image)
        # Return the top 3 predictions
        return preds