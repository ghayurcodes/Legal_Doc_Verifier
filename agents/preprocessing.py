import torchvision.transforms as T
from PIL import Image


# ─────────────────────────────────────────────────────────────
# PREPROCESSING AGENT
# This agent cleans and prepares raw inputs before they are
# passed to the Siamese CNN (image) or RoBERTa (text).
#
# Think of it as the "front door" of the pipeline:
# Nothing enters the models without going through here first.
# ─────────────────────────────────────────────────────────────

class PreprocessingAgent:

    def __init__(self):
        # The same transforms used during training — MUST match exactly
        # If transforms differ between training and inference, model performs badly
        self.sig_transform = T.Compose([
            T.Resize((128, 256)),               # resize to fixed size
            T.Grayscale(num_output_channels=3), # convert to grayscale but keep 3 channels
            T.ToTensor(),                       # pixel values → tensor (0.0 to 1.0)
            T.Normalize(mean=[0.5, 0.5, 0.5],  # shift values to (-1.0 to 1.0)
                        std=[0.5, 0.5, 0.5])
        ])

    def prepare_signature(self, image_path):
        """
        Takes a file path to a signature image.
        Returns a tensor of shape (1, 3, 128, 256) ready for the Siamese model.
        The extra '1' at the front is the batch dimension (model expects batches).
        """
        img    = Image.open(image_path).convert('RGB')
        tensor = self.sig_transform(img).unsqueeze(0)  # add batch dimension
        return tensor

    def prepare_text(self, text):
        """
        Takes a raw string of document text.
        Returns a cleaned string ready for the RoBERTa tokenizer.
        Simple cleaning: strip whitespace, remove extra spaces.
        """
        text = text.strip()                # remove leading/trailing spaces
        text = ' '.join(text.split())      # collapse multiple spaces into one
        return text
