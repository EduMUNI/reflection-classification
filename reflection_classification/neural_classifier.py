from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from .utils.dataset import LABELS


class NeuralClassifier:

    def __init__(self, model_path: str, uses_context: bool, device: str):
        self.config = AutoConfig.from_pretrained(model_path)
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, config=self.config).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.uses_context = uses_context

    def predict_sentence(self, sentence: str, context: str = None):
        if context is None and self.uses_context:
            raise ValueError("You need to pass in context argument, including the sentence")

        features = self.tokenizer(sentence, text_pair=context,
                                  padding="max_length", truncation=True, return_tensors='pt')
        outputs = self.model(**features.to(self.device), return_dict=True)
        argmax = outputs.logits.argmax(dim=-1).detach().cpu().tolist()[0]
        labels = LABELS[argmax]

        return labels
