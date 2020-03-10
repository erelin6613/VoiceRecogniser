from transformers import BertTokenizer, BertModel
from torch import Tensor


class NaturalLanguageModel:

	def __init__(self, base_model):
		self.base_model = base_model

	def tokenize_text(self, text):

		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		return tokenizer.tokenize(text)

	def get_inputs(self, tokens):
		return tokenizer.convert_tokens_to_ids(tokens)

	def get_bert(self):
		return BertModel.from_pretrained('bert-base-uncased')

