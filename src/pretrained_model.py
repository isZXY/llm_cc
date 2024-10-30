from transformers import AutoTokenizer, LlamaTokenizer
from transformers import LlamaModel
from token_config import TokenConfig

class PretrainedLanguageModel:
    def __init__(self, model_name, model_path, device):
        self.device = device
        if model_name == "llama":
            # Load Pretrained Model
            self.llm_model = LlamaModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
            )

            # Load LlamaTokenizer
            self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.llm_model.resize_token_embeddings(len(self.tokenizer))
            
            tokenconfig = TokenConfig()
            new_tokens = list(tokenconfig.get_abr_algorithm_vocab().keys())

            token_ids = self.tokenizer.add_tokens(new_tokens)
            print("Added token IDs:", token_ids)

            self.llm_model.resize_token_embeddings(len(self.tokenizer))
            self.llm_model.to(self.device)

        else:
            print("Unsupported model. More pretrained model support is ongoing...")

    def get_model(self):
        return self.tokenizer, self.llm_model
