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

            special_tokens = set(self.tokenizer.all_special_ids)
            
            tokenconfig = TokenConfig()
            new_tokens = list(tokenconfig.get_abr_algorithm_vocab().keys())

            token_ids = self.tokenizer.add_tokens(new_tokens)
            self.custom_token_size = len([token for token in new_tokens if self.tokenizer.convert_tokens_to_ids(token) not in special_tokens])   
            print("Added token IDs:", token_ids)

            self.vocab_size = len(self.tokenizer)

            self.llm_model.resize_token_embeddings(len(self.tokenizer))
            self.llm_model.to(self.device)

                        

        else:
            print("Unsupported model. More pretrained model support is ongoing...")

    def get_vocab_size(self):
        return self.vocab_size

    def get_custom_token_size(self):
        return self.custom_token_size

    def get_model(self):
        return self.tokenizer, self.llm_model
