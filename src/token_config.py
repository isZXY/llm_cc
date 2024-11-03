import torch

class TokenConfig:
    def __init__(self):
        algo_token_start = 32001
        self.abr_algos = ['genet', 'udr_1', 'udr_2', 'udr_3', 'udr_real', 'mpc', 'bba','mixed']
        self.abr_algorithm_vocab = {algo: algo_token_start + i for i, algo in enumerate(self.abr_algos)}

    def get_abr_algorithm_vocab(self):
        return self.abr_algorithm_vocab
    

    def get_custom_token_indices(self):
        return list(self.abr_algorithm_vocab.values())
    
    def id_to_token_name(self,selected_token_id):
        return self.abr_algos[selected_token_id]