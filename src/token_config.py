class TokenConfig:
    def __init__(self):
        algo_token_start = 32001
        abr_algos = ['genet', 'udr_1', 'udr_2', 'udr_3', 'udr_real', 'mpc', 'bba','mixed']
        self.abr_algorithm_vocab = {algo: algo_token_start + i for i, algo in enumerate(abr_algos)}

    def get_abr_algorithm_vocab(self):
        return self.abr_algorithm_vocab