class _DatasetPool:
    '''
    The experience pool used to save & load data.

    Data Structure:
        Pairs of [Prompts, Probed Time Series, Labels].
        Prompts: string, Natural Language that describe the tasks.
        Probed Time Series: ndarray, The Multi-dimension time series probed during startup.
        Labels: string, The Best Algos's name. "Best" is evaluated during dataset collection.
    '''
    def __init__(self):
        self.prompts = []
        self.probed_ts = []
        self.labels = []

    def add(self, prompts,probed_ts,label):
        self.prompts.append(prompts)
        self.probed_ts.append(probed_ts)
        self.labels.append(label)


    def __len__(self):
        return len(self.labels)