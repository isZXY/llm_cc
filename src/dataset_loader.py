import pickle
from datasets import Dataset

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
    
class DatasetLoader:
    '''
    Load and Return dataset.
    load from dataset_pool, transfer label index, split data(train,validation,test)
    '''
    def __init__(self, dataset_pool_path):
        self.dataset_pool = pickle.load(open(dataset_pool_path, 'rb'))
        self.id2label = {0: "vegas", 1: "htcp",2:"westwood",3:"bbr",4:"cubic",5:"reno",6:"bic"}
        self.label2id = {"vegas": 0, "htcp": 1,"westwood":2,"bbr":3,"cubic":4,"reno":5,"bic":6}

    def load_dataset(self):
        # warning! input data should be normalized during this period
        labels_as_ids = [self.label2id[label] for label in self.dataset_pool.labels]
        result_dict = {
            "features":self.dataset_pool.prompts,
            "label": labels_as_ids,
        }
        dataset= Dataset.from_dict(result_dict)
        train_val_test_split = dataset.train_test_split(test_size=0.4, seed=42)  # 60% train, 40% (val+test)
        valid_test_split = train_val_test_split['test'].train_test_split(test_size=0.5, seed=42)  # 50% val, 50% test
        dataset_train = train_val_test_split['train']
        dataset_val = valid_test_split['train']
        dataset_test = valid_test_split['test']
        return dataset_train,dataset_val,dataset_test
    
    def get_id2label_dict(self):
        return self.id2label