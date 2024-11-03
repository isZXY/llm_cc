import pickle
from torch.utils.data import Dataset, random_split, DataLoader


class _CCDataset(Dataset):
    def __init__(self, dataset_pool_path):
        self.dataset_pool = pickle.load(open(dataset_pool_path, 'rb'))
        self.id2label = {0: "vegas", 1: "htcp", 2: "westwood",
                         3: "bbr", 4: "cubic", 5: "reno", 6: "bic"}
        self.label2id = {"vegas": 0, "htcp": 1, "westwood": 2,
                         "bbr": 3, "cubic": 4, "reno": 5, "bic": 6}
        labels_as_ids = [self.label2id[label]
                         for label in self.dataset_pool.labels]

        self.prompt_list = self.dataset_pool.prompts
        self.ts_list = self.dataset_pool.probed_ts
        self.label_list = labels_as_ids

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        prompt = self.prompt_list[idx]
        ts = self.ts_list[idx]
        label = self.label_list[idx]
        return prompt, ts, label

class _ABRDataset(Dataset):
    def __init__(self, dataset_pool_path):
        self.dataset_pool = pickle.load(open(dataset_pool_path, 'rb'))
        
        self.prompt_list = self.dataset_pool.prompts
        self.ts_list = self.dataset_pool.probed_ts
        self.label_list = self.dataset_pool.labels

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        prompt = self.prompt_list[idx]
        ts = self.ts_list[idx]
        label = self.label_list[idx]
        return prompt, ts, label

class DatasetLoader:
    '''
    Load Dataset and Return dataloader.
    '''

    def __init__(self,task_name,dataset_pool_path, batch_size=2, train_prop=0.6, val_prop=0.2, test_prop=0.2):
        if task_name == "CC":
            self.dataset = _CCDataset(dataset_pool_path)
        elif task_name == "ABR":
            self.dataset = _ABRDataset(dataset_pool_path)
        self.batch_size = batch_size
        self.train_prop = train_prop
        self.val_prop = val_prop
        self.test_prop = test_prop

        assert self.train_prop + self.val_prop + self.test_prop == 1

    def load_dataset(self):
        train_size, val_size, test_size = int(self.train_prop * len(self.dataset)), int(self.val_prop * len(self.dataset)), len(
            self.dataset) - int(self.train_prop * len(self.dataset)) - int(self.val_prop * len(self.dataset))
        train_dataset, val_dataset, test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size])

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,drop_last=True)
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,drop_last=True)
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,drop_last=True)
        return train_loader, val_loader, test_loader

# class DatasetLoader:
#     '''
#     Load and Return dataset.
#     load from dataset_pool, transfer label index, split data(train,validation,test)
#     '''
#     def __init__(self, dataset_pool_path):
#         self.dataset_pool = pickle.load(open(dataset_pool_path, 'rb'))
#         self.id2label = {0: "vegas", 1: "htcp",2:"westwood",3:"bbr",4:"cubic",5:"reno",6:"bic"}
#         self.label2id = {"vegas": 0, "htcp": 1,"westwood":2,"bbr":3,"cubic":4,"reno":5,"bic":6}

#     def load_dataset(self):
#         # warning! input data should be normalized during this period
#         labels_as_ids = [self.label2id[label] for label in self.dataset_pool.labels]
#         result_dict = {
#             "features":self.dataset_pool.prompts,
#             "ts": self.dataset_pool.probed_ts,
#             "label": labels_as_ids,
#         }
#         dataset= Dataset.from_dict(result_dict)
#         train_val_test_split = dataset.train_test_split(test_size=0.4, seed=42)  # 60% train, 40% (val+test)
#         valid_test_split = train_val_test_split['test'].train_test_split(test_size=0.5, seed=42)  # 50% val, 50% test
#         dataset_train = train_val_test_split['train']
#         dataset_val = valid_test_split['train']
#         dataset_test = valid_test_split['test']
#         return dataset_train,dataset_val,dataset_test

#     def get_id2label_dict(self):
#         return self.id2label
