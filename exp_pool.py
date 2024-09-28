
class DatasetPool:
    def __init__(self):
        self.prompts = []
        self.labels = []

    def add(self, prompts, label):
        self.prompts.append(prompts)
        self.labels.append(label)

    def __len__(self):
        return len(self.labels)