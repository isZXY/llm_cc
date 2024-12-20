import pickle
from torch.utils.data import Dataset, random_split, DataLoader
import numpy as np

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


def discount_returns(rewards, gamma, scale):
    returns = [0 for _ in range(len(rewards))]
    returns[-1] = rewards[-1]
    for i in reversed(range(len(rewards) - 1)):
        returns[i] = rewards[i] + gamma * returns[i + 1]
    for i in range(len(returns)):
        returns[i] /= scale  # scale down return
    return returns


class _ABRDataset(Dataset):
    """
    A dataset class that wraps the experience pool.
    """
    def __init__(self, exp_pool_path, gamma=1., scale=10, max_length=8, sample_step=None) -> None:
        """
        :param exp_pool: the experience pool
        :param gamma: the reward discounted factor
        :param scale: the factor to scale the return
        :param max_length: the max length of past features
        """
        if sample_step is None:
            sample_step = max_length

        with open(exp_pool_path, 'rb') as exp_pool:
            self.exp_pool = pickle.load(exp_pool)
            self.exp_pool_size = len(self.exp_pool)

        self.gamma = gamma
        self.scale = scale
        self.max_length = max_length

        self.returns = []
        self.timesteps = []
        self.rewards = []

        self.exp_dataset_info = {}


        self._normalize_rewards()
        self._compute_returns()
        self.exp_dataset_info.update({
            'max_action': max(self.actions),
            'min_action': min(self.actions)
        })

        self.dataset_indices = list(range(0, self.exp_pool_size - max_length + 1, min(sample_step, max_length)))
    
    def sample_batch(self, batch_size=1, batch_indices=None):
        """
        Sample a batch of data from the experience pool.
        :param batch_size: the size of a batch. For CJS task, batch_size should be set to 1 due to the unstructural data format.
        """
        if batch_indices is None:
            batch_indices = np.random.choice(len(self.dataset_indices), size=batch_size)
        batch_states, batch_actions, batch_returns, batch_timesteps = [], [], [], []
        for i in range(batch_size):
            states, actions, returns, timesteps = self[batch_indices[i]]
            batch_states.append(states)
            batch_actions.append(actions)
            batch_returns.append(returns)
            batch_timesteps.append(timesteps)
        return batch_states, batch_actions, batch_returns, batch_timesteps
    
    @property
    def states(self):
        return self.exp_pool.states

    @property
    def actions(self):
        return self.exp_pool.actions
    
    @property
    def dones(self):
        return self.exp_pool.dones
    
    def __len__(self):
        return len(self.dataset_indices)
    
    def __getitem__(self, index):
        start = self.dataset_indices[index]
        end = start + self.max_length
        return self.states[start:end], self.actions[start:end], self.returns[start:end], self.timesteps[start:end]

    def _normalize_rewards(self):
        min_reward, max_reward = min(self.exp_pool.rewards), max(self.exp_pool.rewards)
        rewards = (np.array(self.exp_pool.rewards) - min_reward) / (max_reward - min_reward)
        self.rewards = rewards.tolist()
        self.exp_dataset_info.update({
            'max_reward': max_reward,
            'min_reward': min_reward,
        })

    def _compute_returns(self):
        """
        Compute returns (discounted cumulative rewards)
        """
        episode_start = 0
        while episode_start < self.exp_pool_size:
            try:
                episode_end = self.dones.index(True, episode_start) + 1
            except ValueError:
                episode_end = self.exp_pool_size
            self.returns.extend(discount_returns(self.rewards[episode_start:episode_end], self.gamma, self.scale))
            self.timesteps += list(range(episode_end - episode_start))
            episode_start = episode_end
        assert len(self.returns) == len(self.timesteps)
        self.exp_dataset_info.update({
            # for normalizing rewards/returns
            'max_return': max(self.returns),
            'min_return': min(self.returns),

            # to help determine the maximum size of timesteps embedding
            'min_timestep': min(self.timesteps),
            'max_timestep': max(self.timesteps),
        })



class DatasetLoader:
    '''
    Load Dataset and Return dataloader.
    '''

    def __init__(self,task_name,dataset_pool_path, batch_size=1, train_prop=0.6, val_prop=0.2, test_prop=0.2):
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
