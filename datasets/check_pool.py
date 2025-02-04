import pickle
from collections import Counter


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
        
        # self.prompts = [] 
        self.states = [] # probed ts组成
        self.actions = [] # 对应选择的label
        self.rewards = []
        self.timesteps = []

    def add(self, state, action, reward, timestep):
        # self.prompts.append(prompt)
        self.states.append(state)  # sometime state is also called obs (observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.timesteps.append(timestep)

    def extend(self, states, actions, rewards, timestep):
        # self.prompts.extend(prompts)
        self.states.extend(states)  # sometime state is also called obs (observation)
        self.actions.extend(actions)
        self.rewards.extend(rewards)
        self.timesteps.extend(timestep)


    def __len__(self):
        return len(self.states)


# 替换成你pkl文件的路径
pkl_file_path = '/data3/wuduo/xuanyu/llmcc/datasets/ABR/dataset_pool_ABR.pkl'



# 加载 pkl 文件
with open(pkl_file_path, 'rb') as f:
    data = pickle.load(f)

# 输出加载的数据


data.stat_distribute()
