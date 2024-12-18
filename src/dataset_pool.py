
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
        self.dones = []

    def add(self, state, action, reward, done):
        # self.prompts.append(prompt)
        self.states.append(state)  # sometime state is also called obs (observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def extend(self, states, actions, rewards, dones):
        # self.prompts.extend(prompts)
        self.states.extend(states)  # sometime state is also called obs (observation)
        self.actions.extend(actions)
        self.rewards.extend(rewards)
        self.dones.extend(dones)

    def __len__(self):
        return len(self.states)
