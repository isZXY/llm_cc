class ExperiencePool:
    """
    Experience pool for collecting trajectories.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def add(self, state, action, reward):
        self.states.append(state)  # sometime state is also called obs (observation)
        self.actions.append(action)
        self.rewards.append(reward)

    def __len__(self):
        return len(self.states)


class ExperiencePoolClassification:
    def __init__(self):
        self.features = []
        self.labels = []

    def add(self, feature, label):
        self.features.append(feature)  # sometime state is also called obs (observation)
        self.labels.append(label)

    def __len__(self):
        return len(self.labels)