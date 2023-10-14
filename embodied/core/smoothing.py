from scipy.ndimage import gaussian_filter1d, convolve
import numpy as np

# Wrapper to apply gaussian smoothing before adding to replay buffer
# current strategy: since episode lengths not assumed to be uniform, we will have a buffer to store the current episode
# only when episode completes, will we process and store entire episode's transitions to the replay buffer
class GaussianSmoothing:
    def __init__(self, replay, sigma = 0):
        self._replay = replay
        self.rew_buffer = {}
        self.current_transitions = {}
        self.sigma = float(sigma)
        assert self.sigma > 0, "make sure gaussian smooothing sigma > 0"

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._replay, name)
        except AttributeError:
            raise ValueError(name)

    def __len__(self):
        return len(self._replay)

    def add(self, step, worker=0, load=False):
        step = step.copy()
        step.update({"kwargs": {"load": load}})
        if worker not in self.current_transitions:
            self.current_transitions[worker] = []
            self.rew_buffer[worker] = []
        self.current_transitions[worker].append(step) 
        self.rew_buffer[worker].append(step["reward"])
        if step["is_last"]:
            self._add_current_episode(worker)
    
    def _add_current_episode(self, worker):
        # do gaussian smoothing, then add all steps to replay buffer
        rew_smooth = gaussian_filter1d(np.array(self.rew_buffer[worker]), self.sigma, mode="nearest")
        for i, step in enumerate(self.current_transitions[worker]):
            kwargs = step.pop("kwargs")
            rew_raw = step["reward"]
            step.update({"reward_raw": rew_raw, "reward": rew_smooth[i]})
            self._replay.add(step, worker, **kwargs)
        self.rew_buffer[worker].clear()
        self.current_transitions[worker].clear()

# Wrapper to apply exponential moving average smoothing before adding to replay buffer
class ExponentialSmoothing:
    def __init__(self, replay, alpha = 0):
        self._replay = replay
        self.rew_buffer = {}
        self.alpha = float(alpha)
        self.beta = 1 - self.alpha
        assert self.alpha > 0, "make sure EMA constant > 0"
        assert self.alpha < 1, "make sure EMA constant < 1"

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._replay, name)
        except AttributeError:
            raise ValueError(name)

    def __len__(self):
        return len(self._replay)

    def add(self, step, worker=0, load=False):
        step = step.copy()
        current_rew = step["reward"]
        rew_smooth = self.alpha * self.rew_buffer.get(worker, current_rew) + self.beta * current_rew
        step.update({"reward_raw": current_rew, "reward": rew_smooth})
        self._replay.add(step, worker, load)
        if step["is_last"]:
            self.rew_buffer.pop(worker)
        else:
            self.rew_buffer[worker] = rew_smooth

# Wrapper to apply uniform smoothing before adding to replay buffer
class UniformSmoothing:
    def __init__(self, replay, delta = 0, kernel_type="uniform"):
        self._replay = replay
        self.rew_buffer = {}
        self.current_transitions = {}
        self.delta = int(delta)
        assert self.delta > 0, "make sure window for uniform smoothing is an integer > 0"
        if kernel_type == "uniform":
            self.filter = (1.0/self.delta) * np.array([1] * self.delta)
        elif kernel_type == "uniform_before":
            self.filter = (1.0/self.delta) * np.array([1] * self.delta + [0] * (self.delta-1))
        elif kernel_type == "uniform_after":
            self.filter = (1.0/self.delta) * np.array([0] * (self.delta-1) + [1] * self.delta)
        else:
            raise NotImplementedError("invalid kernel_type setting for uniform smoothing")

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._replay, name)
        except AttributeError:
            raise ValueError(name)

    def __len__(self):
        return len(self._replay)

    def add(self, step, worker=0, load=False):
        step = step.copy()
        step.update({"kwargs": {"load": load}})
        if worker not in self.current_transitions:
            self.current_transitions[worker] = []
            self.rew_buffer[worker] = []
        self.current_transitions[worker].append(step) 
        self.rew_buffer[worker].append(step["reward"])
        if step["is_last"]:
            self._add_current_episode(worker)
    
    def _add_current_episode(self, worker):
        # do uniform smoothing, then add all steps to replay buffer
        rew_smooth = convolve(np.array(self.rew_buffer[worker]), self.filter, mode="nearest")
        for i, step in enumerate(self.current_transitions[worker]):
            kwargs = step.pop("kwargs")
            rew_raw = step["reward"]
            step.update({"reward_raw": rew_raw, "reward": rew_smooth[i]})
            self._replay.add(step, worker, **kwargs)
        self.rew_buffer[worker].clear()
        self.current_transitions[worker].clear()

def gaussian_rewards(episode, sigma):
    if sigma > 0:
        reward_raw = episode["reward"]
        reward_smooth = gaussian_filter1d(reward_raw, sigma, mode="nearest")
        episode.update({"reward_raw": reward_raw, "reward": reward_smooth})

def exponential_rewards(episode, alpha):
    # TODO Speed this up with some vectorization, or using jax.scan
    if alpha > 0:
        reward_raw = episode["reward"]
        assert len(reward_raw.shape) == 1, "episode reward does not have shape (n,)!"
        reward_smooth = np.zeros_like(reward_raw)
        reward_smooth[-1] = reward_raw[0]
        beta = 1 - alpha
        for i, raw_rew in enumerate(reward_raw):
            reward_smooth[i] = alpha * reward_smooth[i-1] + beta * raw_rew
        episode.update({"reward_raw": reward_raw, "reward": reward_smooth})

def uniform_rewards(episode, delta, kernel_type="uniform"):
    if delta > 0:
        reward_raw = episode["reward"]
        delta = int(delta)

        if kernel_type == "uniform":
            filter = (1.0/delta) * np.array([1] * delta)
        elif kernel_type == "uniform_before":
            filter = (1.0/delta) * np.array([1] * delta + [0] * (delta-1))
        elif kernel_type == "uniform_after":
            filter = (1.0/delta) * np.array([0] * (delta-1) + [1] * delta)
        else:
            raise NotImplementedError("invalid kernel_type setting for uniform smoothing")
        
        reward_smooth = convolve(reward_raw, filter, mode="nearest")
        episode.update({"reward_raw": reward_raw, "reward": reward_smooth})