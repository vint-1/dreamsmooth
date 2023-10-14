import embodied


class Agx(embodied.envs.from_gym.FromGym):

  def __init__(self, task, **kwargs):
    assert task.startswith('agx'), f"wrong task name: {task}"

    import agxGym
    self._repeat = kwargs.pop("repeat", 1)
    super().__init__(task, **kwargs)

  def step(self, ac):
    reward = 0
    for _ in range(self._repeat):
        ob = super().step(ac)
        done = ob["is_last"]
        reward += ob["reward"]
        if done:
          break
    ob["reward"] = reward
    return ob
