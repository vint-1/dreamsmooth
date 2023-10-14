import os
import gym
import embodied
import numpy as np


class RoboDesk(embodied.Env):

  def __init__(self, task, mode, repeat=1, length=500, resets=True):
    assert mode in ('train', 'eval')
    # TODO: This env variable is meant for headless GPU machines but may fail
    # on CPU-only machines.
    if 'MUJOCO_GL' not in os.environ:
      os.environ['MUJOCO_GL'] = 'egl'
    try:
      from robodesk import robodesk
    except ImportError:
      import robodesk
    task, reward = task.rsplit('_', 1)
    if mode == 'eval':
      reward = 'success'
    assert reward in ('dense', 'sparse', 'success'), reward
    self._gymenv = robodesk.RoboDesk(task, reward, repeat, length)
    from . import from_gym
    self._env = from_gym.FromGym(self._gymenv)

  @property
  def obs_space(self):
    return self._env.obs_space

  @property
  def act_space(self):
    return self._env.act_space

  def step(self, action):
    obs = self._env.step(action)
    obs['is_terminal'] = False
    return obs

class RoboDeskMulti(embodied.Env):

  def __init__(self, _, task_sequence = ["flat_block_in_bin", "upright_block_off_table", "push_green"], repeat=8, length=500, resets=True, image_size=64):
    # note: we ignore task variable that's passed in
    # if 'MUJOCO_GL' not in os.environ:
    #   os.environ['MUJOCO_GL'] = 'egl'
    # import robodesk
    from absl import logging
    logging.set_verbosity(logging.ERROR)

    # self._gymenv = robodesk.RoboDesk(task = task_sequence[0], reward = 'dense', action_repeat = repeat, episode_length = length * repeat, image_size = image_size)
    from . import robodesk_hd
    self._gymenv = robodesk_hd.RoboDeskHD(task = task_sequence[0], reward = 'dense', action_repeat = repeat, episode_length = length * repeat, image_size = image_size)
    self._gymenv = RobodeskWrapper(self._gymenv, task_sequence = task_sequence)
    from . import from_gym
    self._env = from_gym.FromGym(self._gymenv)

  @property
  def obs_space(self):
    return self._env.obs_space

  @property
  def act_space(self):
    return self._env.act_space

  def step(self, action):
    obs = self._env.step(action)
    obs['is_terminal'] = False
    return obs

class RobodeskWrapper(gym.Wrapper):
  def __init__(self, env, task_sequence = ["flat_block_in_bin", "upright_block_off_table", "push_green"]):
    # if output size is none, don't resize the image from wrapped env
    super().__init__(env)
    self._task_sequence = task_sequence
    self._task_num = 0
    self.env.task = self._task_sequence[self._task_num]
    self._need_new_goal = False
    self._all_goals_done = False

  @property
  def observation_space(self): # TODO get observation spaces to handle downsampling of images
    obs_space = self.env.observation_space
    obs_space.spaces["i_tasks_completed"] = gym.spaces.Box(low=0, high=len(self._task_sequence), dtype=np.float32, shape=(1,))
    return obs_space

  def reset(self, seed=None):
    self._task_num = 0
    self.env.task = self._task_sequence[self._task_num]
    self._need_new_goal = False
    self._all_goals_done = False
    ob = self.env.reset()
    return self._add_info(ob), {}

  def step(self, ac): # TODO downsample images if necessary
    if self._need_new_goal:
        self._next_task()
    ob, rew, done, info = self.env.step(ac)
    # check if next task is needed
    self._need_new_goal = self.env._get_task_reward(self.env.task, 'success') > 0.5
    # give bonus reward
    if self._need_new_goal and not self._all_goals_done:
        rew += 300
        self._all_goals_done = self._all_goals_done or (self._task_num == len(self._task_sequence)-1)
    return self._add_info(ob), rew, done, done, info

  def _next_task(self):
    # move to next task
    self._task_num = min(len(self._task_sequence) - 1, self._task_num + 1)
    self.env.task = self._task_sequence[self._task_num]
    self._need_new_goal = False

  def _add_info(self, ob):
    # adds logging info to observations
    completed_tasks = (self._task_num + 1) if self._all_goals_done else self._task_num
    ob.update({"i_tasks_completed": completed_tasks})
    return ob
