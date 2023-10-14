import os
if 'MUJOCO_GL' not in os.environ:
      os.environ['MUJOCO_GL'] = 'egl'
import numpy as np
import gymnasium as gym
from gym.utils.ezpickle import EzPickle
import gymnasium_robotics.envs.shadow_dexterous_hand as hand
import embodied

class Hand(embodied.Env):

    def __init__(self, task, height = 64, width = 64):
        self._gymenv = FixedMultiGoalHandEnv(task, height = height, width = width)
        self._gymenv = HandWrapper(self._gymenv)
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

class HandWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    @property
    def observation_space(self):
        obs_space = self.env.observation_space
        obs_space.spaces["i_tasks_completed"] = gym.spaces.Box(low=0, high=len(self.env.goal_sequence), dtype=np.float32)
        obs_space.spaces["image"] = gym.spaces.Box(low=0, high=255, shape=(self.env.height, self.env.width, 3), dtype=np.uint8)
        return obs_space

    # def _process_ob(self, ob):
    #     ob.update({"image": self.env.render()})

    # def reset(self, seed=None):
    #     ob, info = self.env.reset()
    #     return self._process_ob(ob), info

    # def step(self, ac):
    #     ob, rew, done, trunc, info = self.env.step(ac)
    #     return self._process_ob(ob), rew, done, trunc, info

class FixedMultiGoalHandEnv(hand.MujocoManipulateEnv, EzPickle):
    camera_config = {
        "distance": 0.28,
        "azimuth": 55.0,
        "elevation": -25.0,
        "lookat": np.array([1.04, 0.92, 0.14]),
    }
    goal_sequence = np.array([
        [0.77783791, -0., -0., -0.62846495],
        [0.99766095, -0., -0., -0.06835659],
        [0.08094724, -0., -0., -0.99671839],
        [0.20883383, 0., 0., 0.97795114],
    ])

    def __init__(
            self,
            task,
            render_mode = "rgb_array",
            **kwargs,
        ):
        task_id, reward_type = task.split("_", 2)
        rotation_mapping = {"RotateZ": ("ignore", "z"), "RotateParallel": ("ignore", "parallel"), "RotateXYZ": ("ignore", "xyz")} #TODO add full rotation setting
        target_position, target_rotation = rotation_mapping[task_id]
        assert task_id == "RotateZ", "only rotate z supported"
        assert reward_type == "dense", "only dense reward supported"

        hand.MujocoManipulateEnv.__init__(
            self,
            model_path=os.path.join("hand", "manipulate_block.xml"),
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type,
            default_camera_config=FixedMultiGoalHandEnv.camera_config,
            render_mode=render_mode,
            **kwargs,
        )
        EzPickle.__init__(self, target_position, target_rotation, reward_type, **kwargs)
        self._need_new_goal = False
        self._goal_num = 0
        self._completed_tasks = 0

    def reset(self, seed = None, options = None):
        #TODO fix issue where random goal flashes on screen for 1 frame at start of episode
        self._goal_num = 0
        self._completed_tasks = 0
        self._need_new_goal = False
        ob, info = super().reset(seed = seed, options = options)
        self.goal = self._get_goal_by_num(0)
        return self._process_obs(ob), info

    def step(self, ac):
        if self._need_new_goal:
            self._set_new_goal()
        ob, r, done, trunc, info = super().step(ac)
        # check if goal has been reached, then do a yeet on the next step
        self._need_new_goal = (self._is_success(ob["achieved_goal"], self.goal).astype(np.float32) > 0.5)
        if self._need_new_goal:
            r += 300
        return self._process_obs(ob), r, done, trunc, info

    def _sample_goal(self):
        return self._get_goal_by_num(self._goal_num)

    def _get_goal_by_num(self, goal_num):
        # Select a goal for the object position.
        target_pos = self._utils.get_joint_qpos(self.model, self.data, "object:joint")[:3]
        # Select a goal for the object rotation.
        target_quat = FixedMultiGoalHandEnv.goal_sequence[goal_num]
        target_quat /= np.linalg.norm(target_quat)  # normalized quaternion
        goal = np.concatenate([target_pos, target_quat])
        return goal

    def _set_new_goal(self):
        self._completed_tasks += 1
        self._goal_num = (self._goal_num + 1) % len(FixedMultiGoalHandEnv.goal_sequence)
        self.goal = self._get_goal_by_num(self._goal_num)
    
    def _process_obs(self, ob):
        # adds pixel info to observations, removes proprio info
        # adds logging info to observations
        img = self.render()
        ob.update({"i_tasks_completed": self._completed_tasks, "image": img})
        return ob