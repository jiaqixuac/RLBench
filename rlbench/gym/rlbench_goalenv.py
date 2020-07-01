from typing import Union, Dict, Tuple

import gym
from gym import spaces
from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
import numpy as np


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class RLBenchGoalEnv(gym.GoalEnv):
    """An gym goal wrapper for RLBench."""

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, task_class, observation_mode='state',
                 render_mode: Union[None, str] = None):
        self._observation_mode = observation_mode
        self._render_mode = render_mode
        obs_config = ObservationConfig()
        if observation_mode == 'state':
            obs_config.set_all_high_dim(False)
            obs_config.set_all_low_dim(True)
        elif observation_mode == 'vision':
            obs_config.set_all(True)
        else:
            raise ValueError(
                'Unrecognised observation_mode: %s.' % observation_mode)
        action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
        self.env = Environment(
            action_mode, obs_config=obs_config, headless=True)
        self.env.launch()
        self.task = self.env.get_task(task_class)

        _, obs = self.task.reset()

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.env.action_size,))

        if observation_mode == 'state':
            self.observation_space = spaces.Dict({
                'observation': spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.get_low_dim_data().shape),
                'achieved_goal': spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.achieved_goal.shape),
                'desired_goal': spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.desired_goal.shape),
            })
        elif observation_mode == 'vision':
            self.observation_space = spaces.Dict({
                "state": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.get_low_dim_data().shape),
                "left_shoulder_rgb": spaces.Box(
                    low=0, high=1, shape=obs.left_shoulder_rgb.shape),
                "right_shoulder_rgb": spaces.Box(
                    low=0, high=1, shape=obs.right_shoulder_rgb.shape),
                "wrist_rgb": spaces.Box(
                    low=0, high=1, shape=obs.wrist_rgb.shape),
                "front_rgb": spaces.Box(
                    low=0, high=1, shape=obs.front_rgb.shape),
                "wrist_mask": spaces.Box(
                    low=0, high=1, shape=obs.wrist_mask.shape),
                'observation': spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.get_low_dim_data().shape),
                'achieved_goal': spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.achieved_goal.shape),
                'desired_goal': spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.desired_goal.shape),
                })
        self.distance_threshold = 0.05

        if render_mode is not None:
            # Add the camera to the scene
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            self._gym_cam = VisionSensor.create([640, 360])
            self._gym_cam.set_pose(cam_placeholder.get_pose())
            self._gym_cam_wrist = VisionSensor.create([360, 360])
            self._gym_cam_wrist_placeholder = Dummy('cam_wrist_placeholder')
            if render_mode == 'human':
                self._gym_cam.set_render_mode(RenderMode.OPENGL3_WINDOWED)
            else:
                self._gym_cam.set_render_mode(RenderMode.OPENGL3)

    def _extract_obs(self, obs) -> Dict[str, np.ndarray]:
        if self._observation_mode == 'state':
            return {
                "observation": obs.get_low_dim_data(),
                "achieved_goal": obs.achieved_goal,
                "desired_goal": obs.desired_goal,
            }
        elif self._observation_mode == 'vision':
            return {
                "state": obs.get_low_dim_data(),
                "left_shoulder_rgb": obs.left_shoulder_rgb,
                "right_shoulder_rgb": obs.right_shoulder_rgb,
                "wrist_rgb": obs.wrist_rgb,
                "front_rgb": obs.front_rgb,
                "wrist_mask": obs.wrist_mask,
                "observation": obs.get_low_dim_data(),
                "achieved_goal": obs.achieved_goal,
                "desired_goal": obs.desired_goal,
            }

    def render(self, mode='human') -> Union[None, np.ndarray]:
        if mode != self._render_mode:
            raise ValueError(
                'The render mode must match the render mode selected in the '
                'constructor. \nI.e. if you want "human" render mode, then '
                'create the env by calling: '
                'gym.make("reach_target-state-v0", render_mode="human").\n'
                'You passed in mode %s, but expected %s.' % (
                    mode, self._render_mode))
        if mode == 'rgb_array':
            img_front = self._gym_cam.capture_rgb()
            self._gym_cam_wrist.set_pose(self._gym_cam_wrist_placeholder.get_pose())
            img_wrist = self._gym_cam_wrist.capture_rgb()
            # img = img_front
            img = np.concatenate((img_front, img_wrist), axis=1)
            assert np.all(img_front >= 0) and np.all(img_front <= 1)
            return (img * 255).astype(np.uint8)

    def reset(self) -> Dict[str, np.ndarray]:
        descriptions, obs = self.task.reset()
        # print(descriptions)
        del descriptions  # Not used.
        return self._extract_obs(obs)

    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, dict]:
        obs, rew, terminate = self.task.step(action)
        info = {
            'is_success': self._is_success(obs.achieved_goal, obs.desired_goal),
        }
        reward = self.compute_reward(obs.achieved_goal, obs.desired_goal, info)
        if reward != rew:
            print("Goal Env Warning, condition reward: {}, distance reward: {}".format(rew, reward))

        return self._extract_obs(obs), reward, False, info

    def close(self) -> None:
        self.env.shutdown()

    def compute_reward(self, achieved_goal, desired_goal, info):
        d = goal_distance(achieved_goal, desired_goal)
        return -(d > self.distance_threshold).astype(np.float32)

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)
