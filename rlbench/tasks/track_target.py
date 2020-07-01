from typing import List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.const import colors
from rlbench.backend.task import Task
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.conditions import DetectedCondition

import cv2
from rlbench.tasks.reach_target import ReachTarget
from rlbench.backend.observation import Observation


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class TrackTarget(ReachTarget):
    distance_threshold = 0.05
    desired_goal = np.array((0.0, 0.0, 0.50), dtype=np.float)

    def init_task(self) -> None:
        super().init_task()
        self.target_handle = self.target.get_handle()

    def _compute_achieved_goal(self, wrist_mask: np.ndarray) -> np.ndarray:
        h, w = wrist_mask.shape
        target_mask = (wrist_mask == self.target_handle).astype(np.float)
        if np.sum(target_mask > 0):
            M = cv2.moments(target_mask)
            cX = M["m10"] / M["m00"]
            cY = M["m01"] / M["m00"]
            res = np.array([
                (cX - w / 2) / (w / 2),
                (cY - h / 2) / (h / 2),
                np.sum(target_mask > 0) / (h * w),
            ])
        else:
            res = np.array([1.0, 1.0, -1.0])
        return res

    def decorate_observation(self, obs: Observation) -> Observation:
        """To retrive the position of the achieved_goal (target) and the desired_goal (0, 0, )
        in the image axis

        :param observation: The Observation for this time step.
        :return: The modified Observation.
        """
        assert obs.wrist_mask is not None, "{} should have wrist_mask".format(self.get_name())
        obs.achieved_goal = self._compute_achieved_goal(obs.wrist_mask)
        obs.desired_goal = self.desired_goal.copy()
        low_dim_data_length = len(obs.task_low_dim_state)
        obs.robot_low_dim_data = obs.get_low_dim_data()[:-low_dim_data_length]
        return obs

    def success(self):
        """Never end according to openai gym robotics
        modify the reward function (directly returned in task environment)"""
        all_met = True
        one_terminate = False
        for cond in self._success_conditions:
            met, terminate = cond.condition_met()
            all_met &= met
            one_terminate |= terminate
        reward = 0.0 if all_met else -1.0
        done = False
        return reward, done

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info):
        d = goal_distance(achieved_goal, desired_goal)
        # return -(d > self.distance_threshold).astype(np.float32)
        return -d

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)
