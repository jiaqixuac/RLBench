from typing import List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.const import colors
from rlbench.backend.task import Task
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.conditions import DetectedCondition

from rlbench.tasks.reach_target import ReachTarget
from rlbench.backend.observation import Observation


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class ReachTargetGoal(ReachTarget):
    distance_threshold = 0.05

    def decorate_observation(self, obs: Observation) -> Observation:
        """To retrive the position of the achieved_goal (tip) and the desired_goal (target)

        :param observation: The Observation for this time step.
        :return: The modified Observation.
        """
        assert (obs.gripper_pose is not None) and (obs.task_low_dim_state is not None), \
            "{} should have low_dim_obs".format(self.get_name())
        obs.achieved_goal = obs.gripper_pose[:3]
        obs.desired_goal = obs.task_low_dim_state.copy()
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
        return -(d > self.distance_threshold).astype(np.float32)

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)
