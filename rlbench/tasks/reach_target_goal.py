from typing import Dict

import numpy as np
from pyrep import PyRep
from pyrep.const import ObjectType
from pyrep.objects.dummy import Dummy
from pyrep.objects.force_sensor import ForceSensor
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object

from rlbench.tasks.reach_target import ReachTarget
from rlbench.backend.observation import Observation


class ReachTargetGoal(ReachTarget):

    def decorate_observation(self, observation: Observation) -> Observation:
        """To retrive the position of the achieved_goal (tip) and the desired_goal (target)

        :param observation: The Observation for this time step.
        :return: The modified Observation.
        """
        assert (observation.gripper_pose is not None) and (observation.task_low_dim_state is not None), \
            "{} should have low_dim_obs".format(self.get_name())
        observation.achieved_goal = observation.gripper_pose[:3]
        observation.desired_goal = observation.task_low_dim_state.copy()
        return observation

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


