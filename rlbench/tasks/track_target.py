from typing import List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.const import colors
from rlbench.backend.task import Task
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.observation import Observation


class ReachTarget(Task):

    def init_task(self) -> None:
        self.target = Shape('target')
        self.distractor0 = Shape('distractor0')
        self.distractor1 = Shape('distractor1')
        self.boundaries = Shape('boundary')
        success_sensor = ProximitySensor('success')
        self.register_success_conditions(
            [DetectedCondition(self.robot.arm.get_tip(), success_sensor)])

    def init_episode(self, index: int) -> List[str]:
        color_name, color_rgb = colors[index]
        self.target.set_color(color_rgb)
        color_choices = np.random.choice(
            list(range(index)) + list(range(index + 1, len(colors))),
            size=2, replace=False)
        for ob, i in zip([self.distractor0, self.distractor1], color_choices):
            name, rgb = colors[i]
            ob.set_color(rgb)
        b = SpawnBoundary([self.boundaries])
        for ob in [self.target, self.distractor0, self.distractor1]:
            b.sample(ob, min_distance=0.2,
                     min_rotation=(0, 0, 0), max_rotation=(0, 0, 0))

        return ['reach the %s target' % color_name,
                'touch the %s ball with the panda gripper' % color_name,
                'reach the %s sphere' %color_name]

    def variation_count(self) -> int:
        return len(colors)

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def get_low_dim_state(self) -> np.ndarray:
        # One of the few tasks that have a custom low_dim_state function.
        return np.array(self.target.get_position())

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