import os
from gym.envs.registration import register

import rlbench.backend.task as task
from rlbench.utils import name_to_task_class
from rlbench.gym.rlbench_env import RLBenchEnv
from rlbench.gym.rlbench_goalenv import RLBenchGoalEnv
from rlbench.gym.rlbench_dvrkenv import RLBenchDvrkEnv

GOAL_TASKS = [
    "reach_target_goal.py",
]

DVRK_TASKS = [
    "track_target.py",
]

TASKS = [t for t in os.listdir(task.TASKS_PATH)
         if t != '__init__.py' and t.endswith('.py')
         and t not in GOAL_TASKS
         and t not in DVRK_TASKS
        ]


def register_task(task_file, observation_mode,
                  entry_point, max_episode_steps=None, render_mode=None):
    task_name = task_file.split('.py')[0]
    task_class = name_to_task_class(task_name)
    if render_mode:
        env_id = '{}-{}-render-v0'.format(task_name, observation_mode)
    else:
        env_id = '{}-{}-v0'.format(task_name, observation_mode)
    register(
        id=env_id,
        entry_point=entry_point,
        kwargs={
            'task_class': task_class,
            'observation_mode': observation_mode,
            'render_mode': render_mode,
        },
        max_episode_steps=max_episode_steps,
    )


for task_file in TASKS:
    entry_point = 'rlbench.gym:RLBenchEnv'
    register_task(task_file,
                  observation_mode='state',
                  entry_point=entry_point)
    register_task(task_file,
                  observation_mode='vision',
                  entry_point=entry_point)

for task_file in GOAL_TASKS:
    entry_point = 'rlbench.gym:RLBenchGoalEnv'
    register_task(task_file,
                  observation_mode='state',
                  entry_point=entry_point,
                  max_episode_steps=50)
    register_task(task_file,
                  observation_mode='vision',
                  entry_point=entry_point,
                  max_episode_steps=50)
    register_task(task_file,
                  observation_mode='state',
                  entry_point=entry_point,
                  max_episode_steps=50,
                  render_mode='rgb_array')
    register_task(task_file,
                  observation_mode='vision',
                  entry_point=entry_point,
                  max_episode_steps=50,
                  render_mode='rgb_array')

for task_file in DVRK_TASKS:
    entry_point = 'rlbench.gym:RLBenchDvrkEnv'
    register_task(task_file,
                  observation_mode='state',
                  entry_point=entry_point,
                  max_episode_steps=100)
    register_task(task_file,
                  observation_mode='vision',
                  entry_point=entry_point,
                  max_episode_steps=100)
    register_task(task_file,
                  observation_mode='state',
                  entry_point=entry_point,
                  max_episode_steps=100,
                  render_mode='rgb_array')
    register_task(task_file,
                  observation_mode='vision',
                  entry_point=entry_point,
                  max_episode_steps=100,
                  render_mode='rgb_array')
