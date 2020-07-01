from gym.envs.registration import register
import rlbench.backend.task as task
import os
from rlbench.utils import name_to_task_class
from rlbench.gym.rlbench_env import RLBenchEnv
from rlbench.gym.rlbench_goalenv import RLBenchGoalEnv

TASKS = [t for t in os.listdir(task.TASKS_PATH)
         if t != '__init__.py' and t.endswith('.py')
         and 'goal' not in t
         and 'wrist' not in t
        ]

GOAL_TASKS = [t for t in os.listdir(task.TASKS_PATH)
         if t != '__init__.py' and t.endswith('.py') and 'goal' in t]

for task_file in TASKS:
    task_name = task_file.split('.py')[0]
    task_class = name_to_task_class(task_name)
    register(
        id='%s-state-v0' % task_name,
        entry_point='rlbench.gym:RLBenchEnv',
        kwargs={
            'task_class': task_class,
            'observation_mode': 'state'
        }
    )
    register(
        id='%s-vision-v0' % task_name,
        entry_point='rlbench.gym:RLBenchEnv',
        kwargs={
            'task_class': task_class,
            'observation_mode': 'vision'
        }
    )

for task_file in GOAL_TASKS:
    task_name = task_file.split('.py')[0]
    task_class = name_to_task_class(task_name)
    register(
        id='%s-state-v0' % task_name,
        entry_point='rlbench.gym:RLBenchGoalEnv',
        kwargs={
            'task_class': task_class,
            'observation_mode': 'state',
        },
        max_episode_steps=50,
    )
    register(
        id='%s-state-render-v0' % task_name,
        entry_point='rlbench.gym:RLBenchGoalEnv',
        kwargs={
            'task_class': task_class,
            'observation_mode': 'state',
            'render_mode': 'rgb_array',
        },
        max_episode_steps=50,
    )
    register(
        id='%s-vision-v0' % task_name,
        entry_point='rlbench.gym:RLBenchGoalEnv',
        kwargs={
            'task_class': task_class,
            'observation_mode': 'vision'
        },
        max_episode_steps=50,
    )
    register(
        id='%s-vision-render-v0' % task_name,
        entry_point='rlbench.gym:RLBenchGoalEnv',
        kwargs={
            'task_class': task_class,
            'observation_mode': 'vision',
            'render_mode': 'rgb_array',
        },
        max_episode_steps=50,
    )
