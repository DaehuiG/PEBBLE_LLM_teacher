# LLM teacher based Preference-Based Reinforcement Learning

PEBBLE reproduce and using LLM teacher for Preference Based Reinforcement Learning

## Reproduce with original Code

ButtonPress Metaworld Environment, 10000, equal.

![scripted](https://github.com/DaehuiG/PEBBLE_LLM_teacher/blob/test/llm/Images/output_scripted_image.png?raw=true)

## First Experiment

ButtonPress Metaworld Environment, 10000, equal + gpt-4o based feedback

prompt
```
"model": "gpt-4o",
"messages" : [{
    "role": "system",
    "content": 
    '''
    You are a helpful and honest judge of good tasking and progress for AI agent in DMControl ButtonPress RL environment. Always answer as helpfully as possible, while being truthful.
    If you don't know the answer to a question, please don't share false information. \n
    I'm looking to have you evalute a buttonpress task in the DMControl RL environment.
    Your role will be to assess which actions are more efficient to achieving good score in given RL environment.
    \n \
    The basic information for the evaluation is as follows. \n
        - Environment : ButtonPress -v2 metaworld \n
        - Task Description : Instruct the robot to press a button located along the y-axis, requiring precise positioning and force application. \n
        - Task Objective : Control robot arm to press button. \n
        - The Pythonic class-like environment abstraction is : \n
    class SawyerButtonPressEnvV2(gym.Env): \n
        def __init__(self): \n
            self.robot: Robot # the Sawyer robot in the environment \n
            self.button: RigidObject # the button object in the environment \n
            self.goal_position: np.ndarray[(3,)] # 3D position of the goal (button pressed position) \n
            self.trajectory: Trajectory # stores the trajectory of the episode \n
        def reset(self) -> np.ndarray: \n
            # Reset the environment and return initial observation \n
        def step(self, action: np.ndarray) -> tuple: \n
            # Perform one step and return (observation, reward, done, info) \n
        def get_trajectory(self) -> Trajectory: \n
            # Return the recorded trajectory \n
    class Robot: \n
        def __init__(self): \n
            self.ee_position: np.ndarray[(3,)] # 3D position of the end-effector \n
            self.joint_positions: np.ndarray[(7,)] # 7 joint positions of Sawyer robot \n
            self.joint_velocities: np.ndarray[(7,)] # 7 joint velocities of Sawyer robot \n
    class RigidObject: \n
        def __init__(self): \n
            self.position: np.ndarray[(3,)] # 3D position of the object (button) \n
            self.quaternion: np.ndarray[(4,)] # quaternion of the object (button) \n
    class Trajectory: \n
        def __init__(self, max_length=25): \n
            self.states: deque # queue of states, max length 25 \n
            self.actions: deque # queue of actions, max length 25 \n
            self.observations: deque # queue of observations, max length 25 \n
        def add_step(self, state: dict, action: np.ndarray, observation: np.ndarray): \n
            # Add a step to the trajectory \n
        def __len__(self) -> int: \n
            # Return the number of steps in the trajectory \n
    class State: \n
        def __init__(self): \n
            self.robot: Robot # state of the robot \n
            self.button: RigidObject # state of the button \n
    \n

    I plan to inform you about two robot trajectories within the above environment, then please choose which one is better. You will also recieve reward sum which was predicted by current reward predictor.
    You MUST return short answer 0, 1 or -1 for all given samples. (Return results with seperated by comma. For example, 1,0,-1,0, ... )
    If first one is better then please return 0, If second one is better then please return 1, If there is no specific difference between two trajectories then just return -1.
    You can return -1 if there is no difference between two trajectory reward. I will inform you how threshold of this difference is."
    '''
    },
    {
    "role": "user",
    "content": (self.prompt_generator(sa_t_1, sa_t_2, sum_r_t_1, sum_r_t_2) + f"If you think reward difference is under {self.teacher_thres_equal}, then feedback should be prefered eqaully. Please return 0, 1 or -1 feedbacks for all {self.mb_size} samples to teach the agent. You MUST return exactly {self.mb_size} of feedbacks.")
    }],
"temperature": 0
```
 + result

![gpt-4o-first](https://github.com/DaehuiG/PEBBLE_LLM_teacher/blob/test/llm/Images/output_gpt_first_image.png?raw=true)

## Install

```
conda env create -f conda_env.yml
pip install -e .[docs,tests,extra]
cd custom_dmcontrol
pip install -e .
cd custom_dmc2gym
pip install -e .
pip install git+https://github.com/rlworkgroup/metaworld.git@master#egg=metaworld
pip install pybullet
```

## Run experiments using GT rewards


### SAC & SAC + unsupervised pre-training

Experiments can be reproduced with the following:

```
./scripts/[env_name]/run_sac.sh 
./scripts/[env_name]/run_sac_unsuper.sh
```


### PPO & PPO + unsupervised pre-training

Experiments can be reproduced with the following:

```
./scripts/[env_name]/run_ppo.sh 
./scripts/[env_name]/run_ppo_unsuper.sh
```

## Run experiments on irrational teacher

To design more realistic models of human teachers, we consider a common stochastic model and systematically manipulate its terms and operators:

```
teacher_beta: rationality constant of stochastic preference model (default: -1 for perfectly rational model)
teacher_gamma: discount factor to model myopic behavior (default: 1)
teacher_eps_mistake: probability of making a mistake (default: 0)
teacher_eps_skip: hyperparameters to control skip threshold (\in [0,1])
teacher_eps_equal: hyperparameters to control equal threshold (\in [0,1])
```

In B-Pref, we tried the following teachers:

`Oracle teacher`: (teacher_beta=-1, teacher_gamma=1, teacher_eps_mistake=0, teacher_eps_skip=0, teacher_eps_equal=0)

`Mistake teacher`: (teacher_beta=-1, teacher_gamma=1, teacher_eps_mistake=0.1, teacher_eps_skip=0, teacher_eps_equal=0)

`Noisy teacher`: (teacher_beta=1, teacher_gamma=1, teacher_eps_mistake=0, teacher_eps_skip=0, teacher_eps_equal=0)

`Skip teacher`: (teacher_beta=-1, teacher_gamma=1, teacher_eps_mistake=0, teacher_eps_skip=0.1, teacher_eps_equal=0)

`Myopic teacher`: (teacher_beta=-1, teacher_gamma=0.9, teacher_eps_mistake=0, teacher_eps_skip=0, teacher_eps_equal=0)

`Equal teacher`: (teacher_beta=-1, teacher_gamma=1, teacher_eps_mistake=0, teacher_eps_skip=0, teacher_eps_equal=0.1)


### PEBBLE

Experiments can be reproduced with the following:

```
./scripts/[env_name]/[teacher_type]/[max_budget]/run_PEBBLE.sh [sampling_scheme: 0=uniform, 1=disagreement, 2=entropy]
```

### PrefPPO

Experiments can be reproduced with the following:

```
./scripts/[env_name]/[teacher_type]/[max_budget]/run_PrefPPO.sh [sampling_scheme: 0=uniform, 1=disagreement, 2=entropy]
```

note: full hyper-paramters for meta-world will be updated soon!

