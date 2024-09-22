import os
import time

from rob831.infrastructure.rl_trainer import RL_Trainer
from rob831.agents.bc_agent import BCAgent
from rob831.policies.loaded_gaussian_policy import LoadedGaussianPolicy

import sys
import numpy as np
import time

import gym
import torch

class BC_Trainer(object):

    def __init__(self, params):

        #######################
        ## AGENT PARAMS
        #######################

        agent_params = {
            'n_layers': params['n_layers'],
            'size': params['size'],
            'learning_rate': params['learning_rate'],
            'learning_rate_step': params['learning_rate_step'],
            'learning_rate_gamma': params['learning_rate_gamma'],
            'max_replay_buffer_size': params['max_replay_buffer_size'],
            }

        self.params = params
        self.params['agent_class'] = BCAgent ## HW1: you will modify this
        self.params['agent_params'] = agent_params

        ################
        ## RL TRAINER
        ################

        self.rl_trainer = RL_Trainer(self.params) ## HW1: you will modify this

        #######################
        ## LOAD EXPERT POLICY
        #######################

        print('Loading expert policy from...', self.params['expert_policy_file'])
        self.loaded_expert_policy = LoadedGaussianPolicy(self.params['expert_policy_file'])
        print('Done restoring expert policy...')

    def run_training_loop(self):

        self.rl_trainer.run_training_loop(
            n_iter=self.params['n_iter'],
            initial_expertdata=self.params['expert_data'],
            collect_policy=self.rl_trainer.agent.actor,
            eval_policy=self.rl_trainer.agent.actor,
            relabel_with_expert=self.params['do_dagger'],
            expert_policy=self.loaded_expert_policy,
        )

def run_expert_policy(params):
    # Load the expert policy
    print('Loading expert policy from...', params['expert_policy_file'])
    expert_policy = LoadedGaussianPolicy(params['expert_policy_file'])
    print('Done restoring expert policy...')

    # Create the environment
    env = gym.make(params['env_name'])
    env.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])

    print(f"created env: {env}")

    # Run the expert policy in the environment for two trajectories
    num_trajectories = 2
    total_rewards = []

    for _ in range(num_trajectories):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = expert_policy.get_action(obs)
            action = np.squeeze(action)  # Remove the extra dimension
            # print(f"dim of obs: {obs.shape} dim of action: {action.shape}")
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            env.render()
        total_rewards.append(total_reward)

    mean_reward = np.mean(total_rewards)
    stddev_reward = np.std(total_rewards)

    print(f"mean: {mean_reward} stddev: {stddev_reward}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem_num', '-p', type=int, default=1) # problem number to run (-p 11, 12, 13, 14, 21, 22, 23, 24)
    parser.add_argument('--expert_policy_file', '-epf', type=str, required=True)  # relative to where you're running this script from
    parser.add_argument('--expert_data', '-ed', type=str, required=True) #relative to where you're running this script from
    parser.add_argument('--env_name', '-env', type=str, help='choices: Ant-v2, Humanoid-v2, Walker-v2, HalfCheetah-v2, Hopper-v2', required=True)
    parser.add_argument('--exp_name', '-exp', type=str, default='pick an experiment name', required=True)
    parser.add_argument('--do_dagger', action='store_true')
    parser.add_argument('--ep_len', type=int, default=1000)

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=5000)  # number of gradient steps for training policy (per iter in n_iter)
    parser.add_argument('--n_iter', '-n', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=1000)  # training data collected (in the env) during each iteration
    parser.add_argument('--eval_batch_size', type=int,
                        default=10000)  # eval data collected (in the env) for logging metrics
    parser.add_argument('--train_batch_size', type=int,
                        default=100)  # number of sampled data points to be used per gradient/train step

    parser.add_argument('--n_layers', type=int, default=5)  # depth, of policy to be learned
    parser.add_argument('--size', type=int, default=64)  # width of each layer, of policy to be learned
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)  # LR for supervised learning
    parser.add_argument('--learning_rate_step', '-lr_step', type=float, default=1000)  # LR for supervised learning
    parser.add_argument('--learning_rate_gamma', '-lr_gamma', type=float, default=0.7)  # LR for supervised learning

    parser.add_argument('--video_log_freq', type=int, default=5)
    parser.add_argument('--scalar_log_freq', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', type=int, default=0)
    parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)
    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    # convert args to dictionary
    params = vars(args)

    ###################
    ### RUN EXPERT POLICY
    ###################
    if args.problem_num == 11:
        run_expert_policy(params)
        sys.exit()

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    if args.do_dagger:
        logdir_prefix = 'q2_'
        assert args.n_iter>1, ('DAGGER needs more than 1 iteration (n_iter>1) of training, to iteratively query the expert and train (after 1st warmstarting from behavior cloning).')
    else:
        logdir_prefix = 'q1_'
        assert args.n_iter==1, ('Vanilla behavior cloning collects expert data just once (n_iter=1)')

    ## directory for logging
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = logdir_prefix + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)


    ###################
    ### RUN TRAINING
    ###################

    trainer = BC_Trainer(params)
    trainer.run_training_loop()

if __name__ == "__main__":
    main()
