import os, argparse, yaml, shutil

import numpy as np
import random
import tensorflow as tf

from rllab.misc.instrument import run_experiment_lite
import rllab.misc.logger as logger
from rllab.misc.ext import set_seed
### environments
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.normalized_env import normalize
### exploration strategies
from rllab.exploration_strategies.gaussian_strategy import GaussianStrategy
from sandbox.gkahn.rnn_critic.exploration_strategies.epsilon_greedy_strategy import EpsilonGreedyStrategy
### RNN critic
from sandbox.gkahn.rnn_critic.algos.rnn_critic import RNNCritic
from sandbox.gkahn.rnn_critic.policies.mlp_policy import RNNCriticMLPPolicy
from sandbox.gkahn.rnn_critic.policies.rnn_policy import RNNCriticRNNPolicy

### parameters loaded from yaml
params = dict()

def run_task(*_):
    # copy yaml for posterity
    shutil.copy(params['yaml_path'], os.path.join(logger.get_snapshot_dir(), os.path.basename(params['yaml_path'])))

    # set seed
    set_seed(params['seed'])

    from rllab.envs.gym_env import GymEnv
    from sandbox.gkahn.rnn_critic.envs.point_env import PointEnv
    from sandbox.gkahn.rnn_critic.envs.chain_env import ChainEnv
    env = TfEnv(normalize(eval(params['alg'].pop('env'))))

    #####################
    ### Create policy ###
    #####################

    policy_type = params['policy']['type']
    get_action_type = params['get_action']['type']

    if policy_type == 'mlp':
        PolicyClass = RNNCriticMLPPolicy
    elif policy_type == 'rnn':
        PolicyClass = RNNCriticRNNPolicy
    else:
        raise Exception('Policy {0} not valid'.format(policy_type))

    policy = PolicyClass(
        env_spec=env.spec,
        get_action_params=params['get_action'][get_action_type],
        **params['policy'][policy_type],
        **params['policy']
    )

    ###################################
    ### Create exploration strategy ###
    ###################################

    es_params = params['alg'].pop('exploration_strategy')
    es_type = es_params['type']
    if es_type == 'gaussian':
        ESClass = GaussianStrategy
    elif es_type == 'epsilon_greedy':
        ESClass = EpsilonGreedyStrategy
    else:
        raise Exception('Exploration strategy {0} not valid'.format(es_type))

    exploration_strategy = ESClass(env_spec=env.spec, **es_params[es_type])

    ########################
    ### Create algorithm ###
    ########################

    algo = RNNCritic(
        env=env,
        policy=policy,
        exploration_strategy=exploration_strategy,
        max_path_length=env.horizon,
        **params['alg']
    )
    algo.train()


def main():
    run_experiment_lite(
        run_task,
        snapshot_mode="all",
        exp_name=params['exp_name'],
        exp_prefix=params['exp_prefix']
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('yaml', type=str)
    args = parser.parse_args()

    assert(os.path.exists(args.yaml))
    with open(args.yaml, 'r') as f:
        params.update(yaml.load(f))
    params['yaml_path'] = args.yaml

    main()