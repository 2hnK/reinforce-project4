from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.connectors.env_to_module import FlattenObservations
from pprint import pprint

# Configure the algorithm.
config = (
    PPOConfig()
    .environment("HalfCheetah-v5")
    .training(
        # Following the paper.
        # ray/rllib/tuned_examples/ppo/benchmark_ppo_mujoco.py
        lambda_=0.95,
        lr=0.0003,
        num_epochs=3,
        train_batch_size=32 * 512,
        minibatch_size=4096,
        vf_loss_coeff=0.01,
        model={
            "fcnet_hiddens": [64, 64],
            "fcnet_activation": "tanh",
            "vf_share_layers": False,
        },
    )
    .learners(
        # num_learners=0; 로컬learner 사용
        num_learners=0,
        num_gpus_per_learner=1
    )
    .debugging(seed=0)
    .env_runners(
        num_env_runners=1,
        num_envs_per_env_runner=1,
        num_cpus_per_env_runner=1,
    )
    .evaluation(evaluation_num_env_runners=1,
                evaluation_interval=0,
                evaluation_duration=5)
)

# Build the algorithm.
algo = config.build_algo()

for i in range(5):
    res = algo.train()
    print(pprint(res))

# Release the algo's resources (remote actors, like EnvRunners and Learners).
algo.stop()
