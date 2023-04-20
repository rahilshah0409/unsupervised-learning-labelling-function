import gym
from waterWorld.experiments.clustering.clusteringEval import affect_of_autoencoder, vary_no_of_eps, vary_no_of_succ_traces

if __name__ == "__main__":
    env_with_static_balls = gym.make(
        "gym_subgoal_automata:WaterWorldRedGreen-v0",
        params={"generation": "random", "use_velocities": False, "environment_seed": 0},
    )

    fixed_start_env_with_static_balls = gym.make(
        "gym_subgoal_automata:WaterWorldRedGreen-v0",
        params={"generation": "random", "use_velocities": False, "environment_seed": 0, "random_restart": False},
    )

    num_succ_traces_arr = []
    num_eps = 50
    vary_no_of_succ_traces(env_with_static_balls, num_succ_traces_arr, num_eps)

    num_succ_traces = 50
    num_eps_arr = []
    vary_no_of_eps(env_with_static_balls, num_eps_arr, num_succ_traces)

    num_succ_traces = 50
    num_eps = 500
    affect_of_autoencoder(env_with_static_balls, num_succ_traces, num_eps)