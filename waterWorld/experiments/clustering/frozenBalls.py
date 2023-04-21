import gym
import sys
sys.path.insert(1, "/home/rahilshah/Documents/Year4/FYP/AEExperiment/AEExperiment")
from waterWorld.clustering.clusteringTraining import run_agent
from waterWorld.experiments.clustering.clusteringEval import affect_of_autoencoder, vary_no_of_eps, vary_no_of_succ_traces

if __name__ == "__main__":
    use_velocities = False
    env_with_static_balls = gym.make(
        "gym_subgoal_automata:WaterWorldRedGreen-v0",
        params={"generation": "random", "use_velocities": use_velocities, "environment_seed": 0},
    )

    fixed_start_env_with_static_balls = gym.make(
        "gym_subgoal_automata:WaterWorldRedGreen-v0",
        params={"generation": "random", "use_velocities": use_velocities, "environment_seed": 0, "random_restart": False},
    )

    # num_succ_traces_arr = [25, 50, 75, 100, 200, 300, 400]
    # num_eps = 500
    # ep_durs, test_trace, test_event_labels, actions = run_agent(env_with_static_balls, num_eps)
    # vary_no_of_succ_traces(env_with_static_balls, num_succ_traces_arr, use_velocities, ep_durs, test_trace, test_event_labels, actions)

    # num_succ_traces = 50
    # num_eps_arr = [100, 200, 300, 400, 500, 600]
    # vary_no_of_eps(env_with_static_balls, num_eps_arr, num_succ_traces, use_velocities)

    num_succ_traces = 50
    num_eps = 500
    affect_of_autoencoder(env_with_static_balls, num_succ_traces, num_eps, use_velocities)