import gym
import sys
sys.path.insert(1, "/home/rahilshah/Documents/Year4/FYP/AEExperiment/AEExperiment")
from waterWorld.clustering.clusteringTraining import get_random_succ_traces, run_agent, train_clustering
from waterWorld.experiments.clustering.clusteringEval import affect_of_autoencoder, get_test_trace, user_playing_with_env, vary_no_of_eps, vary_no_of_succ_traces

if __name__ == "__main__":
    use_velocities = False
    env_with_static_balls = gym.make(
        "gym_subgoal_automata:WaterWorldRedGreen-v0",
        params={"generation": "random", "use_velocities": use_velocities, "environment_seed": 0},
    )

    fixed_start_env_with_static_balls = gym.make(
        "gym_subgoal_automata:WaterWorldRedGreen-v0",
        params={"generation": "random", "use_velocities": use_velocities, "environment_seed": 0, "random_restart": True},
    )

    fixed_start_red_env_with_static_balls = gym.make(
        "gym_subgoal_automata:WaterWorldRed-v0",
        params={"generation": "random", "use_velocities": use_velocities, "environment_seed": 0, "random_restart": False},
    )

    num_succ_traces_arr = [25, 50, 75, 100, 200, 300, 400]
    num_eps = 500
    ep_durs, test_trace, test_event_labels, actions = run_agent(env_with_static_balls, num_eps)
    vary_no_of_succ_traces(env_with_static_balls, num_succ_traces_arr, use_velocities, ep_durs, test_trace, test_event_labels, actions)

    precision_sum = 0
    recall_sum = 0
    for i in range(30):
        num_succ_traces = 50
        num_eps_arr = [500]
        precision, recall = vary_no_of_eps(fixed_start_env_with_static_balls, num_eps_arr, num_succ_traces, use_velocities)
        precision_sum += precision
        recall_sum += recall
    
    print(precision_sum / 30)
    print(recall_sum / 30)