import gym
from clusteringEval import evaluate_cluster_label_prediction, get_test_trace
import sys
sys.path.insert(1, "/home/rahilshah/Documents/Year4/FYP/AEExperiment/AEExperiment")
from waterWorld.clustering.clusteringTraining import get_random_succ_traces, train_clustering, encode_state_seq

if __name__ == "__main__":
    simple_fixed_start_env = gym.make(
        "gym_subgoal_automata:WaterWorldRed-v0",
        params={"generation": "random", "random_restart": False, "environment_seed": 0}
    )
    
    normal_env = gym.make(
        "gym_subgoal_automata:WaterWorldRedGreen-v0",
        params={"generation": "random", "environment_seed": 0},
    )

    fixed_start_env = gym.make(
        "gym_subgoal_automata:WaterWorldRedGreen-v0",
        params={"generation": "random", "random_restart": False, "environment_seed": 0},
    )

    _, state_seqs, _, _ = get_random_succ_traces(normal_env, num_succ_traces=2, num_episodes=10)
    encode_states = True
    kmeans_obj, qbn = train_clustering(state_seqs, 4, encode_states=encode_states)