import gym
from waterWorld.clustering.clusteringTraining import get_random_succ_traces, train_clustering
from waterWorld.experiments.clustering.clusteringEval import evaluate_cluster_label_prediction, get_test_trace

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

    _, state_seqs, _, _ = get_random_succ_traces(normal_env, num_succ_traces=50, num_eps=500)
    kmeans_obj = train_clustering(state_seqs, 4, encode_states=True)
    ep_dur, test_trace, test_events = get_test_trace(normal_env, random_gen=True)
    cluster_labels = kmeans_obj.predict(test_trace)
    evaluate_cluster_label_prediction([cluster_labels], ["Normal environment"], test_events, ep_dur)