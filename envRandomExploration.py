import random
import torch
import gym
import numpy as np
from qbn import QuantisedBottleneckNetwork
import tools as tl

# Defines a policy for an agent to choose actions at each timestep. Initially random
def choose_action():
    NUM_ACTIONS = 5
    action_probs = np.full(NUM_ACTIONS, 1 / NUM_ACTIONS)
    action = random.choices(range(NUM_ACTIONS), weights=action_probs, k=1)[0]
    return action

def extractShortestSuccessfulTraces(trace_lengths, states, events, no_of_traces_to_extract):
    sorted_trace_lengths = sorted(trace_lengths)
    extracted_state_seqs = []
    extracted_events = []
    for i in range(no_of_traces_to_extract):
        (_, index) = sorted_trace_lengths[i]
        extracted_state_seqs.append(states[index])
        extracted_events.append(events[index])

    return extracted_state_seqs, extracted_events

def runAgentWithPolicy(env, num_episodes):
    episode_durations = []
    states_traversed = []
    events_per_episode = []
    for ep in range(num_episodes):
        print("Episode {} in progress".format(ep + 1))
        
        initial_state = env.reset()
        state = torch.tensor(initial_state).float()

        states = []
        states.append(state)

        events_observed = []

        done, terminated, t = False, False, 1

        while not (done or terminated):
            action = choose_action()

            next_state, _, done, observations = env.step(action)

            next_state = torch.tensor(next_state).reshape(-1).float()
            state = next_state
            states.append(state)

            # if (observations != set()): 
            events_observed.append((observations, t))

            if done:
                print("I reached the goal state in {} steps".format(t + 1))
                episode_durations.append((t, ep))
            
            t += 1

        # print(events_observed)
        states_traversed.append(states)
        events_per_episode.append(events_observed)

    return episode_durations, states_traversed, events_per_episode


if __name__ == '__main__':
    env = gym.make("gym_subgoal_automata:WaterWorldRedGreen-v0", params={"generation": "random", "environment_seed": 0})

    NUM_SUCC_TRACES = 1
    NUM_EPISODES = 10

    # Generate successful traces for the task
    episode_durations, states_traversed, episode_events = runAgentWithPolicy(env, NUM_EPISODES)
    print("Episode durations")
    print(episode_durations)

    # Get the shortest traces- they are the most relevant
    relevant_state_seqs, relevant_events = extractShortestSuccessfulTraces(episode_durations, states_traversed, episode_events, NUM_SUCC_TRACES)

    input_vec_dim = 52

    # Hyperparameters
    quant_vector_dim = 80
    training_batch_size = 32
    test_batch_size = 32
    learning_rate = 1e-4
    weight_decay = 0
    epochs = 400
    training_set_size = 8192
    testing_set_size = 2048

    # Generate QBN
    qbn = QuantisedBottleneckNetwork(input_vec_dim, quant_vector_dim, training_batch_size, learning_rate, weight_decay, epochs, training_set_size)

    # Generate dataset for training
    print("Beginning to generate training data")
    obs_training_data = tl.generate_train_data_rand_init(env=env, dataset_size=training_set_size)
    print("Finished generating training data")

    # Train QBN
    qbn = tl.trainQBN(qbn, obs_training_data)
    print("Finished training the QBN")

    # Encoded every state in every sequence with the QBN
    encoded_seqs = []
    for state_seq in relevant_state_seqs:
        encoded_state_seq = []
        for state in state_seq:
            encoded_state = qbn.encode(state)
            encoded_state_seq.append(encoded_state)
        
        encoded_seqs.append(encoded_state_seq)

    # TODO: Extract labels from latent vectors that are common amongst the different 

    # TODO: Compare extracted labels to events that have been stored

