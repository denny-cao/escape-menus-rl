import gym
import numpy as np
import utils
from call_menu_env import CallMenuEnv
import matplotlib.pyplot as plt
import os
import json


def sample(theta, env, N):
    """ Samples N trajectories using the current policy

    :param theta: the model parameters (shape d x 1)
    :param env: the environment used to sample from
    :param N: number of trajectories to sample
    :return:
        trajectories_gradients: lists with sublists for the gradients for each trajectory rollout (should be a 2-D list)
        trajectories_rewards:  lists with sublists for the rewards for each trajectory rollout (should be a 2-D list)
    """
    MAX_STEPS = 200
    total_rewards = []
    total_grads = []

    for _ in range(N):
        state, info = env.reset()
        done = False

        trajectory_rewards = []
        trajectory_grads = []

        for _ in range(MAX_STEPS):
            # Extract features using embeddings from the environment
            phis = utils.extract_features(np.array(info["children_embeddings"]).T, env.action_space.n)
            prob_dist = utils.compute_action_distribution(theta, phis)

            # Apply action mask to ensure only valid actions are chosen
            action_mask = info["action_mask"]
            prob_dist *= action_mask
            prob_dist /= np.sum(prob_dist)

            # Sample an action from the policy
            action = np.random.choice(env.action_space.n, p=prob_dist.flatten())

            # Compute the gradient of the log probability of the action
            grad = utils.compute_log_softmax_grad(theta, phis, action)

            # Take a step in the environment
            _, reward, done, _, info = env.step(action)

            trajectory_rewards.append(reward)
            trajectory_grads.append(grad)

            if done:
                break

        total_rewards.append(trajectory_rewards)
        total_grads.append(trajectory_grads)

    return total_grads, total_rewards


def train(menu, N, T, delta, lamb=1e-3):
    """
    Trains the NPG model on the given menu environment

    :param menu: menu structure for the CallMenuEnv
    :param N: number of trajectories to sample in each time step
    :param T: number of iterations to train the model
    :param delta: trust region size
    :param lamb: lambda for fisher matrix computation
    :return:
        theta: the trained model parameters
        avg_episodes_rewards: list of average rewards for each time step
    """
    theta = np.random.rand(100, 1)
    env = CallMenuEnv(menu, max_children=10)

    avg_episode_rewards = []

    for t in range(T):
        # Collect samples by rolling out N trajectories with the current policy
        trajectories_gradients, trajectories_rewards = sample(theta, env, N)

        # Compute the average reward of the trajectories
        avg_reward = np.mean([np.sum(trajectory) for trajectory in trajectories_rewards])
        avg_episode_rewards.append(avg_reward)

        # Compute the (estimated) Fisher Information matrix using the gradients from the previous steps
        fisher = utils.compute_fisher_matrix(trajectories_gradients, lamb)

        # Compute step size for this NPG step
        v_grad = utils.compute_value_gradient(trajectories_gradients, trajectories_rewards)
        eta = utils.compute_eta(delta, fisher, v_grad)

        # Update model parameters theta by taking an NPG step
        theta += eta * np.linalg.inv(fisher) @ v_grad

    return theta, avg_episode_rewards


def load_menus_from_directory(directory_path):
    """Load all menu JSON files from a given directory."""
    menus = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            with open(os.path.join(directory_path, filename), 'r') as file:
                menus.append(json.load(file))
    return menus

def train_on_multiple_menus(directory_path, N, T, delta, lamb=1e-3):
    """
    Train the NPG model using multiple menus loaded from the given directory.

    :param directory_path: Path to the directory containing menu JSON files
    :param N: Number of trajectories to sample in each time step
    :param T: Number of iterations to train the model
    :param delta: Trust region size
    :param lamb: Lambda for Fisher matrix computation
    :return:
        theta: The trained model parameters
        avg_rewards: Dictionary with menu filenames as keys and average rewards list as values
    """
    menus = load_menus_from_directory(directory_path)
    avg_rewards = {}

    for idx, menu in enumerate(menus):
        print(f"Training on menu {idx + 1} / {len(menus)}")
        env = CallMenuEnv(menu, max_children=10)
        theta = np.random.rand(100, 1)

        avg_episode_rewards = []

        for t in range(T):
            # Collect samples by rolling out N trajectories with the current policy
            trajectories_gradients, trajectories_rewards = sample(theta, env, N)

            # Compute the average reward of the trajectories
            avg_reward = np.mean([np.sum(trajectory) for trajectory in trajectories_rewards])
            avg_episode_rewards.append(avg_reward)

            # Compute the (estimated) Fisher Information matrix using the gradients from the previous steps
            fisher = utils.compute_fisher_matrix(trajectories_gradients, lamb)

            # Compute step size for this NPG step
            v_grad = utils.compute_value_gradient(trajectories_gradients, trajectories_rewards)
            eta = utils.compute_eta(delta, fisher, v_grad)

            # Update model parameters theta by taking an NPG step
            theta += eta * np.linalg.inv(fisher) @ v_grad

        avg_rewards[f"menu_{idx + 1}"] = avg_episode_rewards

    return avg_rewards

if __name__ == "__main__":
    directory_path = "pr_50_br_3_dp_3"
    N = 5
    T = 2
    delta = 1e-2

    np.random.seed(1234)

    avg_rewards = train_on_multiple_menus(directory_path, N, T, delta)

    for menu_name, rewards in avg_rewards.items():
        plt.plot(rewards, label=menu_name)

    plt.title("Avg rewards per menu")
    plt.xlabel("Timestep")
    plt.ylabel("Avg rewards")
    plt.legend()
    plt.show()

