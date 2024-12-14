
import gym
import numpy as np
import utils
from call_menu_env import CallMenuEnv
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    menu = {
        "number": 1,
        "text": "Welcome to our service.",
        "is_target": False,
        "children": [
            {
                "number": 1,
                "text": "For customer service representative, press 1.",
                "is_target": False,
                "children": [
                    {
                        "number": 1,
                        "text": "For technical support, press 1.",
                        "is_target": False,
                        "children": [
                            {
                                "number": 1,
                                "text": "For software related issues, press 1.",
                                "is_target": False,
                                "children": []
                            },
                            {
                                "number": 2,
                                "text": "For hardware related issues, press 2.",
                                "is_target": False,
                                "children": []
                            },
                            {
                                "number": 3,
                                "text": "To speak directly with a technical support representative, press 3.",
                                "is_target": True,
                                "children": []
                            }
                        ]
                    }
                ]
            }
        ]
    }

    np.random.seed(1234)
    theta, episode_rewards = train(menu, N=10, T=5, delta=1e-2)
    plt.plot(episode_rewards)
    plt.title("Avg rewards per timestep")
    plt.xlabel("Timestep")
    plt.ylabel("Avg rewards")
    plt.show()

