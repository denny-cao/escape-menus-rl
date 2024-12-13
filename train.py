import numpy as np
import utils
import matplotlib.pyplot as plt


def sample(theta, env, N):
    """ samples N trajectories using the current policy

    :param theta: the model parameters (shape d x 1)
    :param env: the environment used to sample from
    :param N: number of trajectories to sample
    :return:
        trajectories_gradients: lists with sublists for the gradients for each trajectory rollout (should be a 2-D list)
        trajectories_rewards:  lists with sublists for the rewards for each trajectory rollout (should be a 2-D list)

    Note: the maximum trajectory length is 200 steps
    """

    MAX_STEPS = 200
    total_rewards = []
    total_grads = []

    for _ in range(N):
        state = env.reset()
        done = False

        trajectory_rewards = []
        trajectory_grads = []

        for _ in range(MAX_STEPS):
            # Sample an action from the policy
            phis = utils.extract_features(state, env.action_space.n)
            prob_dist = utils.compute_action_distribution(theta, phis)

            action = np.random.choice(env.action_space.n, p=prob_dist.flatten())

            # Compute the gradient of the log probability of the action
            grad = utils.compute_log_softmax_grad(theta, phis, action)

            # Take a step in the environment
            next_state, reward, done, _ = env.step(action)

            trajectory_rewards.append(reward)
            trajectory_grads.append(grad)

            if done:
                break

            state = next_state

        total_rewards.append(trajectory_rewards)
        total_grads.append(trajectory_grads)

    return total_grads, total_rewards

def train(N, T, delta, lamb=1e-3):
    """

    :param N: number of trajectories to sample in each time step
    :param T: number of iterations to train the model
    :param delta: trust region size
    :param lamb: lambda for fisher matrix computation
    :return:
        theta: the trained model parameters
        avg_episodes_rewards: list of average rewards for each time step
    """
    theta = np.random.rand(100,1)
    env = gym.make('CartPole-v0')
    env.seed(12345)

    avg_episode_rewards = []

    for t in range(T):
        # Collect samples by rolling out N trajectories with the current policy using the sample function
        trajectories_gradients, trajectories_rewards = sample(theta, env, N)

        # Compute the average reward of the trajectories
        avg_reward = np.mean([np.sum(trajectory) for trajectory in trajectories_rewards])
        avg_episode_rewards.append(avg_reward)

        # Compute the (estimated) Fisher Information matrix using the gradients from the previous steps. Use the
        # default value of lambda.
        fisher = utils.compute_fisher_matrix(trajectories_gradients, lamb)

        # Compute step size for this NPG step
        v_grad = utils.compute_value_gradient(trajectories_gradients, trajectories_rewards)
        eta = utils.compute_eta(delta, fisher, v_grad)

        # Update model parameters theta by taking an NPG step
        theta += eta * np.linalg.inv(fisher) @ v_grad

    return theta, avg_episode_rewards


if __name__ == '__main__':
    np.random.seed(1234)
    theta, episode_rewards = train(N=100, T=20, delta=1e-2)
    plt.plot(episode_rewards)
    plt.title("avg rewards per timestep")
    plt.xlabel("timestep")
    plt.ylabel("avg rewards")
    plt.show()
