import numpy as np
import gym
from sklearn.kernel_approximation import RBFSampler
import matplotlib.pyplot as plt
from tqdm import trange

# Utility class for feature extraction and action computation
class FeatureExtractor:
    def __init__(self, gamma=1, random_state=12345):
        self.rbf_feature = RBFSampler(gamma=gamma, random_state=random_state)

    @staticmethod
    def flatten_observation(obs):
        # SOURCE: ChatGPT. Custom environment returns dictionary, and we cannot flatten a dict.
        return np.concatenate([obs["children_count"], obs["text_embedding"]])

    def extract_features(self, state, num_actions):
        state = state.reshape(1, -1)
        state = np.repeat(state, num_actions, axis=0)
        actions = np.arange(num_actions).reshape(-1, 1)
        state_action = np.concatenate([state, actions], axis=-1)
        features = self.rbf_feature.fit_transform(state_action)
        return features.T

class Policy:
    def __init__(self, feature_dim, action_dim):
        self.theta = np.random.rand(feature_dim, 1)
        self.action_dim = action_dim

    def compute_action_distribution(self, phis):
        logits = self.theta.T @ phis
        return self._softmax(logits, axis=1)

    def compute_log_softmax_grad(self, phis, action_idx):
        probs = self.compute_action_distribution(phis)
        grad = phis[:, action_idx] - np.sum(phis * probs, axis=1)
        return grad.reshape(-1, 1)

    @staticmethod
    def _softmax(logits, axis):
        exp_logits = np.exp(logits - np.max(logits, axis=axis, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)

class FisherInformation:
    @staticmethod
    def compute_fisher_matrix(grads, lamb=1e-3):
        d = grads[0][0].shape[0]
        N = len(grads)
        fisher_hat = np.zeros((d, d))
        for trajectory in grads:
            H = len(trajectory)
            for grad in trajectory:
                fisher_hat += (grad @ grad.T) / H
        fisher_hat /= N
        fisher_hat += lamb * np.eye(d)
        return fisher_hat

class ValueGradient:
    @staticmethod
    def compute_value_gradient(grads, rewards):
        N = len(grads)
        b = np.mean([np.sum(r) for r in rewards])
        v_grad_hat = np.zeros(grads[0][0].shape)
        for i in range(N):
            H = len(grads[i])
            for j in range(H):
                discounted_reward = np.sum(rewards[i][j:])
                v_grad_hat += (discounted_reward - b) * grads[i][j] / H
        return v_grad_hat / N

    @staticmethod
    def compute_eta(delta, fisher, v_grad):
        return np.sqrt(delta / (v_grad.T @ np.linalg.inv(fisher) @ v_grad + delta))

class NPGTrainer:
    def __init__(self, env, feature_extractor, policy, fisher, value_gradient):
        self.env = env
        self.feature_extractor = feature_extractor
        self.policy = policy
        self.fisher = fisher
        self.value_gradient = value_gradient

    def sample_trajectories(self, N, max_steps):
        trajectories_rewards = []
        trajectories_grads = []

        for _ in range(N):
            obs, info = self.env.reset()
            done = False
            trajectory_rewards = []
            trajectory_grads = []

            state = self.feature_extractor.flatten_observation(obs)

            for _ in range(max_steps):
                state_features = self.feature_extractor.extract_features(
                    state, self.env.action_space.n
                )

                action_mask = info["action_mask"]
                valid_actions = np.where(action_mask == 1)[0]
                if len(valid_actions) == 0:
                    break

                prob_dist = self.policy.compute_action_distribution(state_features)
                # Get the probability distribution for valid actions
                prob_dist = prob_dist.flatten()
                prob_dist = prob_dist * action_mask
                prob_dist = prob_dist / np.sum(prob_dist)
                action = np.random.choice(np.arange(self.env.action_space.n), p=prob_dist)

                grad = self.policy.compute_log_softmax_grad(state_features, action)

                next_obs, reward, done, truncated, info = self.env.step(action)
                next_state = self.feature_extractor.flatten_observation(next_obs)

                trajectory_rewards.append(reward)
                trajectory_grads.append(grad)

                if done:
                    break

                state = next_state

            trajectories_rewards.append(trajectory_rewards)
            trajectories_grads.append(trajectory_grads)

        return trajectories_grads, trajectories_rewards

    def train(self, N, T, delta, lamb=1e-3, max_steps=200):
        avg_episode_rewards = []

        for _ in trange(T):
            trajectories_grads, trajectories_rewards = self.sample_trajectories(N, max_steps)

            avg_reward = np.mean([np.sum(trajectory) for trajectory in trajectories_rewards])
            avg_episode_rewards.append(avg_reward)

            fisher_matrix = self.fisher.compute_fisher_matrix(trajectories_grads, lamb)
            value_grad = self.value_gradient.compute_value_gradient(trajectories_grads, trajectories_rewards)
            eta = self.value_gradient.compute_eta(delta, fisher_matrix, value_grad)

            self.policy.theta += eta * np.linalg.inv(fisher_matrix) @ value_grad

        return avg_episode_rewards

if __name__ == "__main__":
    # Initialize environment, feature extractor, and components
    from call_menu_env import CallMenuEnv  # Assuming the environment is saved in this file
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
                                "is_target": False,
                                "children": []
                            }
                        ]
                    },
                    {
                        "number": 2,
                        "text": "For billing inquiries, press 2.",
                        "is_target": False,
                        "children": [
                            {
                                "number": 1,
                                "text": "For general billing questions, press 1.",
                                "is_target": False,
                                "children": []
                            },
                            {
                                "number": 2,
                                "text": "For specific invoice queries, press 2.",
                                "is_target": False,
                                "children": []
                            },
                            {
                                "number": 3,
                                "text": "For payment methods, press 3.",
                                "is_target": False,
                                "children": []
                            }
                        ]
                    },
                    {
                        "number": 3,
                        "text": "For questions about our products, press 3.",
                        "is_target": False,
                        "children": [
                            {
                                "number": 1,
                                "text": "For information about product warranties, press 1.",
                                "is_target": False,
                                "children": []
                            },
                            {
                                "number": 2,
                                "text": "To speak with a product specialist, press 2.",
                                "is_target": False,
                                "children": []
                            },
                            {
                                "number": 3,
                                "text": "For information about product returns, press 3.",
                                "is_target": False,
                                "children": []
                            }
                        ]
                    }
                ]
            }
        ]
    }
    env = CallMenuEnv(menu)

    feature_extractor = FeatureExtractor()
    policy = Policy(feature_dim=100, action_dim=env.action_space.n)
    fisher = FisherInformation()
    value_gradient = ValueGradient()
    trainer = NPGTrainer(env, feature_extractor, policy, fisher, value_gradient)

    avg_rewards = trainer.train(N=20, T=20, delta=1e-2)

    plt.plot(avg_rewards)
    plt.title("Average Rewards per Timestep")
    plt.xlabel("Timestep")
    plt.ylabel("Average Rewards")
    plt.show()
