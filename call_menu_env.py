import gym as gym
from gym import spaces
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

class CallMenuEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, menu, max_children=10, model_name="bert-base-uncased", render_mode=None, device="cpu"):
        super().__init__()
        
        self.menu = menu
        self.current_node = None
        self.max_children = max_children
        self.render_mode = render_mode
        
        # Load BERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device
        
        # BERT-base-uncased embeddings are 768-d by default
        embedding_dim = 768
        
        # Observation space:
        # "children_count": shape=(1,)
        # "text_embedding": shape=(768,)
        # For embeddings, allow any float values: use large bounds or np.inf.
        self.observation_space = spaces.Dict({
            "children_count": spaces.Box(low=0, high=self.max_children, shape=(1,), dtype=np.int32),
            "text_embedding": spaces.Box(low=-np.inf, high=np.inf, shape=(embedding_dim,), dtype=np.float32)
        })
        
        # Action space is discrete with max_children possible actions (will be masked later if needed)
        self.action_space = spaces.Discrete(self.max_children)
        self.action_space.n = self.max_children
        
        self.reset()

    def _find_target_node(self, node):
        return node.get("is_target", False)
    
    def _get_children(self, node):
        return node.get("children", [])
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_node = self.menu
        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def _get_observation(self):
        children = self._get_children(self.current_node)
        children_count = np.array([len(children)], dtype=np.int32)
        
        text_embedding = self._compute_text_embedding(self.current_node["text"])
        
        return {
            "children_count": children_count,
            "text_embedding": text_embedding
        }

    def _get_info(self):
        """
        SOURCE: https://datascience.stackexchange.com/questions/61618/valid-actions-in-openai-gym
        Gym does not support dynamic action spaces, so we need to provide a mask of valid actions.
        """

        children = self._get_children(self.current_node)
        action_mask = np.zeros(self.max_children, dtype=np.int32)
        action_mask[:len(children)] = 1
        return {
            "text": self.current_node["text"],
            "action_mask": action_mask
        }
    
    def _compute_text_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        cls_embedding = outputs.last_hidden_state[:,0,:].squeeze(0)
        
        cls_embedding = cls_embedding.cpu().numpy().astype(np.float32)
        return cls_embedding

    def step(self, action):
        children = self._get_children(self.current_node)

        if action < 0 or action >= len(children):
            # Invalid action so penalize and end episode
            reward = -1.0
            done = True
            truncated = False
            observation = self._get_observation()
            info = self._get_info()
            return observation, reward, done, truncated, info
        
        self.current_node = children[action]

        # Check if terminal
        done = False
        truncated = False
        if len(self._get_children(self.current_node)) == 0 or self._find_target_node(self.current_node):
            done = True

        reward = 100.0 if self._find_target_node(self.current_node) else -0.1 # TODO: Maybe try different reward function?
        observation = self._get_observation()
        info = self._get_info()
        return observation, reward, done, truncated, info
    
    def render(self):
        if self.render_mode == "human":
            print("Current Node: ", self.current_node["text"])
            children = self._get_children(self.current_node)
            print("Children:")
            for i, c in enumerate(children):
                print(f" {i}: {c['text']}")

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

    env = CallMenuEnv(menu, max_children=10)
    obs, info = env.reset()
    print("Initial Observation:", obs)
    print("Initial Info:", info)
    
    done = False
    while not done:
        action_mask = info["action_mask"]
        valid_actions = np.where(action_mask == 1)[0]
        if len(valid_actions) == 0:
            break
        action = np.random.choice(valid_actions)
        obs, reward, done, truncated, info = env.step(action)
        print("Step Obs:", obs)
        print("Reward:", reward)
        print("Done:", done)
