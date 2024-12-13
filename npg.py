import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel

LEVELS = 10

class NPG:
    def __init__(self, state_dim, action_dim, lr=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_net = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

    def encode_state(self, text, level, max_len=256):
        """
        Encode text into a fixed size state vector w/ BERT
        """
        inputs = self.bert_tokenizer(text, return_tensors='pt', max_length=max_len, truncation=True, padding='max_length')
        with torch.no_grad():
            embedding = self.bert_model(**inputs).pooler_output.squeeze(0).numpy()

        # One-hot encoding of level
        level_one_hot = np.zeros(LEVELS) 
        level_one_hot[level] = 1

        return np.concatenate([embedding, level_one_hot])

    def select_action(self, state):
        """
        Select action based on policy network
        """
        state = torch.FloatTensor(state, dtype=torch.float32)
        action_probs = self.policy_net(state)
        action = torch.multinomial(action_probs, 1).item()

        return action

    def update(self, trajectories, rewards, gamma=0.99):
        """
        Update policy network with trajectories and rewards
        """
        discounted_rewards = []
        for r in rewards:
            discounted_rewards.append(


