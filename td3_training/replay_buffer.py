import random
from collections import deque
import numpy as np

class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.0001, eps=1e-6, random_seed=1234):
        self.buffer_size = int(buffer_size)
        self.count = 0
        self.buffer = deque(maxlen=buffer_size)  # Automatically manage max size
        self.priorities = deque(maxlen=buffer_size)  # Priorities with maxlen
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.eps = eps
        random.seed(random_seed)

    def add(self, s0, a0, r0, t0, s2_0, s1, a1, r1, t1, s2_1):
        experience = (s0, a0, r0, t0, s2_0, s1, a1, r1, t1, s2_1)
        max_priority = max(self.priorities, default=1.0)  # Default max priority
        self.buffer.append(experience)
        self.priorities.append(float(max_priority))  # Cast to float to ensure scalar value
        self.count = len(self.buffer)  # Update count

    def update_priorities(self, indices, td_errors):
        td_errors = np.abs(td_errors)  # Ensure positive TD errors
        max_td_error = td_errors.max() + self.eps  # Add epsilon to avoid division by zero
        normalized_td_errors = td_errors / max_td_error  # Normalize TD errors to [0, 1]

        for idx, td_error in zip(indices, normalized_td_errors):
            self.priorities[idx] = float(td_error + self.eps)  # Add epsilon to prevent zero priority

    def sample_batch(self, batch_size):
        priorities_array = np.array(list(self.priorities), dtype=np.float32)
        priorities_array = priorities_array / (priorities_array.sum() + self.eps)  # Normalize priorities

        # Compute sampling probabilities
        scaled_priorities = priorities_array ** self.alpha
        sampling_probabilities = scaled_priorities / scaled_priorities.sum()

        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=sampling_probabilities)
        batch = [self.buffer[idx] for idx in indices]

        # Importance-sampling weights
        weights = (len(self.buffer) * sampling_probabilities[indices]) ** -self.beta
        weights /= max(weights)  # Normalize weights

        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        # Split the batch into components
        s0_batch = np.array([_[0] for _ in batch])
        a0_batch = np.array([_[1] for _ in batch])
        r0_batch = np.array([_[2] for _ in batch]).reshape(-1, 1)
        t0_batch = np.array([_[3] for _ in batch]).reshape(-1, 1)
        s2_0_batch = np.array([_[4] for _ in batch])

        s1_batch = np.array([_[5] for _ in batch])
        a1_batch = np.array([_[6] for _ in batch])
        r1_batch = np.array([_[7] for _ in batch]).reshape(-1, 1)
        t1_batch = np.array([_[8] for _ in batch]).reshape(-1, 1)
        s2_1_batch = np.array([_[9] for _ in batch])

        # Return 10 components, weights, and indices
        return (s0_batch, a0_batch, r0_batch, t0_batch, s2_0_batch,
                s1_batch, a1_batch, r1_batch, t1_batch, s2_1_batch), weights, indices


    def clear(self):
        self.buffer.clear()
        self.priorities.clear()
        self.count = 0
