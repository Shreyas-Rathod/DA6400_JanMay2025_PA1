import gymnasium as gym
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
from datetime import datetime
from tqdm import tqdm

# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class DiscretizedWrapper(gym.ObservationWrapper):
    """
    A wrapper that discretizes continuous observation spaces
    """
    def __init__(self, env, n_bins=10):
        super().__init__(env)
        self.n_bins = n_bins
        
        # Get observation space bounds
        if isinstance(env.observation_space, gym.spaces.Box):
            self.low = env.observation_space.low
            self.high = env.observation_space.high
            
            # Create discrete observation space
            self.observation_space = gym.spaces.MultiDiscrete([n_bins] * env.observation_space.shape[0])
        else:
            self.observation_space = env.observation_space
    
    def observation(self, obs):
        if isinstance(self.env.observation_space, gym.spaces.Box):
            # Handle infinite values in observation
            obs_clipped = np.clip(obs, self.low, self.high)
            # Scale to [0, 1]
            scaled = (obs_clipped - self.low) / (self.high - self.low)
            # Handle NaN values that might still occur
            scaled = np.nan_to_num(scaled, nan=0.5)
            # Scale to [0, n_bins-1]
            scaled = np.clip(scaled, 0, 1) * (self.n_bins - 1)
            # Convert to integers
            return scaled.astype(int)
        return obs


class Agent:
    def __init__(self, state_space, action_space, learning_rate=0.1, discount_factor=0.99):
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # Initialize Q-table with proper dimensions
        if isinstance(state_space, gym.spaces.Dict):
            # For MiniGrid environments
            self.q_table = {}
        elif isinstance(state_space, gym.spaces.MultiDiscrete):
            # Create state dimensions from MultiDiscrete space
            state_dims = tuple(dim for dim in state_space.nvec)
            # Initialize Q-table with zeros
            self.q_table = np.zeros(state_dims + (action_space.n,))
        else:
            # For discrete state space
            self.q_table = np.zeros((state_space.n, action_space.n))
    
    def _dict_to_tuple(self, state):
        # Convert dictionary state to a hashable tuple
        if 'image' in state:
            img_tuple = tuple(state['image'].flatten())
            if 'direction' in state:
                return (img_tuple, state['direction'])
            return (img_tuple,)
        return tuple()
    
    def get_q_value(self, state, action):
        if isinstance(self.state_space, gym.spaces.Dict):
            state_key = self._dict_to_tuple(state)
            return self.q_table.get((state_key, action), 0.0)
        elif isinstance(self.state_space, gym.spaces.MultiDiscrete):
            # Ensure state values are within bounds
            state_bounded = np.minimum(state, self.state_space.nvec - 1)
            state_bounded = np.maximum(state_bounded, 0)
            state_tuple = tuple(state_bounded)
            
            # Ensure action is within bounds
            action_bounded = max(0, min(action, self.action_space.n - 1))
            
            return self.q_table[state_tuple + (action_bounded,)]
        
        # Ensure state and action are within bounds
        state_bounded = max(0, min(state, self.state_space.n - 1))
        action_bounded = max(0, min(action, self.action_space.n - 1))
        return self.q_table[state_bounded, action_bounded]

    def set_q_value(self, state, action, value):
        if isinstance(self.state_space, gym.spaces.Dict):
            state_key = self._dict_to_tuple(state)
            self.q_table[(state_key, action)] = value
        elif isinstance(self.state_space, gym.spaces.MultiDiscrete):
            # Ensure state values are within bounds
            state_bounded = np.minimum(state, self.state_space.nvec - 1)
            state_bounded = np.maximum(state_bounded, 0)
            state_tuple = tuple(state_bounded)
            
            # Ensure action is within bounds
            action_bounded = max(0, min(action, self.action_space.n - 1))
            
            self.q_table[state_tuple + (action_bounded,)] = value
        else:
            # Ensure state and action are within bounds
            state_bounded = max(0, min(state, self.state_space.n - 1))
            action_bounded = max(0, min(action, self.action_space.n - 1))
            self.q_table[state_bounded, action_bounded] = value


class SARSAAgent(Agent):
    def __init__(self, state_space, action_space, learning_rate=0.1, discount_factor=0.99, epsilon=0.1, epsilon_decay=0.999, epsilon_min=0.01):
        super().__init__(state_space, action_space, learning_rate, discount_factor)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
    
    def select_action(self, state):
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            if isinstance(self.state_space, gym.spaces.Dict):
                # For dictionary observation space
                q_values = np.array([self.get_q_value(state, a) for a in range(self.action_space.n)])
            elif isinstance(self.state_space, gym.spaces.MultiDiscrete):
                state_tuple = tuple(state)
                q_values = self.q_table[state_tuple]
            else:
                q_values = self.q_table[state]
                
            # Handle the case of equal q-values
            max_value = np.max(q_values)
            max_indices = np.where(q_values == max_value)[0]
            return np.random.choice(max_indices)
    
    def update(self, state, action, reward, next_state, next_action, done):
        current_q = self.get_q_value(state, action)
        
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * self.get_q_value(next_state, next_action)
        
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.set_q_value(state, action, new_q)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def reset_epsilon(self, epsilon):
        self.epsilon = epsilon


class QLearningAgent(Agent):
    def __init__(self, state_space, action_space, learning_rate=0.1, discount_factor=0.99, temperature=1.0, temp_decay=0.999, temp_min=0.1):
        super().__init__(state_space, action_space, learning_rate, discount_factor)
        self.temperature = temperature
        self.temp_decay = temp_decay
        self.temp_min = temp_min
    
    def select_action(self, state):
        # Softmax action selection
        if isinstance(self.state_space, gym.spaces.Dict):
            # For dictionary observation space
            q_values = np.array([self.get_q_value(state, a) for a in range(self.action_space.n)])
        elif isinstance(self.state_space, gym.spaces.MultiDiscrete):
            state_tuple = tuple(state)
            q_values = self.q_table[state_tuple]
        else:
            q_values = self.q_table[state]
        
        # Apply softmax with temperature (avoid overflow)
        q_values = q_values - np.max(q_values)  # For numerical stability
        exp_q = np.exp(q_values / max(self.temperature, 1e-8))
        probs = exp_q / np.sum(exp_q)
        
        # Handle NaN probabilities (can happen with very high temperature)
        if np.isnan(probs).any():
            return self.action_space.sample()
        
        # Choose action based on probabilities
        try:
            return np.random.choice(self.action_space.n, p=probs)
        except:
            # Fallback to random action if probabilities are invalid
            return self.action_space.sample()
    
    def update(self, state, action, reward, next_state, done):
        current_q = self.get_q_value(state, action)
        
        if done:
            target_q = reward
        else:
            if isinstance(self.state_space, gym.spaces.Dict):
                # For dictionary observation space
                next_q_values = np.array([self.get_q_value(next_state, a) for a in range(self.action_space.n)])
                target_q = reward + self.discount_factor * np.max(next_q_values)
            elif isinstance(self.state_space, gym.spaces.MultiDiscrete):
                state_tuple = tuple(next_state)
                target_q = reward + self.discount_factor * np.max(self.q_table[state_tuple])
            else:
                target_q = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.set_q_value(state, action, new_q)
        
        # Decay temperature
        if self.temperature > self.temp_min:
            self.temperature *= self.temp_decay
    
    def reset_temperature(self, temperature):
        self.temperature = temperature


def train_sarsa(env_name, n_episodes=1000, n_runs=5, hyperparams=None, render=False, bins=10):
    """
    Train SARSA agent on the given environment
    
    Args:
        env_name: Name of the Gym environment
        n_episodes: Number of episodes to train
        n_runs: Number of runs with different seeds
        hyperparams: Dictionary of hyperparameters
        render: Whether to render the environment
        bins: Number of bins for discretization
    
    Returns:
        DataFrame with training history
    """
    if hyperparams is None:
        hyperparams = {
            'learning_rate': 0.1,
            'discount_factor': 0.99,
            'epsilon': 1.0,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01
        }
    
    all_run_returns = []
    
    for run in range(n_runs):
        seed = run + 42  # Different seed for each run
        set_seed(seed)
        
        # Create environment with proper version
        env = gym.make(env_name)
        
        # Apply discretization wrapper for continuous observation spaces
        if isinstance(env.observation_space, gym.spaces.Box):
            env = DiscretizedWrapper(env, n_bins=bins)
        
        agent = SARSAAgent(
            env.observation_space,
            env.action_space,
            learning_rate=hyperparams['learning_rate'],
            discount_factor=hyperparams['discount_factor'],
            epsilon=hyperparams['epsilon'],
            epsilon_decay=hyperparams['epsilon_decay'],
            epsilon_min=hyperparams['epsilon_min']
        )
        
        episode_returns = []
        
        for episode in tqdm(range(n_episodes), desc=f"SARSA Run {run+1}/{n_runs}"):
            # Reset environment with seed
            state, _ = env.reset(seed=seed + episode)
            done = False
            truncated = False
            episode_return = 0
            
            # Select first action
            action = agent.select_action(state)
            
            while not (done or truncated):
                # Take action
                next_state, reward, done, truncated, _ = env.step(action)
                episode_return += reward
                
                # Select next action
                next_action = agent.select_action(next_state)
                
                # Update Q-values
                agent.update(state, action, reward, next_state, next_action, done or truncated)
                
                state = next_state
                action = next_action
            
            episode_returns.append(episode_return)
        
        all_run_returns.append(episode_returns)
        plot_single_run(episode_returns, "SARSA", env_name, run+1, hyperparams)
        env.close()
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_run_returns).T
    results_df.columns = [f'run_{i}' for i in range(n_runs)]
    results_df['episode'] = results_df.index
    results_df['mean'] = results_df.iloc[:, :n_runs].mean(axis=1)
    results_df['std'] = results_df.iloc[:, :n_runs].std(axis=1)
    
    return results_df

def train_qlearning(env_name, n_episodes=1000, n_runs=5, hyperparams=None, render=False, bins=10):
    """
    Train Q-Learning agent on the given environment
    
    Args:
        env_name: Name of the Gym environment
        n_episodes: Number of episodes to train
        n_runs: Number of runs with different seeds
        hyperparams: Dictionary of hyperparameters
        render: Whether to render the environment
        bins: Number of bins for discretization
    
    Returns:
        DataFrame with training history
    """
    if hyperparams is None:
        hyperparams = {
            'learning_rate': 0.1,
            'discount_factor': 0.99,
            'temperature': 1.0,
            'temp_decay': 0.995,
            'temp_min': 0.1
        }
    
    all_run_returns = []
    
    for run in range(n_runs):
        seed = run + 42  # Different seed for each run
        set_seed(seed)
        
        # Create environment with proper version
        env = gym.make(env_name)
        
        # Apply discretization wrapper for continuous observation spaces
        if isinstance(env.observation_space, gym.spaces.Box):
            env = DiscretizedWrapper(env, n_bins=bins)
        
        agent = QLearningAgent(
            env.observation_space,
            env.action_space,
            learning_rate=hyperparams['learning_rate'],
            discount_factor=hyperparams['discount_factor'],
            temperature=hyperparams['temperature'],
            temp_decay=hyperparams['temp_decay'],
            temp_min=hyperparams['temp_min']
        )
        
        episode_returns = []
        
        for episode in tqdm(range(n_episodes), desc=f"Q-Learning Run {run+1}/{n_runs}"):
            # Reset environment with seed
            state, _ = env.reset(seed=seed + episode)
            done = False
            truncated = False
            episode_return = 0
            
            while not (done or truncated):
                # Select action
                action = agent.select_action(state)
                
                # Take action
                next_state, reward, done, truncated, _ = env.step(action)
                episode_return += reward
                
                # Update Q-values
                agent.update(state, action, reward, next_state, done or truncated)
                
                state = next_state
            
            episode_returns.append(episode_return)
        
        all_run_returns.append(episode_returns)
        plot_single_run(episode_returns, "Q-Learning", env_name, run+1, hyperparams)
        env.close()
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_run_returns).T
    results_df.columns = [f'run_{i}' for i in range(n_runs)]
    results_df['episode'] = results_df.index
    results_df['mean'] = results_df.iloc[:, :n_runs].mean(axis=1)
    results_df['std'] = results_df.iloc[:, :n_runs].std(axis=1)
    
    return results_df


def plot_single_run(episode_returns, algorithm, env_name, run_number, hyperparams):
    """
    Plot the results of a single training run
    
    Args:
        episode_returns: List of episode returns
        algorithm: 'SARSA' or 'Q-Learning'
        env_name: Name of the environment
        run_number: Run number
        hyperparams: Hyperparameters used
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(episode_returns)), episode_returns)
    plt.title(f'{algorithm} Run {run_number} on {env_name}')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.grid(True)
    
    # Add hyperparameters as text
    param_text = "\n".join([f"{k}: {v}" for k, v in hyperparams.items()])
    plt.figtext(0.02, 0.02, f"Hyperparameters:\n{param_text}", fontsize=9)
    
    # Save plot
    os.makedirs(f"results/{env_name}", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"results/{env_name}/{algorithm}_run{run_number}_{timestamp}.png", dpi=300)
    
    # Display plot in notebook
    plt.show()
    plt.close()


def plot_results(sarsa_results, qlearning_results, env_name, hyperparams_sarsa, hyperparams_qlearning):
    """
    Plot the results of the training with all four types of plots
    
    Args:
        sarsa_results: DataFrame with SARSA results
        qlearning_results: DataFrame with Q-Learning results
        env_name: Name of the environment
        hyperparams_sarsa: Hyperparameters used for SARSA
        hyperparams_qlearning: Hyperparameters used for Q-Learning
    """
    os.makedirs(f"results/{env_name}", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Comparison plot (SARSA vs Q-Learning mean returns)
    plt.figure(figsize=(12, 7))
    plt.plot(sarsa_results['episode'], sarsa_results['mean'], label='SARSA Mean Return', color='blue')
    plt.plot(qlearning_results['episode'], qlearning_results['mean'], label='Q-Learning Mean Return', color='green')
    plt.fill_between(
        sarsa_results['episode'],
        sarsa_results['mean'] - sarsa_results['std'],
        sarsa_results['mean'] + sarsa_results['std'],
        alpha=0.2,
        color='blue'
    )
    plt.fill_between(
        qlearning_results['episode'],
        qlearning_results['mean'] - qlearning_results['std'],
        qlearning_results['mean'] + qlearning_results['std'],
        alpha=0.2,
        color='green'
    )
    plt.title(f'SARSA vs Q-Learning on {env_name}')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/{env_name}/comparison_{timestamp}.png", dpi=300)
    plt.show()
    plt.close()
    
    # 2. Episodic return vs episode number for both algorithms
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    
    # SARSA
    axs[0].plot(sarsa_results['episode'], sarsa_results['mean'], label='Mean Return', color='blue')
    axs[0].set_title(f'SARSA Episodic Return — {env_name}')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Return')
    axs[0].legend()
    axs[0].grid(True)
    
    # Q-Learning
    axs[1].plot(qlearning_results['episode'], qlearning_results['mean'], label='Mean Return', color='green')
    axs[1].set_title(f'Q-Learning Episodic Return — {env_name}')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Return')
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"results/{env_name}/episodic_returns_{timestamp}.png", dpi=300)
    plt.show()
    plt.close()
    
    # 3. Mean and variance across runs
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    
    # SARSA
    axs[0].plot(sarsa_results['episode'], sarsa_results['mean'], label='Mean Return', color='blue')
    axs[0].fill_between(
        sarsa_results['episode'],
        sarsa_results['mean'] - sarsa_results['std'],
        sarsa_results['mean'] + sarsa_results['std'],
        alpha=0.3,
        color='blue',
        label='Std Dev'
    )
    axs[0].set_title(f'SARSA Mean ± Std — {env_name}')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Return')
    axs[0].legend()
    axs[0].grid(True)
    
    # Q-Learning
    axs[1].plot(qlearning_results['episode'], qlearning_results['mean'], label='Mean Return', color='green')
    axs[1].fill_between(
        qlearning_results['episode'],
        qlearning_results['mean'] - qlearning_results['std'],
        qlearning_results['mean'] + qlearning_results['std'],
        alpha=0.3,
        color='green',
        label='Std Dev'
    )
    axs[1].set_title(f'Q-Learning Mean ± Std — {env_name}')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Return')
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"results/{env_name}/mean_variance_{timestamp}.png", dpi=300)
    plt.show()
    plt.close()
    
    # 4. All individual runs in one plot
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    
    # SARSA individual runs
    for i in range(len(sarsa_results.columns) - 3):  # Exclude 'episode', 'mean', 'std'
        axs[0].plot(sarsa_results['episode'], sarsa_results[f'run_{i}'], label=f'Run {i+1}')
    axs[0].set_title(f'SARSA Individual Runs — {env_name}')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Return')
    axs[0].legend()
    axs[0].grid(True)
    
    # Q-Learning individual runs
    for i in range(len(qlearning_results.columns) - 3):  # Exclude 'episode', 'mean', 'std'
        axs[1].plot(qlearning_results['episode'], qlearning_results[f'run_{i}'], label=f'Run {i+1}')
    axs[1].set_title(f'Q-Learning Individual Runs — {env_name}')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Return')
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"results/{env_name}/individual_runs_{timestamp}.png", dpi=300)
    plt.show()
    plt.close()


def hyperparameter_search(env_name, algorithm, param_grid, n_episodes=500, n_runs=3, bins=10):
    """
    Perform a grid search over hyperparameters
    
    Args:
        env_name: Name of the Gym environment
        algorithm: 'sarsa' or 'qlearning'
        param_grid: Dictionary of hyperparameters to search
        n_episodes: Number of episodes to train for each combination
        n_runs: Number of runs with different seeds
        bins: Number of bins for discretization
    
    Returns:
        List of dictionaries with hyperparameters and results
    """
    results = []
    
    # Generate all combinations of hyperparameters
    import itertools
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    for i, combo in enumerate(combinations):
        hyperparams = dict(zip(keys, combo))
        print(f"Testing combination {i+1}/{len(combinations)}: {hyperparams}")
        
        if algorithm == 'sarsa':
            result_df = train_sarsa(env_name, n_episodes=n_episodes, n_runs=n_runs, hyperparams=hyperparams, bins=bins)
        else:
            result_df = train_qlearning(env_name, n_episodes=n_episodes, n_runs=n_runs, hyperparams=hyperparams, bins=bins)
        
        # Calculate final average return
        final_avg_return = result_df['mean'].iloc[-100:].mean()
        
        results.append({
            'hyperparams': hyperparams,
            'final_avg_return': final_avg_return
        })
    
    # Sort by final average return
    results.sort(key=lambda x: x['final_avg_return'], reverse=True)
    
    return results

def run_experiment_cartpole():
    print("Running CartPole-v1 experiments")
    
    # Define hyperparameter grids
    sarsa_param_grid = {
        'learning_rate': [0.1, 0.2],
        'discount_factor': [0.99],
        'epsilon': [1.0],
        'epsilon_decay': [0.995, 0.999],
        'epsilon_min': [0.01]
    }
    
    qlearning_param_grid = {
        'learning_rate': [0.1, 0.2],
        'discount_factor': [0.99],
        'temperature': [1.0],
        'temp_decay': [0.995, 0.999],
        'temp_min': [0.1]
    }
    
    # Hyperparameter search
    print("SARSA hyperparameter search:")
    sarsa_results = hyperparameter_search('CartPole-v1', 'sarsa', sarsa_param_grid, n_episodes=300, bins=10)
    print("\nTop 3 SARSA hyperparameters:")
    for i, result in enumerate(sarsa_results[:3]):
        print(f"{i+1}. {result['hyperparams']} - Avg Return: {result['final_avg_return']:.2f}")
    
    print("\nQ-Learning hyperparameter search:")
    qlearning_results = hyperparameter_search('CartPole-v1', 'qlearning', qlearning_param_grid, n_episodes=300, bins=10)
    print("\nTop 3 Q-Learning hyperparameters:")
    for i, result in enumerate(qlearning_results[:3]):
        print(f"{i+1}. {result['hyperparams']} - Avg Return: {result['final_avg_return']:.2f}")
    
    # Train with best hyperparameters
    best_sarsa_hyperparams = sarsa_results[0]['hyperparams']
    best_qlearning_hyperparams = qlearning_results[0]['hyperparams']
    
    print("\nTraining with best hyperparameters...")
    sarsa_df = train_sarsa('CartPole-v1', n_episodes=500, n_runs=5, hyperparams=best_sarsa_hyperparams, bins=10)
    qlearning_df = train_qlearning('CartPole-v1', n_episodes=500, n_runs=5, hyperparams=best_qlearning_hyperparams, bins=10)
    
    plot_results(sarsa_df, qlearning_df, 'CartPole-v1', best_sarsa_hyperparams, best_qlearning_hyperparams)

def run_experiment_mountaincar():
    print("Running MountainCar-v0 experiments")
    
    # Define hyperparameter grids
    sarsa_param_grid = {
        'learning_rate': [0.1, 0.2],
        'discount_factor': [0.99],
        'epsilon': [1.0],
        'epsilon_decay': [0.995, 0.999],
        'epsilon_min': [0.1]
    }
    
    qlearning_param_grid = {
        'learning_rate': [0.1, 0.2],
        'discount_factor': [0.99],
        'temperature': [1.0],
        'temp_decay': [0.995, 0.999],
        'temp_min': [0.1]
    }
    
    # Hyperparameter search
    print("SARSA hyperparameter search:")
    sarsa_results = hyperparameter_search('MountainCar-v0', 'sarsa', sarsa_param_grid, n_episodes=300, bins=20)
    print("\nTop 3 SARSA hyperparameters:")
    for i, result in enumerate(sarsa_results[:3]):
        print(f"{i+1}. {result['hyperparams']} - Avg Return: {result['final_avg_return']:.2f}")
    
    print("\nQ-Learning hyperparameter search:")
    qlearning_results = hyperparameter_search('MountainCar-v0', 'qlearning', qlearning_param_grid, n_episodes=300, bins=20)
    print("\nTop 3 Q-Learning hyperparameters:")
    for i, result in enumerate(qlearning_results[:3]):
        print(f"{i+1}. {result['hyperparams']} - Avg Return: {result['final_avg_return']:.2f}")
    
    # Train with best hyperparameters
    best_sarsa_hyperparams = sarsa_results[0]['hyperparams']
    best_qlearning_hyperparams = qlearning_results[0]['hyperparams']
    
    print("\nTraining with best hyperparameters...")
    sarsa_df = train_sarsa('MountainCar-v0', n_episodes=500, n_runs=5, hyperparams=best_sarsa_hyperparams, bins=20)
    qlearning_df = train_qlearning('MountainCar-v0', n_episodes=500, n_runs=5, hyperparams=best_qlearning_hyperparams, bins=20)
    
    plot_results(sarsa_df, qlearning_df, 'MountainCar-v0', best_sarsa_hyperparams, best_qlearning_hyperparams)

def run_experiment_minigrid(bonus=True):
    if not bonus:
        print("Skipping MiniGrid-Dynamic-Obstacles-5x5-v0 (bonus task)")
        return
    
    try:
        import minigrid
        print("Running MiniGrid-Dynamic-Obstacles-5x5-v0 experiments (Bonus Task)")
        
        # Define hyperparameter grids
        sarsa_param_grid = {
            'learning_rate': [0.05, 0.1],
            'discount_factor': [0.99],
            'epsilon': [1.0],
            'epsilon_decay': [0.99, 0.995],
            'epsilon_min': [0.05]
        }
        
        qlearning_param_grid = {
            'learning_rate': [0.05, 0.1],
            'discount_factor': [0.99],
            'temperature': [1.0],
            'temp_decay': [0.99, 0.995],
            'temp_min': [0.1]
        }
        
        # Hyperparameter search
        print("SARSA hyperparameter search:")
        sarsa_results = hyperparameter_search('MiniGrid-Dynamic-Obstacles-5x5-v0', 'sarsa', sarsa_param_grid, n_episodes=300, bins=10)
        print("\nTop 3 SARSA hyperparameters:")
        for i, result in enumerate(sarsa_results[:3]):
            print(f"{i+1}. {result['hyperparams']} - Avg Return: {result['final_avg_return']:.2f}")
        
        print("\nQ-Learning hyperparameter search:")
        qlearning_results = hyperparameter_search('MiniGrid-Dynamic-Obstacles-5x5-v0', 'qlearning', qlearning_param_grid, n_episodes=300, bins=10)
        print("\nTop 3 Q-Learning hyperparameters:")
        for i, result in enumerate(qlearning_results[:3]):
            print(f"{i+1}. {result['hyperparams']} - Avg Return: {result['final_avg_return']:.2f}")
        
        # Train with best hyperparameters
        best_sarsa_hyperparams = sarsa_results[0]['hyperparams']
        best_qlearning_hyperparams = qlearning_results[0]['hyperparams']
        
        print("\nTraining with best hyperparameters...")
        sarsa_df = train_sarsa('MiniGrid-Dynamic-Obstacles-5x5-v0', n_episodes=500, n_runs=5, hyperparams=best_sarsa_hyperparams, bins=10)
        qlearning_df = train_qlearning('MiniGrid-Dynamic-Obstacles-5x5-v0', n_episodes=500, n_runs=5, hyperparams=best_qlearning_hyperparams, bins=10)
        
        plot_results(sarsa_df, qlearning_df, 'MiniGrid-Dynamic-Obstacles-5x5-v0', best_sarsa_hyperparams, best_qlearning_hyperparams)
    
    except ImportError:
        print("MiniGrid package not found. Install it using: pip install minigrid")

def main():
    # Set up output directory
    os.makedirs("results", exist_ok=True)
    
    # Run experiments
    run_experiment_cartpole()
    run_experiment_mountaincar()
    run_experiment_minigrid(bonus=True)  # Set to True to run the bonus task
    
    print("All experiments completed!")

if __name__ == "__main__":
    main()
