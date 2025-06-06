# Policy Gradient Methods for CartPole-v1

This repository implements and compares three policy gradient reinforcement learning algorithms on the CartPole-v1 environment:

1. **REINFORCE** - A Monte Carlo policy gradient algorithm
2. **Actor-Critic (AC)** - A hybrid approach with separate policy and value networks
3. **Advantage Actor-Critic (A2C)** - An improved actor-critic using advantage function

## Project Overview

This project explores policy-based reinforcement learning methods, focusing on their implementation, optimization, and comparative analysis in the CartPole-v1 environment. The code includes comprehensive experiments, hyperparameter tuning through grid search, and visualization tools for performance analysis.

## Features

- Implementation of three policy gradient algorithms:
  - REINFORCE with Monte Carlo returns
  - Actor-Critic with separate value network
  - Advantage Actor-Critic with advantage function
- Comprehensive grid search for hyperparameter optimization
- Learning curve visualization and performance comparison
- Detailed logging and result analysis

## Requirements

The project requires the following Python packages:
```
gymnasium>=0.26.0
torch>=2.0.0
numpy>=1.21.0
matplotlib>=3.4.0
pandas>=1.3.0
```

You can install all dependencies using:
```bash
pip install -r requirements.txt
```

## Project Structure

- `policy_networks.py`: Core implementation of policy and value networks, training functions for all three algorithms
- `gird_search.py`: Grid search implementation for hyperparameter optimization
- `requirements.txt`: List of required Python packages
- `results/`: Directory where CSV results are automatically saved
- `plots/`: Directory for generated graphs and learning curves

## Setup Instructions

1. Clone this repository
2. Install Python 3.7 or above
3. (Optional but recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running Experiments

The code is designed to execute one experiment at a time:

### Running Grid Search

To run the grid search for all three algorithms, in `gird_search.py` uncomment the desired function and run:

```bash
python gird_search.py
```

Available grid search functions:
- `run_grid_search_reinforce()` - Grid search for REINFORCE
- `run_grid_search_actor_critic()` - Grid search for Actor-Critic
- `run_grid_search_a2c()` - Grid search for A2C

### Running Individual Algorithms

To run individual algorithms with default settings, in `policy_networks.py` uncomment the corresponding function call and run:

```bash
python policy_networks.py
```

## Output

After running an experiment, the following outputs will be automatically saved:

- **Learning Curves**: Smoothed plots of rewards over time in the "plots" directory
- **Final Performance Charts**: Bar charts comparing the final rewards of different configurations
- **CSV Files**: Detailed logs of episode rewards and steps in the "results" directory
- **Console Logs**: Training progress and best hyperparameter combinations

## Algorithms

### REINFORCE
- Uses Monte Carlo returns to estimate the value function
- Updates policy parameters using the policy gradient theorem
- Normalizes returns for improved stability

### Actor-Critic (AC)
- Combines a policy network with a separate value network
- Uses TD learning for the value function
- Reduces variance compared to REINFORCE

### Advantage Actor-Critic (A2C)
- Uses the advantage function (difference between Monte Carlo return and value estimate)
- Improves training stability and efficiency
- Reduces variance in policy updates

## Author

Praneeth Dathu
Sai krishna mulakayala

## License

This project is available under the MIT License.

