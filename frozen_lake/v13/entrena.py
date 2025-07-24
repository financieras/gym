import gymnasium as gym
import numpy as np
import time
import argparse
import os
import matplotlib.pyplot as plt
from collections import deque
import pickle

def parse_arguments():
    parser = argparse.ArgumentParser(description="Optimized Q-learning agent for FrozenLake with trajectory optimization")
    parser.add_argument('--episodes', type=int, default=50000, help='Training episodes')
    parser.add_argument('--max_steps', type=int, default=100, help='Max steps per episode')
    parser.add_argument('--alpha', type=float, default=0.5, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon_start', type=float, default=1.0, help='Initial exploration rate')
    parser.add_argument('--epsilon_min', type=float, default=0.01, help='Min exploration rate')
    parser.add_argument('--epsilon_decay', type=float, default=0.99995, help='Exploration decay')
    parser.add_argument('--test_episodes', type=int, default=100, help='Testing episodes')
    parser.add_argument('--save_path', type=str, default='frozenlake_agent.pkl', help='Model save path')
    parser.add_argument('--render_mode', type=str, default='none', choices=['human', 'rgb_array', 'none'])
    parser.add_argument('--is_slippery', type=bool, default=True)
    parser.add_argument('--train', action='store_true', help='Enable training')
    parser.add_argument('--test', action='store_true', help='Enable testing')
    parser.add_argument('--optimize_trajectory', action='store_true', help='Enable trajectory optimization')
    return parser.parse_args()

class QLearningAgent:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        self.epsilon = args.epsilon_start
        self.trajectory_memory = deque(maxlen=1000)
        
        # State-action visit counts
        self.visit_counts = np.zeros((env.observation_space.n, env.action_space.n))
        
        # Reward shaping parameters
        self.step_penalty = -0.1
        self.goal_reward = 1.0
        self.hole_penalty = -1.0
        self.proximity_bonus = 0.2
        self.optimal_path_bonus = 2.0
        
        # Optimal path for 4x4 map (RIGHT=2, DOWN=1, LEFT=0, UP=3)
        self.optimal_path = {
            0: 2, 1: 2, 2: 1, 3: 1,
            4: 0, 5: 0, 6: 2, 7: 0,
            8: 0, 9: 2, 10: 1, 11: 0,
            12: 0, 13: 2, 14: 2, 15: 0
        }
    
    def get_epsilon(self, episode):
        """Dynamic epsilon based on progress"""
        if episode < 10000:
            return max(self.args.epsilon_min, 1.0 - (episode / 15000))
        return max(self.args.epsilon_min, self.epsilon * self.args.epsilon_decay)
    
    def get_alpha(self, state, action):
        """Adaptive learning rate based on visit count"""
        base_alpha = self.args.alpha
        visits = self.visit_counts[state, action] + 1
        return base_alpha / (1 + visits * 0.001)
    
    def reward_shaping(self, state, action, next_state, steps, done):
        """Enhanced reward engineering"""
        reward = 0
        
        # Base environment rewards
        if done and next_state == 15:
            reward += self.goal_reward
        elif next_state in [5, 7, 11, 12]:
            reward += self.hole_penalty
        
        # Step penalty
        reward += self.step_penalty
        
        # Proximity bonuses
        if next_state in [13, 14]:  # States adjacent to goal
            reward += self.proximity_bonus
        
        # Optimal path bonus
        if self.args.optimize_trajectory:
            if action == self.optimal_path.get(state, None):
                reward += 0.05
                
        # Trajectory optimization bonus
        if done and next_state == 15 and steps <= 10:
            reward += self.optimal_path_bonus * (1 - steps/20)
        
        return reward
    
    def choose_action(self, state, episode):
        """Epsilon-greedy with decay"""
        if np.random.random() < self.get_epsilon(episode):
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state])
    
    def update_q_table(self, state, action, reward, next_state, done):
        """Q-learning update with adaptive learning rate"""
        self.visit_counts[state, action] += 1
        alpha = self.get_alpha(state, action)
        
        current_q = self.q_table[state, action]
        next_max_q = np.max(self.q_table[next_state]) if not done else 0
        
        # Update Q-value
        new_q = current_q + alpha * (reward + self.args.gamma * next_max_q - current_q)
        self.q_table[state, action] = new_q
        
        # Store trajectory for optimization
        if self.args.optimize_trajectory and done and reward > 0:
            self.trajectory_memory.append((state, action, reward))
    
    def optimize_trajectories(self):
        """Reinforce successful trajectories"""
        if not self.trajectory_memory:
            return
            
        for state, action, reward in self.trajectory_memory:
            self.q_table[state, action] += 0.1 * reward
        
        # Clear memory after optimization
        self.trajectory_memory.clear()
    
    def save(self, path):
        """Save agent state"""
        with open(path, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'visit_counts': self.visit_counts
            }, f)
    
    def load(self, path):
        """Load agent state"""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data['q_table']
                self.visit_counts = data['visit_counts']

def train_agent(args):
    env = gym.make("FrozenLake-v1", is_slippery=args.is_slippery)
    agent = QLearningAgent(env, args)
    
    # Training statistics
    stats = {
        'episode_rewards': [],
        'success_rates': deque(maxlen=100),
        'episode_steps': deque(maxlen=100),
        'successes': 0
    }
    
    for episode in range(args.episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        for step in range(args.max_steps):
            action = agent.choose_action(state, episode)
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Custom reward shaping
            reward = agent.reward_shaping(state, action, next_state, steps, done)
            
            # Q-learning update
            agent.update_q_table(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            if done:
                break
        
        # Update statistics
        stats['episode_rewards'].append(episode_reward)
        success = (done and next_state == 15)
        if success:
            stats['successes'] += 1
            stats['success_rates'].append(1)
            stats['episode_steps'].append(steps)
        else:
            stats['success_rates'].append(0)
        
        # Periodic optimization
        if episode % 100 == 0 and args.optimize_trajectory:
            agent.optimize_trajectories()
        
        # Reduce epsilon
        agent.epsilon = agent.get_epsilon(episode)
        
        # Progress reporting
        if (episode + 1) % 1000 == 0 or episode == 0 or episode == args.episodes - 1:
            success_rate = np.mean(stats['success_rates']) * 100
            avg_steps = np.mean(stats['episode_steps']) if stats['episode_steps'] else 0
            avg_reward = np.mean(stats['episode_rewards'][-1000:]) if episode >= 1000 else 0
            
            print(f"Episode {episode + 1}/{args.episodes}: "
                  f"Success: {success_rate:.1f}% | "
                  f"Avg Steps: {avg_steps:.1f} | "
                  f"Avg Reward: {avg_reward:.3f} | "
                  f"ε: {agent.epsilon:.3f}")
    
    env.close()
    agent.save(args.save_path)
    print(f"Training complete. Model saved to {args.save_path}")
    return agent

def test_agent(args):
    env = gym.make("FrozenLake-v1", is_slippery=args.is_slippery)
    agent = QLearningAgent(env, args)
    agent.load(args.save_path)
    
    successes = 0
    steps_list = []
    
    for ep in range(args.test_episodes):
        state, _ = env.reset()
        done = False
        steps = 0
        path = []
        
        for step in range(args.max_steps):
            if args.render_mode == 'human':
                env.render()
            elif args.render_mode == 'rgb_array':
                try:
                    img = env.render()
                    plt.imshow(img)
                    plt.title(f"Test Episode {ep+1}, Step {steps}")
                    plt.axis('off')
                    plt.pause(0.01)
                    plt.clf()
                except Exception:
                    pass
            
            action = np.argmax(agent.q_table[state])
            path.append((state, action))
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            
            if done:
                success = (state == 15)
                if success:
                    successes += 1
                    steps_list.append(steps)
                    result = f"SUCCESS in {steps} steps"
                else:
                    result = "FAILED"
                
                # Print optimality analysis
                optimal_actions = 0
                for s, a in path:
                    if a == agent.optimal_path.get(s, -1):
                        optimal_actions += 1
                optimality = optimal_actions / len(path) * 100 if path else 0
                
                print(f"Test {ep+1}: {result} | Optimal Actions: {optimality:.1f}%")
                break
        else:
            print(f"Test {ep+1}: TIMEOUT after {args.max_steps} steps")
    
    # Final test report
    success_rate = (successes / args.test_episodes) * 100
    print("\n===== TEST REPORT =====")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if successes > 0:
        avg_steps = np.mean(steps_list)
        min_steps = min(steps_list)
        max_steps = max(steps_list)
        optimal_trajs = sum(1 for s in steps_list if s <= 10) / successes * 100
        
        print(f"Avg Steps (success): {avg_steps:.1f} (Min: {min_steps}, Max: {max_steps})")
        print(f"Optimal Trajectories (≤10 steps): {optimal_trajs:.1f}%")
        
        # Plot step distribution
        plt.figure(figsize=(10, 6))
        plt.hist(steps_list, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=6, color='red', linestyle='--', label='Optimal (6 steps)')
        plt.xlabel('Steps to Success')
        plt.ylabel('Frequency')
        plt.title('Successful Episode Step Distribution')
        plt.legend()
        plt.tight_layout()
        plt.savefig('frozenlake_step_distribution.png')
        plt.close()
        print("Saved step distribution plot to frozenlake_step_distribution.png")

def main():
    args = parse_arguments()
    
    if args.train:
        print("===== TRAINING STARTED =====")
        print(f"Mode: {'Slippery' if args.is_slippery else 'Deterministic'}")
        print(f"Episodes: {args.episodes}")
        train_agent(args)
    
    if args.test:
        print("\n===== TESTING STARTED =====")
        test_agent(args)

if __name__ == "__main__":
    main()