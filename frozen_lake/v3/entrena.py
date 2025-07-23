import gymnasium as gym
import numpy as np
import time

def train_agent(episodes=200000, max_steps=300):
    env = gym.make("FrozenLake-v1", is_slippery=True)
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    alpha = 0.8
    gamma = 0.95
    epsilon = 0.1

    for episode in range(episodes):
        state, _ = env.reset()
        done = False

        for _ in range(max_steps):
            if np.random.uniform() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            q_table[state, action] = old_value + alpha * (reward + gamma * next_max - old_value)

            state = next_state
            if done:
                break

    env.close()
    return q_table

def test_agent(q_table, episodes=5, max_steps=100):
    env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=True)

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        print(f"\nEpisodio {ep+1}")

        for step in range(max_steps):
            env.render()
            action = np.argmax(q_table[state])
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            time.sleep(0.3)
            if done:
                env.render()
                if reward == 1:
                    print("¡Ganaste!")
                else:
                    print("Terminó el episodio.")
                time.sleep(0.5)
                break
    env.close()

if __name__ == "__main__":
    print("Entrenando agente...")
    q_table = train_agent()
    print("Entrenamiento finalizado.")
    print("Testando agente con renderizado...")
    test_agent(q_table)
