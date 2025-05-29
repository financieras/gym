import numpy as np
import gymnasium as gym

# Hiperparámetros
alpha = 0.8           # Tasa de aprendizaje
gamma = 0.95          # Factor de descuento
epsilon = 1.0         # Probabilidad inicial de exploración
epsilon_min = 0.01    # Probabilidad mínima de exploración
epsilon_decay = 0.995 # Decaimiento de epsilon por episodio
num_episodes = 10000  # Número de episodios de entrenamiento
max_steps = 100       # Pasos máximos por episodio

# Crear el entorno
env = gym.make("FrozenLake-v1", is_slippery=True)

# Inicializar la Q-table
state_space = env.observation_space.n
action_space = env.action_space.n
Q = np.zeros((state_space, action_space))

# Entrenamiento con Q-Learning
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False

    for step in range(max_steps):
        # Política epsilon-greedy
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Exploración
        else:
            action = np.argmax(Q[state, :])     # Explotación

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Actualización Q-Learning
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state, :]) - Q[state, action]
        )

        state = next_state

        if done:
            break

    # Decaimiento de epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

print("Entrenamiento completado.\n")

# Evaluación del agente entrenado
env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=True)
for episode in range(5):
    state, _ = env.reset()
    done = False
    print(f"--- Episodio {episode + 1} ---")
    for step in range(max_steps):
        action = np.argmax(Q[state, :])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        if done:
            if reward == 1:
                print("¡Llegaste a la meta!")
            else:
                print("Caíste en un agujero.")
            break

env.close()
