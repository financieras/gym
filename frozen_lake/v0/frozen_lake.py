import gymnasium as gym

# Crear el entorno FrozenLake-v1 (mapa 4x4, resbaladizo por defecto)
env = gym.make("FrozenLake-v1", render_mode="human")

# Reiniciar el entorno y obtener el estado inicial
observation, info = env.reset()

for step in range(100):
    # Elegir una acción aleatoria
    action = env.action_space.sample()
    
    # Ejecutar la acción en el entorno
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Mostrar información relevante
    print(f"Step: {step}")
    print(f"Observation: {observation}")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}, Truncated: {truncated}")
    print("-" * 30)
    
    # Si termina el episodio, reiniciar
    if terminated or truncated:
        print("Episode finished!\n")
        observation, info = env.reset()

# Cerrar el entorno al finalizar
env.close()
