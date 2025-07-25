import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

class QLearningAgent:
    """
    Agente que implementa el algoritmo Q-Learning para resolver el problema de Frozen Lake.
    
    Q-Learning es un algoritmo de aprendizaje por refuerzo que aprende una función Q(s,a)
    que representa el valor esperado de tomar la acción 'a' en el estado 's'.
    """
    
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Inicializa el agente Q-Learning.
        
        Parámetros:
        - env: Entorno de gymnasium (Frozen Lake)
        - learning_rate (alpha): Tasa de aprendizaje (0 < α ≤ 1)
        - discount_factor (gamma): Factor de descuento para recompensas futuras (0 ≤ γ ≤ 1)
        - epsilon: Probabilidad inicial de exploración (estrategia ε-greedy)
        - epsilon_decay: Factor de decaimiento de epsilon
        - epsilon_min: Valor mínimo de epsilon
        """
        self.env = env
        self.n_states = env.observation_space.n  # Número de estados (16 para mapa 4x4)
        self.n_actions = env.action_space.n      # Número de acciones (4: izquierda, abajo, derecha, arriba)
        
        # Parámetros del algoritmo Q-Learning
        self.learning_rate = learning_rate      # α (alpha)
        self.discount_factor = discount_factor  # γ (gamma)
        self.epsilon = epsilon                  # ε (epsilon) para estrategia ε-greedy
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Inicializar la Q-table con valores cero
        # Q-table[estado][acción] = valor Q
        self.q_table = np.zeros((self.n_states, self.n_actions))
        
        print(f"🧊 Agente Q-Learning inicializado:")
        print(f"   - Estados: {self.n_states}")
        print(f"   - Acciones: {self.n_actions}")
        print(f"   - Tasa de aprendizaje (α): {self.learning_rate}")
        print(f"   - Factor de descuento (γ): {self.discount_factor}")
        print(f"   - Epsilon inicial (ε): {self.epsilon}")
    
    def choose_action(self, state):
        """
        Selecciona una acción usando la estrategia ε-greedy:
        - Con probabilidad ε: explora (acción aleatoria)
        - Con probabilidad (1-ε): explota (mejor acción conocida)
        """
        if np.random.random() < self.epsilon:
            # Exploración: elegir acción aleatoria
            return self.env.action_space.sample()
        else:
            # Explotación: elegir la mejor acción conocida (mayor valor Q)
            return np.argmax(self.q_table[state])
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Actualiza el valor Q usando la ecuación de Bellman:
        
        Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        
        Donde:
        - Q(s,a): Valor Q actual para el estado s y acción a
        - α (alpha): Tasa de aprendizaje
        - r: Recompensa inmediata
        - γ (gamma): Factor de descuento
        - max(Q(s',a')): Mejor valor Q posible en el siguiente estado
        """
        # Valor Q actual
        current_q = self.q_table[state, action]
        
        # Mejor valor Q posible en el siguiente estado
        max_next_q = np.max(self.q_table[next_state])
        
        # Calcular el nuevo valor Q usando la ecuación de Bellman
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        # Actualizar la Q-table
        self.q_table[state, action] = new_q
    
    def decay_epsilon(self):
        """
        Reduce gradualmente epsilon para disminuir la exploración con el tiempo.
        Al principio exploramos mucho, pero gradualmente explotamos más el conocimiento adquirido.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train_agent(episodes=1000, render_last_episodes=5):
    """
    Entrena al agente Q-Learning en el entorno Frozen Lake.
    
    Parámetros:
    - episodes: Número de episodios de entrenamiento
    - render_last_episodes: Número de episodios finales a mostrar visualmente
    """
    # Crear el entorno Frozen Lake
    # is_slippery=True significa que el lago es resbaladizo (más realista y desafiante)
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode=None)
    
    # Crear el agente Q-Learning
    agent = QLearningAgent(env)
    
    # Métricas para seguimiento del progreso
    rewards_per_episode = []
    success_rate_window = []
    window_size = 100  # Ventana para calcular tasa de éxito
    
    print(f"\n🚀 Iniciando entrenamiento por {episodes} episodios...")
    print("=" * 60)
    
    # Bucle principal de entrenamiento
    for episode in range(episodes):
        # Reiniciar el entorno para un nuevo episodio
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        # Bucle del episodio
        while not done:
            # El agente elige una acción
            action = agent.choose_action(state)
            
            # Ejecutar la acción en el entorno
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Actualizar el valor Q usando la ecuación de Bellman
            agent.update_q_value(state, action, reward, next_state)
            
            # Actualizar estado y métricas
            state = next_state
            total_reward += reward
            steps += 1
        
        # Reducir epsilon (menos exploración con el tiempo)
        agent.decay_epsilon()
        
        # Guardar métricas
        rewards_per_episode.append(total_reward)
        success_rate_window.append(total_reward > 0)  # Éxito si llegó a la meta
        
        # Mantener ventana deslizante para tasa de éxito
        if len(success_rate_window) > window_size:
            success_rate_window.pop(0)
        
        # Mostrar progreso cada 100 episodios
        if (episode + 1) % 100 == 0:
            success_rate = np.mean(success_rate_window) * 100
            print(f"Episodio {episode + 1:4d} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Tasa éxito últimos {len(success_rate_window)} ep.: {success_rate:.1f}% | "
                  f"Pasos: {steps:2d} | "
                  f"Recompensa: {total_reward:.1f}")
    
    env.close()
    
    # Mostrar algunos episodios finales con visualización
    print(f"\n🎬 Mostrando los últimos {render_last_episodes} episodios con visualización...")
    test_agent(agent, episodes=render_last_episodes, render=True)
    
    # Mostrar estadísticas finales
    print_final_statistics(agent, rewards_per_episode)
    
    # Graficar progreso del entrenamiento
    plot_training_progress(rewards_per_episode)
    
    return agent

def test_agent(agent, episodes=100, render=False):
    """
    Evalúa el rendimiento del agente entrenado.
    """
    render_mode = 'human' if render else None
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode=render_mode)
    
    successes = 0
    total_steps = 0
    
    for episode in range(episodes):
        state, _ = env.reset()
        steps = 0
        done = False
        
        while not done:
            # En evaluación, siempre elegir la mejor acción (sin exploración)
            action = np.argmax(agent.q_table[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            
            if render:
                time.sleep(0.1)  # Pausa para visualización
        
        if reward > 0:  # Llegó a la meta
            successes += 1
        
        total_steps += steps
        
        if render:
            print(f"Episodio {episode + 1}: {'✅ Éxito' if reward > 0 else '❌ Fallo'} en {steps} pasos")
    
    env.close()
    
    success_rate = (successes / episodes) * 100
    avg_steps = total_steps / episodes
    
    if not render:
        print(f"\n📊 Evaluación completada:")
        print(f"   - Tasa de éxito: {success_rate:.1f}% ({successes}/{episodes})")
        print(f"   - Pasos promedio: {avg_steps:.1f}")
    
    return success_rate, avg_steps

def print_final_statistics(agent, rewards_per_episode):
    """
    Muestra estadísticas finales del entrenamiento y la Q-table aprendida.
    """
    print("\n" + "=" * 60)
    print("📈 ESTADÍSTICAS FINALES DEL ENTRENAMIENTO")
    print("=" * 60)
    
    total_episodes = len(rewards_per_episode)
    successful_episodes = sum(1 for r in rewards_per_episode if r > 0)
    success_rate = (successful_episodes / total_episodes) * 100
    
    print(f"Episodios totales: {total_episodes}")
    print(f"Episodios exitosos: {successful_episodes}")
    print(f"Tasa de éxito final: {success_rate:.2f}%")
    print(f"Epsilon final: {agent.epsilon:.4f}")
    
    print(f"\n🧠 Q-TABLE APRENDIDA:")
    print("Cada fila representa un estado, cada columna una acción [Izq, Abajo, Der, Arriba]")
    print("-" * 40)
    for state in range(agent.n_states):
        row = state // 4
        col = state % 4
        print(f"Estado {state:2d} (fila {row}, col {col}): {agent.q_table[state]}")

def plot_training_progress(rewards_per_episode):
    """
    Grafica el progreso del entrenamiento.
    """
    # Calcular media móvil para suavizar la curva
    window_size = 100
    if len(rewards_per_episode) >= window_size:
        moving_avg = []
        for i in range(len(rewards_per_episode) - window_size + 1):
            moving_avg.append(np.mean(rewards_per_episode[i:i + window_size]))
        
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Recompensas por episodio
        plt.subplot(1, 2, 1)
        plt.plot(rewards_per_episode, alpha=0.3, label='Recompensa por episodio')
        plt.plot(range(window_size-1, len(rewards_per_episode)), moving_avg, 
                label=f'Media móvil ({window_size} episodios)', linewidth=2)
        plt.xlabel('Episodio')
        plt.ylabel('Recompensa')
        plt.title('Progreso del Entrenamiento')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Tasa de éxito acumulativa
        plt.subplot(1, 2, 2)
        cumulative_success = np.cumsum([1 if r > 0 else 0 for r in rewards_per_episode])
        success_rate = cumulative_success / np.arange(1, len(rewards_per_episode) + 1) * 100
        plt.plot(success_rate)
        plt.xlabel('Episodio')
        plt.ylabel('Tasa de Éxito (%)')
        plt.title('Tasa de Éxito Acumulativa')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Función principal que ejecuta el entrenamiento completo.
    """
    print("🧊 FROZEN LAKE con Q-LEARNING 🤖")
    print("=" * 50)
    print("Este programa implementa el algoritmo Q-Learning para resolver")
    print("el problema de Frozen Lake, donde un agente debe navegar por")
    print("un lago congelado resbaladizo desde el inicio hasta la meta,")
    print("evitando caer en los agujeros.")
    print()
    print("Leyenda del mapa:")
    print("  S = Inicio (Start)")
    print("  F = Superficie congelada (Frozen)")
    print("  H = Agujero (Hole)")
    print("  G = Meta (Goal)")
    print()
    
    # Entrenar el agente
    trained_agent = train_agent(episodes=2000, render_last_episodes=3)
    
    # Evaluación final sin visualización
    print(f"\n🎯 Evaluación final con 1000 episodios...")
    final_success_rate, final_avg_steps = test_agent(trained_agent, episodes=1000, render=False)
    
    print(f"\n🏆 RESULTADO FINAL:")
    print(f"   El agente alcanzó una tasa de éxito del {final_success_rate:.1f}%")
    print(f"   con un promedio de {final_avg_steps:.1f} pasos por episodio.")
    
    if final_success_rate > 70:
        print("   ¡Excelente! El agente ha aprendido una buena política. 🎉")
    elif final_success_rate > 50:
        print("   ¡Bien! El agente ha aprendido una política decente. 👍")
    else:
        print("   El agente necesita más entrenamiento. 🤔")

# Ejecutar el programa principal
if __name__ == "__main__":
    main()