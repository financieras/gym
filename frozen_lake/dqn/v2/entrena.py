import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import pickle

# Configurar dispositivo (GPU si está disponible, sino CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

class QNetwork(nn.Module):
    """
    Red Neuronal para aproximar la función Q(s,a)
    
    La red toma como entrada un estado (representado como one-hot vector)
    y devuelve los valores Q para todas las acciones posibles.
    """
    def __init__(self, state_size=16, action_size=4, hidden_size=256):
        super(QNetwork, self).__init__()
        
        # Red más profunda para mejor aproximación
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size//2)
        self.fc4 = nn.Linear(hidden_size//2, action_size)
        
        # Dropout para evitar overfitting
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, state):
        """Propagación hacia adelante de la red"""
        x = torch.relu(self.fc1(state))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        q_values = self.fc4(x)
        return q_values

class ReplayBuffer:
    """
    Buffer de experiencias mejorado con priorización opcional
    """
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Almacenar una experiencia en el buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Obtener una muestra aleatoria de experiencias"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """
    Agente DQN mejorado con mejor estrategia de exploración y métricas
    """
    def __init__(self, state_size=16, action_size=4, lr=0.0005, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.9999, epsilon_min=0.05):
        
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay  # Decay más lento
        self.epsilon_min = epsilon_min      # Mínimo más alto para mantener exploración
        
        # Redes neuronales más grandes
        self.q_network = QNetwork(state_size, action_size, hidden_size=256).to(device)
        self.target_network = QNetwork(state_size, action_size, hidden_size=256).to(device)
        
        # Optimizador con weight decay para regularización
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=1e-5)
        
        # Buffer más grande
        self.memory = ReplayBuffer(capacity=50000)
        
        self.update_target_network()
        
        # Métricas mejoradas
        self.scores = []
        self.losses = []
        self.steps_to_goal = []  # Nueva métrica: pasos para llegar a la meta
        self.optimal_solutions = []  # Contador de soluciones óptimas (6 pasos)
        self.epsilon_history = []  # Historial de epsilon
    
    def state_to_tensor(self, state):
        """Convertir el estado (entero 0-15) a tensor one-hot"""
        one_hot = np.zeros(self.state_size)
        one_hot[state] = 1
        return torch.FloatTensor(one_hot).unsqueeze(0).to(device)
    
    def select_action(self, state, training=True):
        """
        Selección de acción mejorada con epsilon-greedy adaptativo
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = self.state_to_tensor(state)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Almacenar experiencia en el buffer de replay"""
        self.memory.push(state, action, reward, next_state, done)
    
    def calculate_shaped_reward(self, state, next_state, reward, done, steps):
        """
        Recompensa moldeada para incentivizar rutas más cortas y exploración eficiente
        """
        # Recompensa base del entorno
        shaped_reward = reward
        
        if done and reward > 0:  # Llegó a la meta
            # Bonificación por llegar rápido (más pasos = menos bonificación)
            time_bonus = max(0, (20 - steps) * 0.1)
            shaped_reward += time_bonus
            
            # Bonificación extra por solución óptima (6 pasos)
            if steps <= 6:
                shaped_reward += 2.0
            elif steps <= 10:
                shaped_reward += 1.0
        
        elif not done:  # No terminó
            # Penalización pequeña por cada paso (incentiva eficiencia)
            shaped_reward -= 0.01
            
            # Recompensa por acercarse a la meta (distancia Manhattan)
            goal_row, goal_col = 3, 3
            current_row, current_col = divmod(state, 4)
            next_row, next_col = divmod(next_state, 4)
            
            current_distance = abs(current_row - goal_row) + abs(current_col - goal_col)
            next_distance = abs(next_row - goal_row) + abs(next_col - goal_col)
            
            # Pequeña recompensa por acercarse
            if next_distance < current_distance:
                shaped_reward += 0.05
            elif next_distance > current_distance:
                shaped_reward -= 0.02
        
        return shaped_reward
    
    def train(self, batch_size=64):
        """
        Entrenamiento mejorado con mejor manejo de experiencias
        """
        if len(self.memory) < batch_size:
            return
        
        # Muestra más grande para mejor estabilidad
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # Conversión eficiente a tensores
        states = torch.FloatTensor(np.array([self.state_to_one_hot(s) for s in states])).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array([self.state_to_one_hot(s) for s in next_states])).to(device)
        dones = torch.BoolTensor(dones).to(device)
        
        # Valores Q actuales
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: usar red principal para seleccionar acción, red objetivo para evaluar
        with torch.no_grad():
            # Seleccionar mejores acciones con red principal
            next_actions = self.q_network(next_states).argmax(1)
            # Evaluar esas acciones con red objetivo
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Huber Loss para mayor estabilidad
        loss = nn.SmoothL1Loss()(current_q_values.squeeze(), target_q_values)
        
        # Optimización
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping para estabilidad
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.losses.append(loss.item())
        
        # Decay más gradual de epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def state_to_one_hot(self, state):
        """Función auxiliar para convertir estado a one-hot"""
        one_hot = np.zeros(self.state_size)
        one_hot[state] = 1
        return one_hot
    
    def update_target_network(self):
        """Actualizar red objetivo"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath):
        """Guardar el modelo entrenado"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'scores': self.scores,
            'losses': self.losses,
            'steps_to_goal': self.steps_to_goal,
            'optimal_solutions': self.optimal_solutions,
            'epsilon_history': self.epsilon_history
        }, filepath)
        print(f"Modelo guardado en {filepath}")
    
    def load_model(self, filepath):
        """Cargar un modelo previamente entrenado"""
        checkpoint = torch.load(filepath, map_location=device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.scores = checkpoint.get('scores', [])
        self.losses = checkpoint.get('losses', [])
        self.steps_to_goal = checkpoint.get('steps_to_goal', [])
        self.optimal_solutions = checkpoint.get('optimal_solutions', [])
        self.epsilon_history = checkpoint.get('epsilon_history', [])
        print(f"Modelo cargado desde {filepath}")

def train_agent(episodes=3000, max_steps=100):
    """
    Función principal para entrenar el agente DQN con métricas mejoradas
    """
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode=None)
    agent = DQNAgent()
    
    # Métricas detalladas
    scores_window = deque(maxlen=100)
    steps_window = deque(maxlen=100)
    optimal_count = 0
    recent_optimal_count = 0
    
    print("Iniciando entrenamiento mejorado...")
    print(f"Entorno: {env.spec.id}")
    print(f"Objetivo: Llegar a la meta en el mínimo de pasos (óptimo: 6 pasos)")
    print("-" * 60)
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, env_reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            
            # Usar recompensa moldeada para mejor aprendizaje
            shaped_reward = agent.calculate_shaped_reward(state, next_state, env_reward, done, steps)
            
            agent.store_experience(state, action, shaped_reward, next_state, done)
            agent.train(batch_size=64)
            
            state = next_state
            total_reward += env_reward  # Usar recompensa original para métricas
            
            if done:
                break
        
        # Registrar métricas
        agent.scores.append(total_reward)
        agent.epsilon_history.append(agent.epsilon)
        scores_window.append(total_reward)
        
        if total_reward > 0:  # Si llegó a la meta
            agent.steps_to_goal.append(steps)
            steps_window.append(steps)
            
            if steps <= 6:  # Solución óptima
                optimal_count += 1
                recent_optimal_count += 1
                agent.optimal_solutions.append(episode)
        
        # Actualizar red objetivo cada 50 episodios
        if episode % 50 == 0:
            agent.update_target_network()
        
        # Mostrar progreso cada 100 episodios
        if episode % 100 == 0:
            avg_score = np.mean(scores_window)
            avg_steps = np.mean(steps_window) if len(steps_window) > 0 else 0
            success_rate = sum(scores_window) / len(scores_window)
            optimal_rate = recent_optimal_count / 100 if episode >= 100 else optimal_count / (episode + 1)
            
            print(f"Episodio {episode:4d} | "
                  f"Éxito: {success_rate:.2%} | "
                  f"Pasos prom: {avg_steps:.1f} | "
                  f"Soluc. óptimas: {optimal_rate:.2%} | "
                  f"Epsilon: {agent.epsilon:.3f}")
            
            recent_optimal_count = 0  # Reset contador para siguiente ventana
        
        # Criterio de parada mejorado
        if (len(scores_window) == 100 and 
            np.mean(scores_window) >= 0.85 and 
            len(steps_window) > 0 and 
            np.mean(steps_window) <= 10):
            print(f"\n¡Entorno resuelto eficientemente en {episode} episodios!")
            print(f"Tasa de éxito: {np.mean(scores_window):.2%}")
            print(f"Pasos promedio: {np.mean(steps_window):.1f}")
            break
    
    env.close()
    
    # Estadísticas finales
    successful_episodes = [i for i, score in enumerate(agent.scores) if score > 0]
    if len(agent.steps_to_goal) > 0:
        print(f"\n=== ESTADÍSTICAS FINALES ===")
        print(f"Episodios totales: {episode + 1}")
        print(f"Episodios exitosos: {len(successful_episodes)}")
        print(f"Tasa de éxito final: {len(successful_episodes)/(episode + 1):.2%}")
        print(f"Pasos promedio a la meta: {np.mean(agent.steps_to_goal):.1f}")
        print(f"Mínimo pasos registrado: {min(agent.steps_to_goal)}")
        print(f"Soluciones óptimas (≤6 pasos): {optimal_count}")
        print(f"Tasa de soluciones óptimas: {optimal_count/len(successful_episodes):.2%}")
    
    return agent

def evaluate_agent(agent, episodes=100, render=False):
    """
    Evaluación detallada del agente entrenado
    """
    render_mode = 'human' if render else None
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode=render_mode)
    
    total_rewards = []
    steps_to_goal = []
    success_count = 0
    optimal_count = 0
    
    print(f"\nEvaluando agente durante {episodes} episodios...")
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(100):
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        total_rewards.append(total_reward)
        if total_reward > 0:
            success_count += 1
            steps_to_goal.append(steps)
            if steps <= 6:
                optimal_count += 1
    
    env.close()
    
    # Estadísticas detalladas
    success_rate = success_count / episodes
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(steps_to_goal) if steps_to_goal else 0
    optimal_rate = optimal_count / success_count if success_count > 0 else 0
    
    print(f"\n--- RESULTADOS DE EVALUACIÓN DETALLADA ---")
    print(f"Episodios evaluados: {episodes}")
    print(f"Tasa de éxito: {success_rate:.2%}")
    print(f"Recompensa promedio: {avg_reward:.3f}")
    print(f"Episodios exitosos: {success_count}/{episodes}")
    
    if steps_to_goal:
        print(f"Pasos promedio a la meta: {avg_steps:.1f}")
        print(f"Mínimo pasos: {min(steps_to_goal)}")
        print(f"Máximo pasos: {max(steps_to_goal)}")
        print(f"Soluciones óptimas (≤6 pasos): {optimal_count}/{success_count}")
        print(f"Tasa de soluciones óptimas: {optimal_rate:.2%}")
        
        # Distribución de pasos
        steps_distribution = {}
        for steps in steps_to_goal:
            steps_distribution[steps] = steps_distribution.get(steps, 0) + 1
        print(f"Distribución de pasos: {dict(sorted(steps_distribution.items()))}")
    
    return success_rate, avg_reward, avg_steps, optimal_rate

def plot_training_results(agent):
    """
    Visualización mejorada de los resultados del entrenamiento
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Gráfico 1: Recompensas por episodio
    ax1.plot(agent.scores, alpha=0.6, color='blue', linewidth=0.8, label='Recompensas')
    
    if len(agent.scores) >= 100:
        moving_avg = []
        for i in range(99, len(agent.scores)):
            moving_avg.append(np.mean(agent.scores[i-99:i+1]))
        ax1.plot(range(99, len(agent.scores)), moving_avg, color='red', linewidth=2, label='Promedio móvil (100 ep.)')
        ax1.legend()
    
    ax1.set_xlabel('Episodio')
    ax1.set_ylabel('Recompensa')
    ax1.set_title('Recompensas durante el Entrenamiento')
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Pasos para llegar a la meta
    if agent.steps_to_goal:
        episodes_with_success = []
        current_episode = 0
        for i, score in enumerate(agent.scores):
            if score > 0:
                episodes_with_success.append(i)
        
        ax2.scatter(episodes_with_success, agent.steps_to_goal, alpha=0.6, color='green', s=20)
        ax2.axhline(y=6, color='red', linestyle='--', label='Óptimo (6 pasos)')
        ax2.set_xlabel('Episodio')
        ax2.set_ylabel('Pasos a la Meta')
        ax2.set_title('Eficiencia del Camino a la Meta')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Gráfico 3: Pérdidas del entrenamiento
    if agent.losses:
        ax3.plot(agent.losses, alpha=0.7, color='orange')
        ax3.set_xlabel('Paso de entrenamiento')
        ax3.set_ylabel('Pérdida (Smooth L1)')
        ax3.set_title('Pérdida durante el Entrenamiento')
        ax3.grid(True, alpha=0.3)
    
    # Gráfico 4: Evolución de Epsilon
    if agent.epsilon_history:
        ax4.plot(agent.epsilon_history, color='purple', linewidth=2)
        ax4.set_xlabel('Episodio')
        ax4.set_ylabel('Epsilon')
        ax4.set_title('Evolución de la Exploración (Epsilon)')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def demo_trained_agent(agent, num_demos=5):
    """
    Demostración mejorada del agente entrenado
    """
    print("\n--- DEMOSTRACIÓN DEL AGENTE ENTRENADO MEJORADO ---")
    
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode=None)
    
    # Mapa del entorno para referencia
    map_desc = ["SFFF", "FHFH", "FFFH", "HFFG"]
    print("Mapa del entorno:")
    for i, row in enumerate(map_desc):
        print(f"Fila {i}: {' '.join(row)}")
    print("S=Inicio, F=Hielo, H=Agujero, G=Meta")
    print("Posición objetivo: (3,3) - Estado 15")
    print("-" * 40)
    
    successful_demos = 0
    optimal_demos = 0
    
    for demo in range(num_demos):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        path = [state]
        
        print(f"\nDemo {demo + 1}:")
        
        for step in range(50):  # Límite más alto
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            steps += 1
            total_reward += reward
            path.append(next_state)
            
            actions_names = ['←', '↓', '→', '↑']
            row, col = divmod(state, 4)
            next_row, next_col = divmod(next_state, 4)
            
            print(f"  Paso {steps}: ({row},{col})→{actions_names[action]}→({next_row},{next_col})")
            
            state = next_state
            
            if terminated or truncated:
                if reward > 0:
                    successful_demos += 1
                    if steps <= 6:
                        optimal_demos += 1
                    print(f"  ✅ ¡ÉXITO en {steps} pasos! {'(ÓPTIMO)' if steps <= 6 else ''}")
                else:
                    print(f"  ❌ Cayó en agujero en paso {steps}")
                break
        else:
            print(f"  ⏰ Se agotó el tiempo ({steps} pasos)")
        
        print(f"  Recompensa: {total_reward}")
        print(f"  Camino: {' → '.join(map(str, path))}")
    
    env.close()
    
    print(f"\n--- RESUMEN DE DEMOSTRACIONES ---")
    print(f"Demos exitosas: {successful_demos}/{num_demos} ({successful_demos/num_demos:.1%})")
    print(f"Demos óptimas: {optimal_demos}/{num_demos} ({optimal_demos/num_demos:.1%})")
    if successful_demos > 0:
        print(f"Tasa óptima entre éxitos: {optimal_demos/successful_demos:.1%}")

# Función principal
if __name__ == "__main__":
    print("=== DEEP Q-LEARNING MEJORADO PARA FROZEN LAKE V1 ===")
    print("Versión optimizada con:")
    print("• Recompensas moldeadas para rutas eficientes")
    print("• Métricas de soluciones óptimas (≤6 pasos)")
    print("• Exploración mejorada con epsilon decay gradual")
    print("• Red neuronal más profunda y técnicas de estabilización")
    print("• Double DQN y gradient clipping\n")
    
    # Entrenar el agente mejorado
    trained_agent = train_agent(episodes=3000)
    
    # Guardar el modelo
    trained_agent.save_model('frozen_lake_dqn_optimized.pth')
    
    # Evaluación detallada
    success_rate, avg_reward, avg_steps, optimal_rate = evaluate_agent(trained_agent, episodes=200)
    
    # Visualizar resultados
    plot_training_results(trained_agent)
    
    # Demostración mejorada
    demo_choice = input("\n¿Quieres ver demostraciones del agente optimizado? (s/n): ")
    if demo_choice.lower() == 's':
        demo_trained_agent(trained_agent, num_demos=10)
    
    print("\n=== ENTRENAMIENTO OPTIMIZADO COMPLETADO ===")
    print("El modelo optimizado ha sido guardado como 'frozen_lake_dqn_optimized.pth'")