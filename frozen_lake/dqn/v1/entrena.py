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
    def __init__(self, state_size=16, action_size=4, hidden_size=128):
        super(QNetwork, self).__init__()
        
        # Definir las capas de la red neuronal
        # Capa de entrada: state_size (16) -> hidden_size (128)
        self.fc1 = nn.Linear(state_size, hidden_size)
        # Capa oculta: hidden_size -> hidden_size  
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Capa de salida: hidden_size -> action_size (4)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state):
        """
        Propagación hacia adelante de la red
        """
        # Aplicar ReLU después de cada capa excepto la última
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        # La capa de salida no tiene activación (valores Q pueden ser negativos)
        q_values = self.fc3(x)
        return q_values

class ReplayBuffer:
    """
    Buffer de experiencias para almacenar las transiciones (s, a, r, s', done)
    
    Esto permite entrenar la red con experiencias pasadas, rompiendo la correlación
    temporal entre experiencias consecutivas y mejorando la estabilidad del entrenamiento.
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Almacenar una experiencia en el buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Obtener una muestra aleatoria de experiencias"""
        batch = random.sample(self.buffer, batch_size)
        # Separar los componentes de la experiencia
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """
    Agente que implementa Deep Q-Network (DQN)
    
    DQN combina Q-Learning con redes neuronales profundas para aproximar
    la función de valor-acción Q(s,a) en espacios de estados grandes.
    """
    def __init__(self, state_size=16, action_size=4, lr=0.001, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        
        self.state_size = state_size      # Tamaño del espacio de estados (16 posiciones)
        self.action_size = action_size    # Tamaño del espacio de acciones (4 direcciones)
        self.lr = lr                      # Tasa de aprendizaje
        self.gamma = gamma                # Factor de descuento (importancia del futuro)
        self.epsilon = epsilon            # Probabilidad de exploración inicial
        self.epsilon_decay = epsilon_decay # Tasa de decaimiento de epsilon
        self.epsilon_min = epsilon_min    # Valor mínimo de epsilon
        
        # Crear las redes neuronales
        # Red principal: se entrena constantemente
        self.q_network = QNetwork(state_size, action_size).to(device)
        # Red objetivo: se actualiza periódicamente para estabilizar el entrenamiento
        self.target_network = QNetwork(state_size, action_size).to(device)
        
        # Optimizador Adam para entrenar la red principal
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Buffer de experiencias para Experience Replay
        self.memory = ReplayBuffer()
        
        # Inicializar la red objetivo con los mismos pesos que la red principal
        self.update_target_network()
        
        # Listas para almacenar métricas de entrenamiento
        self.scores = []
        self.losses = []
    
    def state_to_tensor(self, state):
        """
        Convertir el estado (entero 0-15) a tensor one-hot
        
        Ejemplo: estado 5 -> [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
        """
        one_hot = np.zeros(self.state_size)
        one_hot[state] = 1
        return torch.FloatTensor(one_hot).unsqueeze(0).to(device)
    
    def select_action(self, state, training=True):
        """
        Seleccionar acción usando la estrategia epsilon-greedy
        
        Con probabilidad epsilon: explorar (acción aleatoria)
        Con probabilidad 1-epsilon: explotar (mejor acción según Q-network)
        """
        if training and random.random() < self.epsilon:
            # Exploración: seleccionar acción aleatoria
            return random.randrange(self.action_size)
        
        # Explotación: seleccionar la mejor acción según la red neuronal
        state_tensor = self.state_to_tensor(state)
        with torch.no_grad():  # No calcular gradientes para ahorrar memoria
            q_values = self.q_network(state_tensor)
        
        # Devolver la acción con el mayor valor Q
        return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Almacenar experiencia en el buffer de replay"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train(self, batch_size=32):
        """
        Entrenar la red neuronal usando experiencias del buffer
        
        Implementa la ecuación de Bellman:
        Q(s,a) = r + γ * max(Q(s',a'))
        """
        # No entrenar si no hay suficientes experiencias
        if len(self.memory) < batch_size:
            return
        
        # Obtener muestra aleatoria del buffer
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # Convertir a tensores
        states = torch.FloatTensor([self.state_to_one_hot(s) for s in states]).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor([self.state_to_one_hot(s) for s in next_states]).to(device)
        dones = torch.BoolTensor(dones).to(device)
        
        # Calcular valores Q actuales: Q(s,a)
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Calcular valores Q objetivo usando la red objetivo
        with torch.no_grad():
            # max(Q(s',a')) - máximo valor Q para el siguiente estado
            next_q_values = self.target_network(next_states).max(1)[0]
            # Si el episodio terminó, el valor objetivo es solo la recompensa
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Calcular la pérdida (diferencia entre Q actual y Q objetivo)
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Retropropagación y actualización de pesos
        self.optimizer.zero_grad()  # Limpiar gradientes anteriores
        loss.backward()             # Calcular gradientes
        self.optimizer.step()       # Actualizar pesos
        
        # Guardar la pérdida para análisis
        self.losses.append(loss.item())
        
        # Decaer epsilon (reducir exploración gradualmente)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def state_to_one_hot(self, state):
        """Función auxiliar para convertir estado a one-hot"""
        one_hot = np.zeros(self.state_size)
        one_hot[state] = 1
        return one_hot
    
    def update_target_network(self):
        """
        Copiar los pesos de la red principal a la red objetivo
        
        Esto se hace periódicamente para estabilizar el entrenamiento.
        La red objetivo proporciona valores Q más estables para el cálculo del objetivo.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath):
        """Guardar el modelo entrenado"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'scores': self.scores,
            'losses': self.losses
        }, filepath)
        print(f"Modelo guardado en {filepath}")
    
    def load_model(self, filepath):
        """Cargar un modelo previamente entrenado"""
        checkpoint = torch.load(filepath, map_location=device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.scores = checkpoint['scores']
        self.losses = checkpoint['losses']
        print(f"Modelo cargado desde {filepath}")

def train_agent(episodes=2000, max_steps=100):
    """
    Función principal para entrenar el agente DQN
    """
    # Crear el entorno Frozen Lake
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode=None)
    
    # Crear el agente DQN
    agent = DQNAgent()
    
    # Variables para monitorear el progreso
    scores_window = deque(maxlen=100)  # Ventana para calcular promedio móvil
    
    print("Iniciando entrenamiento...")
    print(f"Entorno: {env.spec.id}")
    print(f"Espacio de estados: {env.observation_space.n}")
    print(f"Espacio de acciones: {env.action_space.n}")
    print("-" * 50)
    
    for episode in range(episodes):
        # Reiniciar el entorno
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Seleccionar acción
            action = agent.select_action(state)
            
            # Ejecutar acción en el entorno
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Almacenar experiencia
            agent.store_experience(state, action, reward, next_state, done)
            
            # Entrenar el agente
            agent.train()
            
            # Actualizar estado y recompensa total
            state = next_state
            total_reward += reward
            
            # Si el episodio terminó, salir del bucle
            if done:
                break
        
        # Guardar puntuación del episodio
        agent.scores.append(total_reward)
        scores_window.append(total_reward)
        
        # Actualizar red objetivo cada 100 episodios
        if episode % 100 == 0:
            agent.update_target_network()
        
        # Mostrar progreso cada 100 episodios
        if episode % 100 == 0:
            avg_score = np.mean(scores_window)
            print(f"Episodio {episode:4d} | Promedio últimos 100: {avg_score:.3f} | Epsilon: {agent.epsilon:.3f}")
        
        # Criterio de parada: si el agente resuelve el entorno consistentemente
        if len(scores_window) == 100 and np.mean(scores_window) >= 0.78:
            print(f"\n¡Entorno resuelto en {episode} episodios!")
            print(f"Promedio últimos 100 episodios: {np.mean(scores_window):.3f}")
            break
    
    env.close()
    return agent

def evaluate_agent(agent, episodes=100, render=False):
    """
    Evaluar el rendimiento del agente entrenado
    """
    render_mode = 'human' if render else None
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode=render_mode)
    
    total_rewards = []
    success_count = 0
    
    print(f"\nEvaluando agente durante {episodes} episodios...")
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(100):  # Máximo 100 pasos por episodio
            # Seleccionar acción sin exploración (epsilon = 0)
            action = agent.select_action(state, training=False)
            
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        total_rewards.append(total_reward)
        if total_reward > 0:
            success_count += 1
        
        if render and episode < 5:  # Mostrar solo los primeros 5 episodios
            print(f"Episodio {episode + 1}: Recompensa = {total_reward}")
    
    env.close()
    
    # Mostrar estadísticas
    success_rate = success_count / episodes
    avg_reward = np.mean(total_rewards)
    
    print(f"\n--- RESULTADOS DE EVALUACIÓN ---")
    print(f"Episodios: {episodes}")
    print(f"Tasa de éxito: {success_rate:.2%}")
    print(f"Recompensa promedio: {avg_reward:.3f}")
    print(f"Episodios exitosos: {success_count}/{episodes}")
    
    return success_rate, avg_reward

def plot_training_results(agent):
    """
    Visualizar los resultados del entrenamiento
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfico de recompensas por episodio
    ax1.plot(agent.scores, alpha=0.6, color='blue', linewidth=0.8)
    
    # Calcular y graficar promedio móvil
    if len(agent.scores) >= 100:
        moving_avg = []
        for i in range(99, len(agent.scores)):
            moving_avg.append(np.mean(agent.scores[i-99:i+1]))
        ax1.plot(range(99, len(agent.scores)), moving_avg, color='red', linewidth=2, label='Promedio móvil (100 episodios)')
        ax1.legend()
    
    ax1.set_xlabel('Episodio')
    ax1.set_ylabel('Recompensa')
    ax1.set_title('Recompensas durante el Entrenamiento')
    ax1.grid(True, alpha=0.3)
    
    # Gráfico de pérdidas
    if agent.losses:
        ax2.plot(agent.losses, alpha=0.7, color='orange')
        ax2.set_xlabel('Paso de entrenamiento')
        ax2.set_ylabel('Pérdida (MSE)')
        ax2.set_title('Pérdida durante el Entrenamiento')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def demo_trained_agent(agent):
    """
    Demostración visual del agente entrenado
    """
    print("\n--- DEMOSTRACIÓN DEL AGENTE ENTRENADO ---")
    print("Observa cómo el agente navega por el lago congelado...")
    
    # Crear entorno con renderizado
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode='human')
    
    for demo in range(3):  # Hacer 3 demostraciones
        print(f"\nDemostración {demo + 1}:")
        state, _ = env.reset()
        total_reward = 0
        step_count = 0
        
        print("Mapa del entorno:")
        print("S = Inicio, F = Hielo, H = Agujero, G = Meta")
        
        while True:
            # Mostrar estado actual
            row, col = divmod(state, 4)
            print(f"Paso {step_count + 1}: Posición ({row}, {col}) - Estado {state}")
            
            # Seleccionar acción (sin exploración)
            action = agent.select_action(state, training=False)
            actions_names = ['←', '↓', '→', '↑']
            print(f"Acción elegida: {actions_names[action]}")
            
            # Ejecutar acción
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            step_count += 1
            
            if terminated:
                if reward > 0:
                    print("¡Éxito! El agente llegó a la meta.")
                else:
                    print("El agente cayó en un agujero.")
                break
            elif truncated:
                print("Se agotó el tiempo máximo.")
                break
            elif step_count >= 20:  # Limitar pasos para la demo
                print("Demo limitada a 20 pasos.")
                break
        
        print(f"Recompensa total: {total_reward}")
        input("Presiona Enter para la siguiente demostración...")
    
    env.close()

# Función principal
if __name__ == "__main__":
    print("=== DEEP Q-LEARNING PARA FROZEN LAKE V1 ===")
    print("Este proyecto implementa DQN para resolver el entorno Frozen Lake.")
    print("El agente debe aprender a navegar por un lago congelado resbaladizo")
    print("desde el inicio hasta la meta evitando caer en los agujeros.\n")
    
    # Entrenar el agente
    trained_agent = train_agent(episodes=2000)
    
    # Guardar el modelo entrenado
    trained_agent.save_model('frozen_lake_dqn_model.pth')
    
    # Evaluar el agente
    success_rate, avg_reward = evaluate_agent(trained_agent, episodes=100)
    
    # Visualizar resultados del entrenamiento
    plot_training_results(trained_agent)
    
    # Preguntear si se quiere ver una demostración
    demo_choice = input("\n¿Quieres ver una demostración visual del agente entrenado? (s/n): ")
    if demo_choice.lower() == 's':
        demo_trained_agent(trained_agent)
    
    print("\n=== ENTRENAMIENTO COMPLETADO ===")
    print("El modelo ha sido guardado como 'frozen_lake_dqn_model.pth'")
    print("Puedes cargar este modelo más tarde para continuar el entrenamiento o hacer más evaluaciones.")