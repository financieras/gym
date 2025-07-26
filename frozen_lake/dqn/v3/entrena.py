import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import time

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

class SimpleQNetwork(nn.Module):
    """
    Red neuronal simplificada y enfocada para Frozen Lake
    Para 16 estados, una red pequeña es más efectiva
    """
    def __init__(self, state_size=16, action_size=4):
        super(SimpleQNetwork, self).__init__()
        
        # Red más simple y eficiente
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)
        
        # Inicialización Xavier para mejor convergencia
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

class OptimalPathDQNAgent:
    """
    Agente DQN especializado en encontrar rutas óptimas
    """
    def __init__(self, state_size=16, action_size=4):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hiperparámetros optimizados para rutas cortas
        self.lr = 0.001
        self.gamma = 0.95  # Reducido para priorizar recompensas inmediatas
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.1  # Mayor exploración mantenida
        
        # Redes neuronales simples
        self.q_network = SimpleQNetwork(state_size, action_size).to(device)
        self.target_network = SimpleQNetwork(state_size, action_size).to(device)
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        
        # Buffer más pequeño para mejor uso de experiencias relevantes
        self.memory = deque(maxlen=10000)
        
        self.update_target_network()
        
        # Métricas
        self.scores = []
        self.steps_to_goal = []
        self.optimal_solutions = []
        self.episode_rewards = []
        
        # Mapa de distancias Manhattan a la meta (para recompensa dirigida)
        self.distance_map = self._create_distance_map()
    
    def _create_distance_map(self):
        """Crear mapa de distancias Manhattan a la meta (posición 15)"""
        distance_map = {}
        goal_row, goal_col = 3, 3
        
        for state in range(16):
            row, col = divmod(state, 4)
            distance = abs(row - goal_row) + abs(col - goal_col)
            distance_map[state] = distance
        
        return distance_map
    
    def state_to_tensor(self, state):
        """Convertir estado a tensor one-hot"""
        one_hot = np.zeros(self.state_size)
        one_hot[state] = 1
        return torch.FloatTensor(one_hot).unsqueeze(0).to(device)
    
    def select_action(self, state, training=True):
        """Selección de acción con epsilon-greedy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = self.state_to_tensor(state)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return q_values.argmax().item()
    
    def calculate_reward(self, state, next_state, env_reward, done, steps):
        """
        Sistema de recompensas fuertemente dirigido hacia rutas óptimas
        """
        reward = 0
        
        if done and env_reward > 0:  # Llegó a la meta
            # Recompensa base por éxito
            reward = 10.0
            
            # Bonificación masiva por soluciones óptimas
            if steps <= 6:
                reward += 50.0  # Recompensa muy alta por óptimo
                print(f"¡SOLUCIÓN ÓPTIMA EN {steps} PASOS!")
            elif steps <= 8:
                reward += 20.0  # Buena recompensa por casi-óptimo
            elif steps <= 12:
                reward += 10.0  # Recompensa moderada
            else:
                reward += 5.0   # Recompensa mínima por llegar
                
        elif done and env_reward == 0:  # Cayó en agujero
            reward = -10.0
            
        else:  # Paso normal
            # Recompensa por acercarse a la meta
            current_distance = self.distance_map[state]
            next_distance = self.distance_map[next_state]
            
            if next_distance < current_distance:
                reward = 2.0  # Recompensa significativa por acercarse
            elif next_distance > current_distance:
                reward = -1.0  # Penalización por alejarse
            else:
                reward = -0.1  # Pequeña penalización por no progresar
        
        return reward
    
    def store_experience(self, state, action, reward, next_state, done):
        """Almacenar experiencia"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self, batch_size=32):
        """Entrenamiento con enfoque en experiencias exitosas"""
        if len(self.memory) < batch_size:
            return
        
        # Muestreo con sesgo hacia experiencias exitosas
        batch = self._prioritized_sample(batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Conversión eficiente a tensores
        states = torch.FloatTensor(np.array([self._state_to_one_hot(s) for s in states])).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array([self._state_to_one_hot(s) for s in next_states])).to(device)
        dones = torch.BoolTensor(dones).to(device)
        
        # Calcular valores Q
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Pérdida y optimización
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Actualizar epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _prioritized_sample(self, batch_size):
        """Muestreo que favorece experiencias con recompensas altas"""
        if len(self.memory) < batch_size * 2:
            return random.sample(self.memory, min(batch_size, len(self.memory)))
        
        # Separar experiencias por tipo
        positive_experiences = [exp for exp in self.memory if exp[2] > 0]
        other_experiences = [exp for exp in self.memory if exp[2] <= 0]
        
        # Tomar más experiencias positivas si están disponibles
        num_positive = min(batch_size // 2, len(positive_experiences))
        num_other = batch_size - num_positive
        
        sample = []
        if positive_experiences:
            sample.extend(random.sample(positive_experiences, num_positive))
        if other_experiences and num_other > 0:
            sample.extend(random.sample(other_experiences, min(num_other, len(other_experiences))))
        
        # Completar con experiencias aleatorias si es necesario
        while len(sample) < batch_size and len(self.memory) > len(sample):
            remaining = random.sample(self.memory, batch_size - len(sample))
            sample.extend(remaining)
        
        return sample[:batch_size]
    
    def _state_to_one_hot(self, state):
        """Convertir estado a one-hot"""
        one_hot = np.zeros(self.state_size)
        one_hot[state] = 1
        return one_hot
    
    def update_target_network(self):
        """Actualizar red objetivo"""
        self.target_network.load_state_dict(self.q_network.state_dict())

def train_optimal_agent(episodes=1500):
    """
    Entrenamiento enfocado en encontrar rutas óptimas
    """
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode=None)
    agent = OptimalPathDQNAgent()
    
    print("=== ENTRENAMIENTO ENFOCADO EN RUTAS ÓPTIMAS ===")
    print("Objetivo: Maximizar soluciones en 6 pasos")
    print("Estrategia: Recompensas fuertemente dirigidas + muestreo prioritario")
    print("-" * 60)
    
    # Métricas de seguimiento
    recent_optimal = deque(maxlen=100)
    recent_success = deque(maxlen=100)
    recent_steps = deque(maxlen=100)
    
    optimal_count = 0
    total_optimal = 0
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_env_reward = 0
        steps = 0
        episode_done = False
        
        for step in range(100):  # Límite de pasos por episodio
            action = agent.select_action(state)
            next_state, env_reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            
            # Calcular recompensa moldeada
            shaped_reward = agent.calculate_reward(state, next_state, env_reward, done, steps)
            
            # Almacenar experiencia
            agent.store_experience(state, action, shaped_reward, next_state, done)
            
            # Entrenar
            agent.train_step()
            
            state = next_state
            total_env_reward += env_reward
            
            if done:
                episode_done = True
                break
        
        # Registrar métricas
        agent.scores.append(total_env_reward)
        
        if total_env_reward > 0:  # Éxito
            agent.steps_to_goal.append(steps)
            recent_success.append(1)
            recent_steps.append(steps)
            
            if steps <= 6:  # Solución óptima
                optimal_count += 1
                total_optimal += 1
                recent_optimal.append(1)
                agent.optimal_solutions.append(episode)
            else:
                recent_optimal.append(0)
        else:
            recent_success.append(0)
            recent_optimal.append(0)
        
        # Actualizar red objetivo
        if episode % 25 == 0:
            agent.update_target_network()
        
        # Reporte de progreso
        if episode % 100 == 0 and episode > 0:
            success_rate = np.mean(recent_success) if recent_success else 0
            optimal_rate = np.mean(recent_optimal) if recent_optimal else 0
            avg_steps = np.mean(recent_steps) if recent_steps else 0
            
            print(f"Episodio {episode:4d} | "
                  f"Éxito: {success_rate:.1%} | "
                  f"Óptimas: {optimal_rate:.1%} | "
                  f"Pasos prom: {avg_steps:.1f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Total óptimas: {total_optimal}")
        
        # Criterio de parada temprana
        if (len(recent_optimal) >= 100 and 
            np.mean(recent_optimal) >= 0.3 and  # 30% de soluciones óptimas
            np.mean(recent_success) >= 0.8):    # 80% de éxito
            print(f"\n¡OBJETIVO ALCANZADO en episodio {episode}!")
            print(f"Tasa de soluciones óptimas: {np.mean(recent_optimal):.1%}")
            print(f"Tasa de éxito: {np.mean(recent_success):.1%}")
            break
    
    env.close()
    
    # Estadísticas finales
    successful_episodes = sum(agent.scores)
    print(f"\n=== ESTADÍSTICAS FINALES ===")
    print(f"Episodios totales: {episode + 1}")
    print(f"Episodios exitosos: {successful_episodes}")
    print(f"Soluciones óptimas totales: {total_optimal}")
    
    if agent.steps_to_goal:
        print(f"Tasa de éxito: {successful_episodes/(episode + 1):.1%}")
        print(f"Pasos promedio: {np.mean(agent.steps_to_goal):.1f}")
        print(f"Mínimo pasos: {min(agent.steps_to_goal)}")
        print(f"Tasa de soluciones óptimas: {total_optimal/successful_episodes:.1%}")
    
    return agent

def test_optimal_paths(agent, num_tests=50):
    """
    Prueba exhaustiva para encontrar todas las rutas óptimas
    """
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode=None)
    
    print(f"\n=== BÚSQUEDA DE RUTAS ÓPTIMAS ({num_tests} intentos) ===")
    
    optimal_paths = []
    all_successful_paths = []
    path_counts = {}
    
    for test in range(num_tests):
        state, _ = env.reset()
        path = [state]
        steps = 0
        
        for step in range(20):  # Límite razonable
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            path.append(state)
            steps += 1
            
            if terminated or truncated:
                if reward > 0:  # Éxito
                    path_str = " → ".join(map(str, path))
                    all_successful_paths.append((steps, path_str))
                    
                    if steps <= 6:  # Óptimo
                        optimal_paths.append((steps, path_str))
                        path_counts[path_str] = path_counts.get(path_str, 0) + 1
                break
    
    env.close()
    
    # Mostrar resultados
    print(f"Rutas exitosas encontradas: {len(all_successful_paths)}")
    print(f"Rutas óptimas encontradas: {len(optimal_paths)}")
    
    if optimal_paths:
        print("\n🎯 RUTAS ÓPTIMAS DESCUBIERTAS:")
        unique_optimal = {}
        for steps, path in optimal_paths:
            if path not in unique_optimal:
                unique_optimal[path] = []
            unique_optimal[path].append(steps)
        
        for i, (path, step_list) in enumerate(unique_optimal.items(), 1):
            avg_steps = np.mean(step_list)
            count = len(step_list)
            print(f"{i}. {path} (promedio: {avg_steps:.1f} pasos, encontrado {count} veces)")
    
    # Distribución de pasos
    if all_successful_paths:
        steps_distribution = {}
        for steps, _ in all_successful_paths:
            steps_distribution[steps] = steps_distribution.get(steps, 0) + 1
        
        print(f"\n📊 DISTRIBUCIÓN DE PASOS:")
        for steps in sorted(steps_distribution.keys()):
            count = steps_distribution[steps]
            percentage = count / len(all_successful_paths) * 100
            marker = "🎯" if steps <= 6 else "✅" if steps <= 10 else "⚠️"
            print(f"{marker} {steps} pasos: {count} veces ({percentage:.1f}%)")
    
    return len(optimal_paths), len(all_successful_paths)

def demonstrate_best_agent(agent, num_demos=5):
    """
    Demostración de las mejores ejecuciones del agente
    """
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode=None)
    
    print("\n=== DEMOSTRACIÓN DE MEJORES EJECUCIONES ===")
    print("Mapa de Frozen Lake 4x4:")
    print("┌─────┬─────┬─────┬─────┐")
    print("│  S  │  F  │  F  │  F  │  Fila 0")
    print("├─────┼─────┼─────┼─────┤")
    print("│  F  │  H  │  F  │  H  │  Fila 1")
    print("├─────┼─────┼─────┼─────┤")
    print("│  F  │  F  │  F  │  H  │  Fila 2")
    print("├─────┼─────┼─────┼─────┤")
    print("│  H  │  F  │  F  │  G  │  Fila 3")
    print("└─────┴─────┴─────┴─────┘")
    print("S=Start(0), F=Frozen, H=Hole, G=Goal(15)")
    print("-" * 50)
    
    successful_demos = []
    
    # Recolectar demostraciones exitosas
    for _ in range(num_demos * 3):  # Intentar más para encontrar buenas demos
        state, _ = env.reset()
        path = [state]
        actions_taken = []
        steps = 0
        
        for step in range(25):
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            actions_taken.append(action)
            path.append(next_state)
            steps += 1
            state = next_state
            
            if terminated or truncated:
                if reward > 0:  # Éxito
                    successful_demos.append((steps, path, actions_taken, reward))
                break
    
    env.close()
    
    # Ordenar por número de pasos (mejores primero)
    successful_demos.sort(key=lambda x: x[0])
    
    # Mostrar las mejores demos
    actions_names = ['←', '↓', '→', '↑']
    
    for i, (steps, path, actions, reward) in enumerate(successful_demos[:num_demos]):
        print(f"\n🎮 Demo {i+1}: {'🎯 ÓPTIMO' if steps <= 6 else '✅ EXITOSO'} ({steps} pasos)")
        
        action_sequence = " → ".join([actions_names[a] for a in actions])
        path_sequence = " → ".join([f"{p}({p//4},{p%4})" for p in path])
        
        print(f"   Acciones: {action_sequence}")
        print(f"   Camino: {path_sequence}")
        print(f"   Recompensa: {reward}")
        
        # Verificar si es una ruta conocida óptima
        if steps <= 6:
            print("   🏆 ¡Esta es una solución óptima!")
    
    if not successful_demos:
        print("❌ No se encontraron demostraciones exitosas en esta ejecución.")
        print("   Esto puede indicar que el agente necesita más entrenamiento.")
    
    return len([d for d in successful_demos if d[0] <= 6])

# Función principal optimizada
if __name__ == "__main__":
    print("🧊 === FROZEN LAKE DQN - BÚSQUEDA DE RUTAS ÓPTIMAS ===")
    print("Objetivo: Encontrar consistentemente rutas de 6 pasos a la meta")
    print("Estrategia: Recompensas dirigidas + muestreo prioritario")
    print()
    
    # Entrenar agente especializado
    start_time = time.time()
    trained_agent = train_optimal_agent(episodes=1500)
    training_time = time.time() - start_time
    
    print(f"\n⏱️ Tiempo de entrenamiento: {training_time:.1f} segundos")
    
    # Buscar rutas óptimas
    optimal_found, total_successful = test_optimal_paths(trained_agent, num_tests=100)
    
    # Demostrar mejores ejecuciones
    optimal_demos = demonstrate_best_agent(trained_agent, num_demos=5)
    
    # Resumen final
    print(f"\n🏁 === RESUMEN FINAL ===")
    print(f"Rutas óptimas encontradas en pruebas: {optimal_found}/100")
    print(f"Tasa de éxito en pruebas: {total_successful/100:.1%}")
    print(f"Demostraciones óptimas: {optimal_demos}/5")
    
    if optimal_found > 0:
        print("🎯 ¡El agente ha aprendido a encontrar rutas óptimas!")
    elif total_successful > 50:
        print("✅ El agente encuentra la meta consistentemente, pero puede mejorar la eficiencia.")
    else:
        print("⚠️ El agente necesita más entrenamiento para mejores resultados.")
    
    # Guardar modelo
    torch.save({
        'q_network_state_dict': trained_agent.q_network.state_dict(),
        'target_network_state_dict': trained_agent.target_network.state_dict(),
        'scores': trained_agent.scores,
        'steps_to_goal': trained_agent.steps_to_goal,
        'optimal_solutions': trained_agent.optimal_solutions
    }, 'frozen_lake_optimal_dqn.pth')
    
    print("\n💾 Modelo guardado como 'frozen_lake_optimal_dqn.pth'")