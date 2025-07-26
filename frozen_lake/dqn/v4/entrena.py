import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

class SmartQNetwork(nn.Module):
    """Red neuronal optimizada con mejor inicialización"""
    def __init__(self, state_size=16, action_size=4):
        super(SmartQNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
        
        # Inicialización especializada para Q-learning
        self._init_weights()
        
    def _init_weights(self):
        """Inicialización optimizada para convergencia rápida"""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(layer.bias, 0.1)
        
        # Última capa con valores pequeños para estabilidad inicial
        nn.init.uniform_(self.fc4.weight, -0.1, 0.1)
        nn.init.constant_(self.fc4.bias, 0)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        q_values = self.fc4(x)
        return q_values

class CurriculumDQNAgent:
    """
    Agente DQN con curriculum learning y rutas guiadas
    """
    def __init__(self):
        self.state_size = 16
        self.action_size = 4
        
        # Hiperparámetros optimizados
        self.lr = 0.002  # Learning rate más alto para convergencia rápida
        self.gamma = 0.95
        self.epsilon = 0.9  # Epsilon inicial más bajo
        self.epsilon_decay = 0.9992  # Decay más lento
        self.epsilon_min = 0.15  # Mínimo más alto para exploración continua
        
        # Redes neuronales
        self.q_network = SmartQNetwork().to(device)
        self.target_network = SmartQNetwork().to(device)
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=self.lr, weight_decay=1e-4)
        
        # Memoria con experiencias de alta calidad
        self.memory = deque(maxlen=20000)
        self.optimal_memory = deque(maxlen=1000)  # Memoria especial para rutas óptimas
        
        self.update_target_network()
        
        # Métricas
        self.scores = []
        self.steps_to_goal = []
        self.optimal_solutions = []
        self.learning_phase = "exploration"  # exploration -> optimization -> mastery
        
        # Rutas óptimas conocidas para guided learning
        self.known_optimal_paths = [
            [0, 4, 8, 9, 13, 14, 15],    # Ruta encontrada
            [0, 1, 2, 6, 10, 14, 15],    # Ruta alternativa posible
            [0, 4, 8, 9, 10, 14, 15],    # Otra variante
        ]
        
        # Curriculum learning parameters
        self.phase_episodes = {"exploration": 800, "optimization": 500, "mastery": 200}
        self.current_episode = 0
        
    def get_learning_phase(self):
        """Determinar la fase actual de aprendizaje"""
        if self.current_episode < self.phase_episodes["exploration"]:
            return "exploration"
        elif self.current_episode < self.phase_episodes["exploration"] + self.phase_episodes["optimization"]:
            return "optimization"
        else:
            return "mastery"
    
    def state_to_tensor(self, state):
        """Convertir estado a tensor one-hot"""
        one_hot = np.zeros(self.state_size)
        one_hot[state] = 1
        return torch.FloatTensor(one_hot).unsqueeze(0).to(device)
    
    def calculate_manhattan_distance(self, state):
        """Calcular distancia Manhattan a la meta"""
        row, col = divmod(state, 4)
        goal_row, goal_col = 3, 3
        return abs(row - goal_row) + abs(col - goal_col)
    
    def get_optimal_action_hint(self, state):
        """Dar pista sobre la acción óptima basada en rutas conocidas"""
        for path in self.known_optimal_paths:
            if state in path:
                current_idx = path.index(state)
                if current_idx < len(path) - 1:
                    next_state = path[current_idx + 1]
                    # Calcular acción necesaria
                    current_row, current_col = divmod(state, 4)
                    next_row, next_col = divmod(next_state, 4)
                    
                    if next_row > current_row:
                        return 1  # Down
                    elif next_row < current_row:
                        return 3  # Up
                    elif next_col > current_col:
                        return 2  # Right
                    elif next_col < current_col:
                        return 0  # Left
        return None
    
    def select_action(self, state, training=True):
        """Selección de acción con curriculum learning"""
        phase = self.get_learning_phase()
        
        if not training:
            # En evaluación, siempre usar la mejor acción
            state_tensor = self.state_to_tensor(state)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
        
        # Aplicar curriculum learning
        if phase == "exploration":
            # Fase de exploración: mayor randomness pero con hints ocasionales
            if random.random() < 0.2:  # 20% del tiempo usar hints
                hint = self.get_optimal_action_hint(state)
                if hint is not None:
                    return hint
            
            if random.random() < self.epsilon:
                return random.randrange(self.action_size)
                
        elif phase == "optimization":
            # Fase de optimización: balancear hints con Q-learning
            if random.random() < 0.4:  # 40% del tiempo usar hints
                hint = self.get_optimal_action_hint(state)
                if hint is not None:
                    return hint
            
            if random.random() < self.epsilon:
                return random.randrange(self.action_size)
                
        else:  # mastery
            # Fase de maestría: confiar más en la red entrenada
            if random.random() < 0.1:  # Solo 10% hints
                hint = self.get_optimal_action_hint(state)
                if hint is not None and random.random() < 0.5:
                    return hint
            
            if random.random() < self.epsilon:
                return random.randrange(self.action_size)
        
        # Usar red neuronal
        state_tensor = self.state_to_tensor(state)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def calculate_advanced_reward(self, state, next_state, env_reward, done, steps, path):
        """Sistema de recompensas multi-nivel"""
        reward = 0
        phase = self.get_learning_phase()
        
        if done and env_reward > 0:  # Llegó a la meta
            base_reward = 100
            
            # Recompensa masiva por soluciones óptimas
            if steps <= 6:
                reward = base_reward + 200  # ¡Recompensa enorme!
                print(f"🎯 ¡RUTA ÓPTIMA PERFECTA EN {steps} PASOS!")
            elif steps <= 8:
                reward = base_reward + 50
            elif steps <= 12:
                reward = base_reward + 20
            else:
                reward = base_reward
            
            # Bonificación por seguir rutas conocidas
            path_states = [s for s, _, _, _, _ in path]
            for known_path in self.known_optimal_paths:
                if self._path_similarity(path_states, known_path) > 0.7:
                    reward += 50
                    break
                    
        elif done and env_reward == 0:  # Cayó en agujero
            reward = -20
            
        else:  # Paso intermedio
            # Recompensa por progreso direccional
            current_distance = self.calculate_manhattan_distance(state)
            next_distance = self.calculate_manhattan_distance(next_state)
            
            if next_distance < current_distance:
                reward = 5  # Recompensa significativa por acercarse
            elif next_distance > current_distance:
                reward = -2  # Penalización por alejarse
            else:
                reward = -0.5  # Penalización por no progresar
            
            # Bonificación por estar en rutas conocidas
            for known_path in self.known_optimal_paths:
                if next_state in known_path:
                    reward += 2
                    break
            
            # Penalización por pasos excesivos en fases avanzadas
            if phase != "exploration" and steps > 15:
                reward -= 1
        
        return reward
    
    def _path_similarity(self, path1, path2):
        """Calcular similitud entre dos caminos"""
        if not path1 or not path2:
            return 0
        
        common_states = len(set(path1) & set(path2))
        return common_states / max(len(path1), len(path2))
    
    def store_experience(self, state, action, reward, next_state, done, is_optimal=False):
        """Almacenar experiencia con clasificación especial"""
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)
        
        # Almacenar experiencias óptimas por separado
        if is_optimal or reward > 50:
            self.optimal_memory.append(experience)
    
    def train_step(self, batch_size=64):
        """Entrenamiento con muestreo inteligente"""
        if len(self.memory) < batch_size:
            return
        
        phase = self.get_learning_phase()
        
        # Muestreo adaptativo según la fase
        if phase == "exploration":
            # Muestreo normal
            batch = random.sample(self.memory, min(batch_size, len(self.memory)))
        else:
            # Muestreo que favorece experiencias óptimas
            optimal_ratio = 0.3 if phase == "optimization" else 0.5
            num_optimal = int(batch_size * optimal_ratio)
            num_regular = batch_size - num_optimal
            
            batch = []
            if len(self.optimal_memory) > 0:
                optimal_sample_size = min(num_optimal, len(self.optimal_memory))
                batch.extend(random.sample(self.optimal_memory, optimal_sample_size))
            
            if len(self.memory) > 0:
                regular_sample_size = min(num_regular, len(self.memory))
                batch.extend(random.sample(self.memory, regular_sample_size))
        
        if not batch:
            return
        
        # Procesar batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array([self._state_to_one_hot(s) for s in states])).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array([self._state_to_one_hot(s) for s in next_states])).to(device)
        dones = torch.BoolTensor(dones).to(device)
        
        # Double DQN
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Huber loss para estabilidad
        loss = nn.SmoothL1Loss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Actualizar epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _state_to_one_hot(self, state):
        """Convertir estado a one-hot"""
        one_hot = np.zeros(self.state_size)
        one_hot[state] = 1
        return one_hot
    
    def update_target_network(self):
        """Actualizar red objetivo"""
        self.target_network.load_state_dict(self.q_network.state_dict())

def train_curriculum_agent(episodes=1500):
    """Entrenamiento con curriculum learning"""
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode=None)
    agent = CurriculumDQNAgent()
    
    print("🎓 === CURRICULUM LEARNING DQN ===")
    print("Fases: Exploración → Optimización → Maestría")
    print("Estrategia: Rutas guiadas + recompensas progresivas")
    print("-" * 60)
    
    # Métricas
    recent_optimal = deque(maxlen=100)
    recent_success = deque(maxlen=100)
    optimal_count = 0
    phase_optimal_counts = {"exploration": 0, "optimization": 0, "mastery": 0}
    
    for episode in range(episodes):
        agent.current_episode = episode
        current_phase = agent.get_learning_phase()
        
        state, _ = env.reset()
        total_env_reward = 0
        steps = 0
        episode_path = []
        
        for step in range(100):
            action = agent.select_action(state)
            next_state, env_reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            
            episode_path.append((state, action, env_reward, next_state, done))
            
            # Calcular recompensa avanzada
            reward = agent.calculate_advanced_reward(state, next_state, env_reward, done, steps, episode_path)
            
            # Determinar si es experiencia óptima
            is_optimal = (done and env_reward > 0 and steps <= 6)
            
            agent.store_experience(state, action, reward, next_state, done, is_optimal)
            agent.train_step()
            
            state = next_state
            total_env_reward += env_reward
            
            if done:
                break
        
        # Registrar métricas
        agent.scores.append(total_env_reward)
        
        if total_env_reward > 0:
            agent.steps_to_goal.append(steps)
            recent_success.append(1)
            
            if steps <= 6:
                optimal_count += 1
                phase_optimal_counts[current_phase] += 1
                recent_optimal.append(1)
                agent.optimal_solutions.append(episode)
            else:
                recent_optimal.append(0)
        else:
            recent_success.append(0)
            recent_optimal.append(0)
        
        # Actualizar red objetivo
        if episode % 20 == 0:
            agent.update_target_network()
        
        # Reportar progreso
        if episode % 100 == 0 and episode > 0:
            success_rate = np.mean(recent_success) if recent_success else 0
            optimal_rate = np.mean(recent_optimal) if recent_optimal else 0
            avg_steps = np.mean([s for s in agent.steps_to_goal[-100:] if s]) if agent.steps_to_goal else 0
            
            print(f"Episodio {episode:4d} | "
                  f"Fase: {current_phase:12s} | "
                  f"Éxito: {success_rate:.1%} | "
                  f"Óptimas: {optimal_rate:.1%} | "
                  f"Pasos: {avg_steps:.1f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Total óptimas: {optimal_count}")
        
        # Criterio de parada temprana
        if (len(recent_optimal) >= 100 and 
            np.mean(recent_optimal) >= 0.4 and  # 40% óptimas
            np.mean(recent_success) >= 0.8):    # 80% éxito
            print(f"\n🎯 ¡MAESTRÍA ALCANZADA en episodio {episode}!")
            break
    
    env.close()
    
    # Estadísticas por fase
    print(f"\n=== PROGRESO POR FASES ===")
    for phase, count in phase_optimal_counts.items():
        phase_episodes = agent.phase_episodes[phase]
        rate = count / phase_episodes if phase_episodes > 0 else 0
        print(f"{phase.capitalize():12s}: {count:2d} óptimas / {phase_episodes} episodios ({rate:.2%})")
    
    successful_episodes = sum(agent.scores)
    print(f"\n=== ESTADÍSTICAS FINALES ===")
    print(f"Episodios totales: {episode + 1}")
    print(f"Episodios exitosos: {successful_episodes}")
    print(f"Soluciones óptimas: {optimal_count}")
    print(f"Tasa de éxito: {successful_episodes/(episode + 1):.1%}")
    if agent.steps_to_goal:
        print(f"Pasos promedio: {np.mean(agent.steps_to_goal):.1f}")
        print(f"Tasa de soluciones óptimas: {optimal_count/successful_episodes:.1%}")
    
    return agent

def intensive_optimal_search(agent, num_tests=500):
    """Búsqueda intensiva de rutas óptimas"""
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode=None)
    
    print(f"\n🔍 === BÚSQUEDA INTENSIVA DE RUTAS ÓPTIMAS ({num_tests} intentos) ===")
    
    optimal_paths = []
    successful_paths = []
    unique_optimal_paths = set()
    
    for test in range(num_tests):
        state, _ = env.reset()
        path = [state]
        steps = 0
        
        for step in range(15):  # Límite estricto para forzar eficiencia
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            path.append(state)
            steps += 1
            
            if terminated or truncated:
                if reward > 0:
                    path_str = " → ".join(map(str, path))
                    successful_paths.append((steps, path_str))
                    
                    if steps <= 6:
                        optimal_paths.append((steps, path_str))
                        unique_optimal_paths.add(path_str)
                break
    
    env.close()
    
    # Análisis detallado
    optimal_count = len(optimal_paths)
    success_count = len(successful_paths)
    
    print(f"✅ Rutas exitosas: {success_count}/{num_tests} ({success_count/num_tests:.1%})")
    print(f"🎯 Rutas óptimas: {optimal_count}/{num_tests} ({optimal_count/num_tests:.1%})")
    print(f"🔄 Rutas óptimas únicas encontradas: {len(unique_optimal_paths)}")
    
    if unique_optimal_paths:
        print(f"\n🏆 RUTAS ÓPTIMAS ÚNICAS DESCUBIERTAS:")
        for i, path in enumerate(sorted(unique_optimal_paths), 1):
            count = sum(1 for _, p in optimal_paths if p == path)
            frequency = count / optimal_count if optimal_count > 0 else 0
            print(f"{i}. {path}")
            print(f"   Encontrada {count} veces ({frequency:.1%} de las rutas óptimas)")
        
        # Verificar si son realmente las rutas teóricas óptimas
        theoretical_optimal = [
            "0 → 4 → 8 → 9 → 13 → 14 → 15",
            "0 → 1 → 2 → 6 → 10 → 14 → 15", 
            "0 → 4 → 8 → 9 → 10 → 14 → 15"
        ]
        
        print(f"\n🧮 VERIFICACIÓN TEÓRICA:")
        for i, theoretical in enumerate(theoretical_optimal, 1):
            if theoretical in unique_optimal_paths:
                print(f"✅ Ruta teórica {i} ENCONTRADA: {theoretical}")
            else:
                print(f"❌ Ruta teórica {i} NO encontrada: {theoretical}")
    
    # Distribución de pasos
    if successful_paths:
        steps_dist = {}
        for steps, _ in successful_paths:
            steps_dist[steps] = steps_dist.get(steps, 0) + 1
        
        print(f"\n📊 DISTRIBUCIÓN DE EFICIENCIA:")
        for steps in sorted(steps_dist.keys()):
            count = steps_dist[steps]
            pct = count / success_count * 100
            if steps <= 6:
                icon = "🎯"
            elif steps <= 10:
                icon = "✅"
            else:
                icon = "⚠️"
            print(f"{icon} {steps:2d} pasos: {count:3d} veces ({pct:5.1f}%)")
    
    return optimal_count, success_count, len(unique_optimal_paths)

# Función principal
if __name__ == "__main__":
    print("🧊 === FROZEN LAKE CURRICULUM DQN ===")
    print("Objetivo: >40% de rutas óptimas en evaluación final")
    print("Estrategia: Curriculum Learning + Rutas Guiadas")
    print()
    
    # Entrenamiento con curriculum
    start_time = time.time()
    trained_agent = train_curriculum_agent(episodes=1500)
    training_time = time.time() - start_time
    
    print(f"\n⏱️ Tiempo de entrenamiento: {training_time:.1f} segundos")
    
    # Búsqueda intensiva
    optimal_found, total_successful, unique_paths = intensive_optimal_search(trained_agent, num_tests=500)
    
    # Evaluación final
    optimal_rate = optimal_found / 500 if 500 > 0 else 0
    success_rate = total_successful / 500 if 500 > 0 else 0
    
    print(f"\n🏁 === EVALUACIÓN FINAL ===")
    print(f"🎯 Rutas óptimas: {optimal_found}/500 ({optimal_rate:.1%})")
    print(f"✅ Tasa de éxito: {success_rate:.1%}")
    print(f"🔄 Rutas únicas: {unique_paths}")
    
    # Evaluación del éxito
    if optimal_rate >= 0.4:
        print("🏆 ¡EXCELENTE! Objetivo de 40% de rutas óptimas ALCANZADO!")
    elif optimal_rate >= 0.2:
        print("🎯 ¡BUENO! Gran mejora, cerca del objetivo.")
    elif optimal_rate >= 0.1:
        print("✅ PROGRESO SIGNIFICATIVO desde la versión anterior.")
    else:
        print("⚠️ Necesita más optimización.")
    
    # Guardar modelo mejorado
    torch.save({
        'q_network_state_dict': trained_agent.q_network.state_dict(),
        'target_network_state_dict': trained_agent.target_network.state_dict(),
        'scores': trained_agent.scores,
        'steps_to_goal': trained_agent.steps_to_goal,
        'optimal_solutions': trained_agent.optimal_solutions,
        'optimal_rate': optimal_rate
    }, 'frozen_lake_curriculum_dqn.pth')
    
    print(f"\n💾 Modelo guardado como 'frozen_lake_curriculum_dqn.pth'")
    print(f"📈 Mejora desde versión anterior: {optimal_rate:.1%} vs 1.0%")