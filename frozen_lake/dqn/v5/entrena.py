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

class EnhancedQNetwork(nn.Module):
    """Red neuronal optimizada con regularización mejorada"""
    def __init__(self, state_size=16, action_size=4):
        super(EnhancedQNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, action_size)
        
        # Dropout para regularización
        self.dropout = nn.Dropout(0.1)
        
        # Inicialización mejorada
        self._init_weights()
        
    def _init_weights(self):
        """Inicialización Kaiming con bias positivo pequeño"""
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(layer.bias, 0.01)
        
        # Última capa con inicialización pequeña
        nn.init.uniform_(self.fc5.weight, -0.05, 0.05)
        nn.init.constant_(self.fc5.bias, 0)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        q_values = self.fc5(x)
        return q_values

class ExtendedCurriculumDQNAgent:
    """
    Agente DQN con exploración prolongada y entrenamiento extenso
    """
    def __init__(self):
        self.state_size = 16
        self.action_size = 4
        
        # Hiperparámetros para entrenamiento largo
        self.lr = 0.0005  # Learning rate más conservador
        self.gamma = 0.99  # Factor de descuento más alto para largo plazo
        
        # EPSILON CON DECAY MUY LENTO - CLAVE PARA EXPLORACIÓN PROLONGADA
        self.epsilon = 0.95  # Epsilon inicial alto
        self.epsilon_decay = 0.99995  # Decay extremadamente lento
        self.epsilon_min = 0.25  # Mínimo alto para mantener exploración
        
        # Redes neuronales más profundas
        self.q_network = EnhancedQNetwork().to(device)
        self.target_network = EnhancedQNetwork().to(device)
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=self.lr, weight_decay=1e-5)
        
        # Memoria más grande para entrenamiento largo
        self.memory = deque(maxlen=50000)
        self.optimal_memory = deque(maxlen=5000)  # Memoria especial más grande
        self.successful_memory = deque(maxlen=15000)  # Nueva: memoria para rutas exitosas
        
        self.update_target_network()
        
        # Métricas extendidas
        self.scores = []
        self.steps_to_goal = []
        self.optimal_solutions = []
        self.epsilon_history = []
        self.phase_transitions = []
        
        # RUTAS ÓPTIMAS CORREGIDAS según tu información
        self.known_optimal_paths = [
            [0, 1, 2, 6, 10, 14, 15],    # Ruta 1: Norte-Este-Sur
            [0, 4, 8, 9, 10, 14, 15],    # Ruta 2: Sur-Este-Sur  
            [0, 4, 8, 9, 13, 14, 15],    # Ruta 3: Sur-Sur-Este
        ]
        
        # Phases con mucho más tiempo para exploración
        self.phase_episodes = {
            "exploration": 3000,    # Triplicado para más exploración
            "optimization": 2000,   # Cuadruplicado para mejor optimización  
            "mastery": 1000        # Quintuplicado para consolidación
        }
        self.current_episode = 0
        
        # Contadores de rutas encontradas
        self.found_paths = set()
        self.path_frequencies = {}
        
    def get_learning_phase(self):
        """Determinar fase actual con tiempos extendidos"""
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
        """Dar pista sobre la acción óptima basada en las 3 rutas conocidas"""
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
        """Selección de acción con exploración muy prolongada"""
        phase = self.get_learning_phase()
        
        if not training:
            # En evaluación: usar red neuronal con pequeña exploración
            if random.random() < 0.05:  # 5% exploración en evaluación
                return random.randrange(self.action_size)
            state_tensor = self.state_to_tensor(state)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
        
        # Curriculum learning con exploración extendida
        if phase == "exploration":
            # Fase de exploración: hints moderados + mucha exploración
            if random.random() < 0.15:  # 15% hints (reducido para más exploración natural)
                hint = self.get_optimal_action_hint(state)
                if hint is not None:
                    return hint
            
            # Exploración alta y prolongada
            if random.random() < self.epsilon:
                return random.randrange(self.action_size)
                
        elif phase == "optimization":
            # Fase de optimización: más hints, exploración moderada
            if random.random() < 0.30:  # 30% hints
                hint = self.get_optimal_action_hint(state)
                if hint is not None:
                    return hint
            
            if random.random() < self.epsilon:
                return random.randrange(self.action_size)
                
        else:  # mastery
            # Fase de maestría: hints altos para consolidar conocimiento
            if random.random() < 0.40:  # 40% hints para consolidar
                hint = self.get_optimal_action_hint(state)
                if hint is not None:
                    return hint
            
            if random.random() < self.epsilon:
                return random.randrange(self.action_size)
        
        # Usar red neuronal
        state_tensor = self.state_to_tensor(state)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def calculate_reward(self, state, next_state, env_reward, done, steps, path):
        """Sistema de recompensas mejorado para descubrimiento de rutas"""
        reward = 0
        phase = self.get_learning_phase()
        
        if done and env_reward > 0:  # Llegó a la meta
            base_reward = 100
            
            # Recompensa masiva por soluciones óptimas
            if steps <= 6:
                reward = base_reward + 500  # Recompensa masiva
                print(f"🎯 ¡RUTA ÓPTIMA PERFECTA EN {steps} PASOS!")
                
                # Registrar ruta encontrada
                path_states = [s for s, _, _, _, _ in path] + [next_state]
                path_str = " → ".join(map(str, path_states))
                self.found_paths.add(path_str)
                self.path_frequencies[path_str] = self.path_frequencies.get(path_str, 0) + 1
                
            elif steps <= 8:
                reward = base_reward + 150
            elif steps <= 12:
                reward = base_reward + 50
            else:
                reward = base_reward + 10
            
            # Bonificación extra por rutas nuevas/poco frecuentes
            path_states = [s for s, _, _, _, _ in path] + [next_state]
            for known_path in self.known_optimal_paths:
                if path_states == known_path:
                    reward += 200  # Bonificación por ruta teórica exacta
                    print(f"🏆 ¡RUTA TEÓRICA EXACTA ENCONTRADA!")
                    break
                    
        elif done and env_reward == 0:  # Cayó en agujero
            reward = -15
            
        else:  # Paso intermedio
            # Recompensa por progreso direccional
            current_distance = self.calculate_manhattan_distance(state)
            next_distance = self.calculate_manhattan_distance(next_state)
            
            if next_distance < current_distance:
                reward = 3  # Recompensa por acercarse
            elif next_distance > current_distance:
                reward = -1  # Penalización por alejarse
            else:
                reward = -0.2  # Penalización pequeña por no progresar
            
            # Bonificación por estar en rutas conocidas
            for known_path in self.known_optimal_paths:
                if next_state in known_path:
                    reward += 1
                    break
            
            # Penalización por pasos excesivos según la fase
            if phase == "optimization" and steps > 20:
                reward -= 0.5
            elif phase == "mastery" and steps > 15:
                reward -= 1
        
        return reward
    
    def store_experience(self, state, action, reward, next_state, done):
        """Almacenar experiencia con clasificación múltiple"""
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)
        
        # Clasificar experiencias por calidad
        if reward > 100:  # Experiencias óptimas
            self.optimal_memory.append(experience)
        elif reward > 50:  # Experiencias exitosas
            self.successful_memory.append(experience)
    
    def train_step(self, batch_size=128):  # Batch más grande para estabilidad
        """Entrenamiento con muestreo adaptativo mejorado"""
        if len(self.memory) < batch_size:
            return
        
        phase = self.get_learning_phase()
        
        # Muestreo adaptativo más agresivo
        batch = []
        
        if phase == "exploration":
            # En exploración: muestreo balanceado
            batch = random.sample(self.memory, min(batch_size, len(self.memory)))
            
        elif phase == "optimization":
            # En optimización: favorecer experiencias exitosas
            optimal_ratio = 0.4
            successful_ratio = 0.3
            regular_ratio = 0.3
            
            num_optimal = int(batch_size * optimal_ratio)
            num_successful = int(batch_size * successful_ratio) 
            num_regular = batch_size - num_optimal - num_successful
            
            if len(self.optimal_memory) > 0:
                optimal_sample_size = min(num_optimal, len(self.optimal_memory))
                batch.extend(random.sample(self.optimal_memory, optimal_sample_size))
            
            if len(self.successful_memory) > 0:
                successful_sample_size = min(num_successful, len(self.successful_memory))
                batch.extend(random.sample(self.successful_memory, successful_sample_size))
            
            if len(self.memory) > 0:
                remaining = batch_size - len(batch)
                if remaining > 0:
                    regular_sample_size = min(remaining, len(self.memory))
                    batch.extend(random.sample(self.memory, regular_sample_size))
                    
        else:  # mastery
            # En maestría: maximum focus en experiencias óptimas
            optimal_ratio = 0.6
            successful_ratio = 0.3
            regular_ratio = 0.1
            
            num_optimal = int(batch_size * optimal_ratio)
            num_successful = int(batch_size * successful_ratio)
            num_regular = batch_size - num_optimal - num_successful
            
            if len(self.optimal_memory) > 0:
                optimal_sample_size = min(num_optimal, len(self.optimal_memory))
                batch.extend(random.sample(self.optimal_memory, optimal_sample_size))
            
            if len(self.successful_memory) > 0:
                successful_sample_size = min(num_successful, len(self.successful_memory))
                batch.extend(random.sample(self.successful_memory, successful_sample_size))
            
            if len(self.memory) > 0:
                remaining = batch_size - len(batch)
                if remaining > 0:
                    regular_sample_size = min(remaining, len(self.memory))
                    batch.extend(random.sample(self.memory, regular_sample_size))
        
        if not batch:
            return
        
        # Completar batch si es necesario
        while len(batch) < batch_size and len(self.memory) > 0:
            additional = random.sample(self.memory, min(batch_size - len(batch), len(self.memory)))
            batch.extend(additional)
        
        # Procesar batch
        states, actions, rewards, next_states, dones = zip(*batch[:batch_size])
        
        states = torch.FloatTensor(np.array([self._state_to_one_hot(s) for s in states])).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array([self._state_to_one_hot(s) for s in next_states])).to(device)
        dones = torch.BoolTensor(dones).to(device)
        
        # Double DQN con Huber loss
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.SmoothL1Loss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Actualizar epsilon MUY lentamente
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Registrar epsilon para análisis
        self.epsilon_history.append(self.epsilon)
    
    def _state_to_one_hot(self, state):
        """Convertir estado a one-hot"""
        one_hot = np.zeros(self.state_size)
        one_hot[state] = 1
        return one_hot
    
    def update_target_network(self):
        """Actualizar red objetivo"""
        self.target_network.load_state_dict(self.q_network.state_dict())

def train_extended_agent(episodes=6000):  # ENTRENAMIENTO MUCHO MÁS LARGO
    """Entrenamiento extendido con exploración prolongada"""
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode=None)
    agent = ExtendedCurriculumDQNAgent()
    
    print("🚀 === ENTRENAMIENTO EXTENDIDO CON EXPLORACIÓN PROLONGADA ===")
    print(f"Duración: {episodes} episodios (~{episodes/100:.0f}x más largo)")
    print("Epsilon decay: 0.99995 (extremadamente lento)")
    print("Fases: Exploración(3000) → Optimización(2000) → Maestría(1000)")
    print("Objetivo: Encontrar las 3 rutas óptimas teóricas")
    print("-" * 70)
    
    # Métricas de seguimiento
    recent_optimal = deque(maxlen=200)  # Ventana más grande
    recent_success = deque(maxlen=200)
    recent_steps = deque(maxlen=200)
    
    optimal_count = 0
    phase_optimal_counts = {"exploration": 0, "optimization": 0, "mastery": 0}
    phase_success_counts = {"exploration": 0, "optimization": 0, "mastery": 0}
    
    start_time = time.time()
    last_report_time = start_time
    
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
            reward = agent.calculate_reward(state, next_state, env_reward, done, steps, episode_path)
            
            agent.store_experience(state, action, reward, next_state, done)
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
            recent_steps.append(steps)
            phase_success_counts[current_phase] += 1
            
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
        
        # Actualizar red objetivo con menos frecuencia para estabilidad
        if episode % 50 == 0:
            agent.update_target_network()
        
        # Reportar progreso cada 200 episodios para entrenamiento largo
        if episode % 200 == 0 and episode > 0:
            current_time = time.time()
            elapsed = current_time - last_report_time
            total_elapsed = current_time - start_time
            
            success_rate = np.mean(recent_success) if recent_success else 0
            optimal_rate = np.mean(recent_optimal) if recent_optimal else 0
            avg_steps = np.mean(recent_steps) if recent_steps else 0
            
            # Calcular tiempo estimado restante
            episodes_per_second = 200 / elapsed if elapsed > 0 else 0
            remaining_episodes = episodes - episode
            eta_seconds = remaining_episodes / episodes_per_second if episodes_per_second > 0 else 0
            eta_minutes = eta_seconds / 60
            
            print(f"Ep {episode:4d} | {current_phase:12s} | "
                  f"Éxito: {success_rate:.1%} | Óptimas: {optimal_rate:.1%} | "
                  f"Pasos: {avg_steps:.1f} | ε: {agent.epsilon:.3f} | "
                  f"Total ópt: {optimal_count} | Rutas únicas: {len(agent.found_paths)} | "
                  f"ETA: {eta_minutes:.1f}min")
            
            last_report_time = current_time
        
        # Criterio de parada temprana mejorado
        if (len(recent_optimal) >= 200 and 
            np.mean(recent_optimal) >= 0.5 and  # 50% óptimas
            np.mean(recent_success) >= 0.8 and  # 80% éxito
            len(agent.found_paths) >= 3):       # Al menos 3 rutas únicas
            print(f"\n🏆 ¡MAESTRÍA COMPLETA ALCANZADA en episodio {episode}!")
            print(f"Rutas únicas encontradas: {len(agent.found_paths)}")
            break
    
    env.close()
    
    training_time = time.time() - start_time
    
    # Estadísticas detalladas por fase
    print(f"\n=== PROGRESO DETALLADO POR FASES ===")
    for phase in ["exploration", "optimization", "mastery"]:
        phase_episodes = agent.phase_episodes[phase]
        opt_count = phase_optimal_counts[phase]
        suc_count = phase_success_counts[phase]
        opt_rate = opt_count / phase_episodes if phase_episodes > 0 else 0
        suc_rate = suc_count / phase_episodes if phase_episodes > 0 else 0
        
        print(f"{phase.capitalize():12s}: {opt_count:2d} óptimas, {suc_count:3d} éxitos / {phase_episodes} eps")
        print(f"{'':14s}  Tasa óptimas: {opt_rate:.2%}, Tasa éxito: {suc_rate:.2%}")
    
    # Análisis de rutas encontradas
    print(f"\n=== RUTAS ÚNICAS DESCUBIERTAS ===")
    for i, path in enumerate(sorted(agent.found_paths), 1):
        frequency = agent.path_frequencies.get(path, 0)
        print(f"{i}. {path} (encontrada {frequency} veces)")
    
    successful_episodes = sum(agent.scores)
    print(f"\n=== ESTADÍSTICAS FINALES ===")
    print(f"Tiempo de entrenamiento: {training_time:.1f} segundos ({training_time/60:.1f} minutos)")
    print(f"Episodios completados: {episode + 1}")
    print(f"Episodios exitosos: {successful_episodes}")
    print(f"Soluciones óptimas: {optimal_count}")
    print(f"Rutas únicas encontradas: {len(agent.found_paths)}")
    print(f"Tasa de éxito final: {successful_episodes/(episode + 1):.1%}")
    if agent.steps_to_goal:
        print(f"Pasos promedio: {np.mean(agent.steps_to_goal):.1f}")
        print(f"Tasa de soluciones óptimas: {optimal_count/successful_episodes:.1%}")
    
    return agent

def comprehensive_evaluation(agent, num_tests=1000):
    """Evaluación comprehensiva con 1000 pruebas"""
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode=None)
    
    print(f"\n🔬 === EVALUACIÓN COMPREHENSIVA ({num_tests} pruebas) ===")
    
    optimal_paths = []
    successful_paths = []
    unique_optimal_paths = set()
    path_details = {}
    
    # Rutas teóricas para verificación
    theoretical_paths = [
        "0 → 1 → 2 → 6 → 10 → 14 → 15",
        "0 → 4 → 8 → 9 → 10 → 14 → 15", 
        "0 → 4 → 8 → 9 → 13 → 14 → 15"
    ]
    
    found_theoretical = {path: 0 for path in theoretical_paths}
    
    for test in range(num_tests):
        if test % 200 == 0:
            print(f"  Progreso: {test}/{num_tests} ({test/num_tests:.1%})")
            
        state, _ = env.reset()
        path = [state]
        steps = 0
        
        for step in range(20):  # Límite generoso
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
                        
                        # Registrar detalles
                        if path_str not in path_details:
                            path_details[path_str] = {"count": 0, "steps_list": []}
                        path_details[path_str]["count"] += 1
                        path_details[path_str]["steps_list"].append(steps)
                        
                        # Verificar si es teórica
                        if path_str in theoretical_paths:
                            found_theoretical[path_str] += 1
                break
    
    env.close()
    
    # Análisis de resultados
    optimal_count = len(optimal_paths)
    success_count = len(successful_paths)
    
    print(f"\n📊 === RESULTADOS FINALES ===")
    print(f"✅ Rutas exitosas: {success_count}/{num_tests} ({success_count/num_tests:.1%})")
    print(f"🎯 Rutas óptimas: {optimal_count}/{num_tests} ({optimal_count/num_tests:.1%})")
    print(f"🔄 Rutas óptimas únicas: {len(unique_optimal_paths)}")
    
    if unique_optimal_paths:
        print(f"\n🏆 RUTAS ÓPTIMAS ÚNICAS ENCONTRADAS:")
        for i, path in enumerate(sorted(unique_optimal_paths), 1):
            details = path_details[path]
            avg_steps = np.mean(details["steps_list"])
            frequency = optimal_count / len(optimal_paths) if optimal_paths else 0
            print(f"{i}. {path}")
            print(f"   Encontrada: {details['count']} veces ({details['count']/optimal_count:.1%} de óptimas)")
            print(f"   Pasos promedio: {avg_steps:.1f}")
        
        print(f"\n🧮 VERIFICACIÓN DE RUTAS TEÓRICAS:")
        theoretical_found = 0
        for i, path in enumerate(theoretical_paths, 1):
            count = found_theoretical[path]
            if count > 0:
                print(f"✅ Ruta teórica {i}: ENCONTRADA {count} veces")
                theoretical_found += 1
            else:
                print(f"❌ Ruta teórica {i}: NO encontrada")
        
        print(f"\n🎯 COMPLETITUD: {theoretical_found}/3 rutas teóricas encontradas ({theoretical_found/3:.1%})")
    
    # Distribución de eficiencia
    if successful_paths:
        steps_dist = {}
        for steps, _ in successful_paths:
            steps_dist[steps] = steps_dist.get(steps, 0) + 1
        
        print(f"\n📈 DISTRIBUCIÓN DE EFICIENCIA:")
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
    
    return optimal_count, success_count, len(unique_optimal_paths), theoretical_found

# Función principal mejorada
if __name__ == "__main__":
    print("🧊 === FROZEN LAKE DQN - ENTRENAMIENTO EXTENDIDO ===")
    print("🎯 Objetivo: Encontrar las 3 rutas óptimas teóricas")
    print("⏱️ Duración: ~10-15 minutos de entrenamiento intenso")
    print("🔍 Exploración: Epsilon decay extremadamente lento (0.99995)")
    print("📍 Rutas objetivo:")
    print("   1. 0→1→2→6→10→14→15")
    print("   2. 0→4→8→9→10→14→15") 
    print("   3. 0→4→8→9→13→14→15")
    print()
    
    # Preguntear al usuario si quiere el entrenamiento completo
    response = input("⚠️ El entrenamiento completo tomará 10-15 minutos. ¿Continuar? (s/n): ")
    if response.lower() != 's':
        print("Entrenamiento cancelado.")
        exit()
    
    print("\n🚀 Iniciando entrenamiento extendido...")
    
    # Entrenamiento extendido
    start_time = time.time()
    trained_agent = train_extended_agent(episodes=6000)
    training_time = time.time() - start_time
    
    print(f"\n⏱️ Entrenamiento completado en {training_time:.1f} segundos ({training_time/60:.1f} minutos)")
    
    # Evaluación comprehensiva
    print("\n🔬 Iniciando evaluación comprehensiva...")
    optimal_found, total_successful, unique_paths, theoretical_found = comprehensive_evaluation(trained_agent, num_tests=1000)
    
    # Evaluación final del éxito
    optimal_rate = optimal_found / 1000
    success_rate = total_successful / 1000
    
    print(f"\n🏁 === EVALUACIÓN FINAL ===")
    print(f"🎯 Rutas óptimas: {optimal_found}/1000 ({optimal_rate:.1%})")
    print(f"✅ Tasa de éxito: {success_rate:.1%}")
    print(f"🔄 Rutas únicas: {unique_paths}")
    print(f"🧮 Rutas teóricas encontradas: {theoretical_found}/3")
    
    # Evaluación del progreso
    previous_rate = 0.006  # 0.6% de la versión anterior
    improvement = optimal_rate / previous_rate if previous_rate > 0 else float('inf')
    
    print(f"\n📈 === ANÁLISIS DE MEJORA ===")
    print(f"Versión anterior: {previous_rate:.1%}")
    print(f"Versión actual: {optimal_rate:.1%}")
    print(f"Mejora: {improvement:.1f}x mejor")
    
    if optimal_rate >= 0.4:
        print("🏆 ¡EXCELENTE! Objetivo de 40% SUPERADO!")
    elif optimal_rate >= 0.2:
        print("🎯 ¡MUY BUENO! Gran progreso hacia el objetivo.")
    elif optimal_rate >= 0.1:
        print("✅ ¡BUENO! Mejora significativa.")
    elif optimal_rate > previous_rate:
        print("📈 PROGRESO POSITIVO.")
    else:
        print("⚠️ Necesita más optimización.")
    
    if theoretical_found == 3:
        print("🎯 ¡PERFECTO! Todas las rutas teóricas encontradas!")
    elif theoretical_found == 2:
        print("🎯 ¡EXCELENTE! 2/3 rutas teóricas encontradas!")
    elif theoretical_found == 1:
        print("✅ BUENO! 1/3 rutas teóricas encontrada.")
    else:
        print("⚠️ Ninguna ruta teórica encontrada.")
    
    # Guardar modelo final
    torch.save({
        'q_network_state_dict': trained_agent.q_network.state_dict(),
        'target_network_state_dict': trained_agent.target_network.state_dict(),
        'scores': trained_agent.scores,
        'steps_to_goal': trained_agent.steps_to_goal,
        'optimal_solutions': trained_agent.optimal_solutions,
        'epsilon_history': trained_agent.epsilon_history,
        'found_paths': list(trained_agent.found_paths),
        'path_frequencies': trained_agent.path_frequencies,
        'optimal_rate': optimal_rate,
        'theoretical_found': theoretical_found,
        'training_episodes': len(trained_agent.scores),
        'training_time': training_time
    }, 'frozen_lake_extended_dqn.pth')
    
    print(f"\n💾 Modelo guardado como 'frozen_lake_extended_dqn.pth'")
    print(f"📊 Modelo incluye {len(trained_agent.epsilon_history)} pasos de entrenamiento")
    print(f"🎯 Rutas únicas descubiertas: {len(trained_agent.found_paths)}")
    
    # Resumen final
    print(f"\n🎊 === RESUMEN FINAL ===")
    print(f"⏱️ Tiempo total: {training_time/60:.1f} minutos")
    print(f"🔄 Episodios: {len(trained_agent.scores)}")
    print(f"🎯 Mejora: {improvement:.1f}x desde versión anterior")
    print(f"🏆 Rutas teóricas: {theoretical_found}/3")
    print(f"📈 Tasa óptima final: {optimal_rate:.1%}")
    
    if optimal_rate >= 0.2:
        print("\n🎉 ¡MISIÓN CUMPLIDA! El agente ha aprendido a encontrar rutas óptimas de manera consistente.")
    else:
        print("\n💡 Para mejorar más, considera:")
        print("   - Aumentar episodios a 8000-10000")
        print("   - Ajustar epsilon_min a 0.3 para más exploración")
        print("   - Probar diferentes arquitecturas de red")