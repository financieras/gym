"""
agent.py - Agente DQN Mejorado para Frozen Lake

Este archivo contiene la implementación del agente DQN mejorado con:
- Recompensas moldeadas para incentivar rutas óptimas
- Exploración dirigida hacia la meta
- Bonificaciones por seguir rutas teóricas conocidas
- Sistema de análisis y métricas avanzado
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple

from config import *

# Estructura para almacenar experiencias
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class QNetwork(nn.Module):
    """
    Red neuronal mejorada para aproximar la función Q(s,a)
    """
    
    def __init__(self, state_size=NetworkConfig.INPUT_SIZE, 
                 action_size=NetworkConfig.OUTPUT_SIZE, 
                 hidden_sizes=NetworkConfig.HIDDEN_SIZES):
        """Inicializar la red neuronal con arquitectura optimizada"""
        super(QNetwork, self).__init__()
        
        # Construir capas dinámicamente
        layers = []
        
        # Capa de entrada
        layers.append(nn.Linear(state_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        if NetworkConfig.DROPOUT_RATE > 0:
            layers.append(nn.Dropout(NetworkConfig.DROPOUT_RATE))
        
        # Capas ocultas
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            if NetworkConfig.DROPOUT_RATE > 0:
                layers.append(nn.Dropout(NetworkConfig.DROPOUT_RATE))
        
        # Capa de salida
        layers.append(nn.Linear(hidden_sizes[-1], action_size))
        
        self.network = nn.Sequential(*layers)
        
        # Inicialización de pesos optimizada
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializar pesos con He initialization para ReLU"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(module.bias, 0.01)
    
    def forward(self, state):
        """Propagación hacia adelante"""
        return self.network(state)

class ReplayBuffer:
    """
    Buffer de experiencia mejorado para Experience Replay
    """
    
    def __init__(self, capacity=DQNConfig.MEMORY_SIZE):
        """Inicializar buffer de experiencia"""
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)  # Para futuro priority replay
    
    def push(self, state, action, reward, next_state, done):
        """Añadir experiencia al buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
        # Prioridad basada en magnitud de recompensa (experiencias importantes)
        priority = abs(reward) + 1.0
        self.priorities.append(priority)
    
    def sample(self, batch_size):
        """Muestrear batch con ligero sesgo hacia experiencias importantes"""
        if len(self.buffer) <= batch_size:
            experiences = list(self.buffer)
        else:
            # 80% muestreo aleatorio, 20% muestreo por prioridad
            random_size = int(batch_size * 0.8)
            priority_size = batch_size - random_size
            
            # Muestreo aleatorio
            random_experiences = random.sample(self.buffer, random_size)
            
            # Muestreo por prioridad (experiencias con mayor recompensa)
            if priority_size > 0:
                priorities_array = np.array(self.priorities)
                probabilities = priorities_array / priorities_array.sum()
                priority_indices = np.random.choice(
                    len(self.buffer), 
                    size=min(priority_size, len(self.buffer)), 
                    p=probabilities,
                    replace=False
                )
                priority_experiences = [self.buffer[i] for i in priority_indices]
            else:
                priority_experiences = []
            
            experiences = random_experiences + priority_experiences
        
        # Separar componentes
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(DEVICE)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """Retornar tamaño actual del buffer"""
        return len(self.buffer)

class DQNAgent:
    """
    Agente DQN mejorado especializado en encontrar rutas óptimas
    """
    
    def __init__(self):
        """Inicializar el agente DQN mejorado"""
        
        # Redes neuronales
        self.q_network_local = QNetwork().to(DEVICE)
        self.q_network_target = QNetwork().to(DEVICE)
        
        # Optimizador con weight decay
        self.optimizer = optim.AdamW(
            self.q_network_local.parameters(), 
            lr=NetworkConfig.LEARNING_RATE,
            weight_decay=NetworkConfig.WEIGHT_DECAY
        )
        
        # Scheduler para learning rate adaptativo
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=1000, 
            gamma=0.95
        )
        
        # Buffer de experiencia mejorado
        self.memory = ReplayBuffer()
        
        # Parámetros de exploración
        self.epsilon = DQNConfig.EPSILON_START
        
        # Contadores
        self.t_step = 0
        self.episode_count = 0
        
        # Métricas de entrenamiento
        self.losses = []
        self.q_values = []
        self.episode_rewards = []
        self.episode_steps = []
        self.optimal_episodes = []
        
        # Estadísticas de rutas
        self.total_episodes = 0
        self.successful_episodes = 0
        self.optimal_solutions = 0
        self.theoretical_paths_found = {i: 0 for i in range(len(OptimalPaths.PATHS_4X4))}
        
        # Mapa de distancias para navegación inteligente
        self.distance_map = self._create_distance_map()
        
        # Inicializar target network
        self.hard_update(self.q_network_local, self.q_network_target)
    
    def _create_distance_map(self):
        """Crear mapa de distancias Manhattan a la meta"""
        distance_map = {}
        goal_row, goal_col = 3, 3
        
        for state in range(16):
            row, col = divmod(state, 4)
            distance = abs(row - goal_row) + abs(col - goal_col)
            distance_map[state] = distance
        
        return distance_map
    
    def state_to_tensor(self, state):
        """Convertir estado entero a tensor one-hot"""
        one_hot = np.zeros(EnvConfig.STATE_SIZE)
        one_hot[state] = 1.0
        return one_hot
    
    def calculate_shaped_reward(self, state, action, next_state, env_reward, done, steps, full_path=None):
        """
        Sistema de recompensas moldeadas para incentivar rutas óptimas
        
        Args:
            state (int): Estado actual
            action (int): Acción tomada
            next_state (int): Siguiente estado
            env_reward (float): Recompensa original del entorno
            done (bool): Si el episodio terminó
            steps (int): Número de pasos hasta ahora
            full_path (list): Camino completo (opcional)
            
        Returns:
            float: Recompensa moldeada
        """
        reward = env_reward  # Empezar con recompensa base
        
        if done and env_reward > 0:  # Llegó a la meta exitosamente
            # Recompensa base por éxito
            reward = RewardConfig.BASE_SUCCESS_REWARD
            
            # Bonificaciones masivas por eficiencia
            if steps <= EvalConfig.OPTIMAL_STEPS:  # Ruta óptima (≤6 pasos)
                reward += RewardConfig.OPTIMAL_BONUS
                self.optimal_solutions += 1
                self.optimal_episodes.append(self.total_episodes)
                print(f"{LogConfig.EMOJIS['OPTIMAL']} ¡RUTA ÓPTIMA! {steps} pasos - Recompensa: {reward}")
                
                # Verificar si es ruta teórica exacta
                if full_path:
                    theoretical_index = OptimalPaths.check_path_match(full_path)
                    if theoretical_index is not None:
                        reward += RewardConfig.THEORETICAL_PATH_BONUS
                        self.theoretical_paths_found[theoretical_index] += 1
                        print(f"{LogConfig.EMOJIS['THEORETICAL']} ¡RUTA TEÓRICA {theoretical_index + 1}!")
                        
            elif steps <= 8:  # Casi óptima
                reward += RewardConfig.NEAR_OPTIMAL_BONUS
            elif steps <= 12:  # Buena
                reward += RewardConfig.GOOD_BONUS
            else:  # Regular con penalización por ineficiencia
                inefficiency_penalty = min((steps - 12) * 0.5, 10)
                reward -= inefficiency_penalty
            
        elif done and env_reward == 0:  # Cayó en agujero
            reward = RewardConfig.HOLE_PENALTY
            
        else:  # Paso intermedio
            # Recompensa por progreso hacia la meta
            current_distance = self.distance_map[state]
            next_distance = self.distance_map[next_state]
            
            if next_distance < current_distance:
                reward = RewardConfig.PROGRESS_REWARD  # Progreso hacia meta
            elif next_distance > current_distance:
                reward = RewardConfig.REGRESS_PENALTY  # Alejarse de meta
            else:
                reward = RewardConfig.NO_PROGRESS_PENALTY  # Sin progreso
            
            # Bonificación por estar en rutas óptimas conocidas
            if OptimalPaths.is_on_optimal_path(next_state):
                reward += RewardConfig.OPTIMAL_PATH_STEP_BONUS
            
            # Penalización creciente por pasos excesivos
            if steps > RewardConfig.STEP_PENALTY_THRESHOLD:
                step_penalty = (steps - RewardConfig.STEP_PENALTY_THRESHOLD) * RewardConfig.STEP_PENALTY_RATE
                reward -= step_penalty
        
        return reward
    
    def act(self, state, training=True):
        """
        Seleccionar acción usando epsilon-greedy mejorado con exploración dirigida
        """
        state_tensor = torch.from_numpy(self.state_to_tensor(state)).float().unsqueeze(0).to(DEVICE)
        
        # Obtener valores Q
        self.q_network_local.eval()
        with torch.no_grad():
            action_values = self.q_network_local(state_tensor)
        self.q_network_local.train()
        
        # Epsilon-greedy con exploración inteligente
        if training and random.random() < self.epsilon:
            # 30% exploración completamente aleatoria, 70% dirigida
            if random.random() < 0.3:
                return random.choice(np.arange(EnvConfig.ACTION_SIZE))
            else:
                return self._get_guided_action(state)
        else:
            return np.argmax(action_values.cpu().data.numpy())
    
    def _get_guided_action(self, state):
        """
        Obtener acción guiada hacia la meta o siguiendo rutas óptimas
        """
        # Primero, intentar seguir una ruta óptima conocida
        for path in OptimalPaths.PATHS_4X4:
            optimal_action = OptimalPaths.get_optimal_actions(state, path)
            if optimal_action is not None:
                # 60% de probabilidad de seguir la ruta óptima
                if random.random() < 0.6:
                    return optimal_action
        
        # Si no está en ruta óptima, moverse hacia la meta
        return self._get_direction_to_goal(state)
    
    def _get_direction_to_goal(self, state):
        """Obtener acción que nos acerque más a la meta"""
        current_distance = self.distance_map[state]
        best_actions = []
        best_distance = current_distance
        
        # Evaluar todas las acciones posibles
        for action in range(EnvConfig.ACTION_SIZE):
            next_state = self._simulate_action(state, action)
            if next_state is not None:
                next_distance = self.distance_map[next_state]
                if next_distance < best_distance:
                    best_distance = next_distance
                    best_actions = [action]
                elif next_distance == best_distance:
                    best_actions.append(action)
        
        # Si ninguna acción mejora, elegir aleatoriamente
        if not best_actions:
            best_actions = list(range(EnvConfig.ACTION_SIZE))
        
        return random.choice(best_actions)
    
    def _simulate_action(self, state, action):
        """Simular el resultado de una acción (aproximado)"""
        row, col = divmod(state, 4)
        
        if action == 0:  # Izquierda
            new_row, new_col = row, max(0, col - 1)
        elif action == 1:  # Abajo
            new_row, new_col = min(3, row + 1), col
        elif action == 2:  # Derecha
            new_row, new_col = row, min(3, col + 1)
        elif action == 3:  # Arriba
            new_row, new_col = max(0, row - 1), col
        else:
            return None
        
        return new_row * 4 + new_col
    
    def step(self, state, action, reward, next_state, done, steps, full_path=None):
        """
        Paso de entrenamiento con recompensa moldeada
        """
        # Calcular recompensa moldeada
        shaped_reward = self.calculate_shaped_reward(
            state, action, next_state, reward, done, steps, full_path
        )
        
        # Almacenar experiencia
        self.memory.push(
            self.state_to_tensor(state),
            action,
            shaped_reward,
            self.state_to_tensor(next_state),
            done
        )
        
        # Entrenar cada UPDATE_EVERY pasos
        self.t_step = (self.t_step + 1) % DQNConfig.UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > DQNConfig.MIN_MEMORY_SIZE:
                experiences = self.memory.sample(NetworkConfig.BATCH_SIZE)
                self.learn(experiences)
    
    def learn(self, experiences):
        """
        Actualizar parámetros usando un batch de experiencias (Double DQN)
        """
        states, actions, rewards, next_states, dones = experiences
        
        # Double DQN: usar red local para seleccionar acciones, target para evaluar
        with torch.no_grad():
            # Seleccionar mejores acciones con red local
            next_actions = self.q_network_local(next_states).argmax(1).unsqueeze(1)
            # Evaluar esas acciones con red target
            Q_targets_next = self.q_network_target(next_states).gather(1, next_actions)
            # Calcular valores objetivo
            Q_targets = rewards + (DQNConfig.GAMMA * Q_targets_next * (1 - dones))
        
        # Obtener valores Q esperados de la red local
        Q_expected = self.q_network_local(states).gather(1, actions)
        
        # Calcular pérdida (Huber loss para mayor estabilidad)
        loss = F.smooth_l1_loss(Q_expected, Q_targets)
        
        # Optimización
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping para estabilidad
        torch.nn.utils.clip_grad_norm_(self.q_network_local.parameters(), 1.0)
        self.optimizer.step()
        
        # Actualizar learning rate
        self.scheduler.step()
        
        # Guardar métricas
        self.losses.append(loss.item())
        self.q_values.append(Q_expected.mean().item())
        
        # Soft update de target network
        self.soft_update(self.q_network_local, self.q_network_target, DQNConfig.TAU)
    
    def soft_update(self, local_model, target_model, tau):
        """Actualización suave de target network"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def hard_update(self, local_model, target_model):
        """Actualización completa de target network"""
        target_model.load_state_dict(local_model.state_dict())
    
    def update_epsilon(self):
        """Actualizar epsilon con decay más gradual"""
        self.epsilon = max(DQNConfig.EPSILON_END, DQNConfig.EPSILON_DECAY * self.epsilon)
    
    def end_episode(self, episode_reward, steps):
        """Finalizar episodio y actualizar estadísticas"""
        self.total_episodes += 1
        self.episode_rewards.append(episode_reward)
        self.episode_steps.append(steps)
        
        if episode_reward > 0:
            self.successful_episodes += 1
    
    def get_statistics(self):
        """Obtener estadísticas completas del entrenamiento"""
        if self.total_episodes == 0:
            return {
                'success_rate': 0.0,
                'optimal_rate': 0.0,
                'avg_steps': 0.0,
                'total_episodes': 0,
                'successful_episodes': 0,
                'optimal_solutions': 0,
                'theoretical_paths_found': 0
            }
        
        success_rate = self.successful_episodes / self.total_episodes
        optimal_rate = self.optimal_solutions / max(self.successful_episodes, 1)
        avg_steps = np.mean([steps for i, steps in enumerate(self.episode_steps) 
                           if self.episode_rewards[i] > 0]) if self.successful_episodes > 0 else 0
        theoretical_found = sum(1 for count in self.theoretical_paths_found.values() if count > 0)
        
        return {
            'success_rate': success_rate,
            'optimal_rate': optimal_rate,
            'avg_steps': avg_steps,
            'total_episodes': self.total_episodes,
            'successful_episodes': self.successful_episodes,
            'optimal_solutions': self.optimal_solutions,
            'theoretical_paths_found': theoretical_found
        }
    
    def save(self, filepath=TrainingConfig.MODEL_PATH):
        """Guardar modelo completo con todas las métricas"""
        torch.save({
            'q_network_local_state_dict': self.q_network_local.state_dict(),
            'q_network_target_state_dict': self.q_network_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'losses': self.losses,
            'q_values': self.q_values,
            'episode_rewards': self.episode_rewards,
            'episode_steps': self.episode_steps,
            'optimal_episodes': self.optimal_episodes,
            'total_episodes': self.total_episodes,
            'successful_episodes': self.successful_episodes,
            'optimal_solutions': self.optimal_solutions,
            'theoretical_paths_found': self.theoretical_paths_found,
            'distance_map': self.distance_map
        }, filepath)
        print(f"{LogConfig.EMOJIS['SAVE']} Modelo guardado en {filepath}")
    
    def load(self, filepath=TrainingConfig.MODEL_PATH):
        """Cargar modelo completo"""
        try:
            checkpoint = torch.load(filepath, map_location=DEVICE)
            self.q_network_local.load_state_dict(checkpoint['q_network_local_state_dict'])
            self.q_network_target.load_state_dict(checkpoint['q_network_target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.epsilon = checkpoint.get('epsilon', DQNConfig.EPSILON_END)
            self.episode_count = checkpoint.get('episode_count', 0)
            self.losses = checkpoint.get('losses', [])
            self.q_values = checkpoint.get('q_values', [])
            self.episode_rewards = checkpoint.get('episode_rewards', [])
            self.episode_steps = checkpoint.get('episode_steps', [])
            self.optimal_episodes = checkpoint.get('optimal_episodes', [])
            self.total_episodes = checkpoint.get('total_episodes', 0)
            self.successful_episodes = checkpoint.get('successful_episodes', 0)
            self.optimal_solutions = checkpoint.get('optimal_solutions', 0)
            self.theoretical_paths_found = checkpoint.get('theoretical_paths_found', 
                                                        {i: 0 for i in range(len(OptimalPaths.PATHS_4X4))})
            
            print(f"{LogConfig.EMOJIS['SUCCESS']} Modelo cargado desde {filepath}")
            
            # Mostrar estadísticas del modelo cargado
            stats = self.get_statistics()
            print(f"  Episodios entrenados: {stats['total_episodes']}")
            print(f"  Tasa de éxito: {stats['success_rate']:.1%}")
            print(f"  Rutas óptimas: {stats['optimal_solutions']}")
            print(f"  Rutas teóricas encontradas: {stats['theoretical_paths_found']}/3")
            
        except FileNotFoundError:
            print(f"{LogConfig.EMOJIS['WARNING']} No se encontró el archivo {filepath}")
        except Exception as e:
            print(f"{LogConfig.EMOJIS['ERROR']} Error cargando modelo: {e}")
    
    def analyze_performance(self):
        """Analizar rendimiento detallado del agente"""
        if not self.episode_rewards:
            print("No hay datos de entrenamiento para analizar")
            return
        
        stats = self.get_statistics()
        
        print(f"\n{LogConfig.EMOJIS['INFO']} === ANÁLISIS DE RENDIMIENTO ===")
        print(f"Episodios totales: {stats['total_episodes']}")
        print(f"Episodios exitosos: {stats['successful_episodes']} ({stats['success_rate']:.1%})")
        print(f"Rutas óptimas: {stats['optimal_solutions']} ({stats['optimal_rate']:.1%} de éxitos)")
        print(f"Pasos promedio: {stats['avg_steps']:.1f}")
        
        # Análisis temporal del aprendizaje
        if len(self.episode_rewards) >= 1000:
            early_success = sum(1 for r in self.episode_rewards[:1000] if r > 0) / 1000
            if len(self.episode_rewards) >= 2000:
                late_success = sum(1 for r in self.episode_rewards[-1000:] if r > 0) / 1000
                improvement = late_success - early_success
                print(f"Mejora temporal: {early_success:.1%} → {late_success:.1%} ({improvement:+.1%})")
        
        # Análisis de rutas teóricas
        print(f"\nRutas teóricas encontradas:")
        for i, count in self.theoretical_paths_found.items():
            path_str = " → ".join(map(str, OptimalPaths.PATHS_4X4[i]))
            status = "✅" if count > 0 else "❌"
            print(f"  {status} Ruta {i+1}: {path_str} ({count} veces)")
        
        # Evolución de epsilon
        if hasattr(self, 'epsilon'):
            print(f"\nEpsilon actual: {self.epsilon:.3f}")
        
        # Análisis de pérdidas
        if self.losses:
            recent_loss = np.mean(self.losses[-100:]) if len(self.losses) >= 100 else np.mean(self.losses)
            print(f"Pérdida promedio reciente: {recent_loss:.4f}")
    
    def get_action_probabilities(self, state):
        """Obtener probabilidades de acción para análisis"""
        state_tensor = torch.from_numpy(self.state_to_tensor(state)).float().unsqueeze(0).to(DEVICE)
        
        self.q_network_local.eval()
        with torch.no_grad():
            q_values = self.q_network_local(state_tensor)
            probabilities = F.softmax(q_values, dim=1)
        self.q_network_local.train()
        
        return q_values.cpu().data.numpy().flatten(), probabilities.cpu().data.numpy().flatten()
    
    def reset_statistics(self):
        """Reiniciar todas las estadísticas (útil para nueva evaluación)"""
        self.total_episodes = 0
        self.successful_episodes = 0
        self.optimal_solutions = 0
        self.theoretical_paths_found = {i: 0 for i in range(len(OptimalPaths.PATHS_4X4))}
        self.episode_rewards = []
        self.episode_steps = []
        self.optimal_episodes = []

def create_agent():
    """
    Crear y retornar un nuevo agente DQN mejorado
    
    Returns:
        DQNAgent: Agente DQN inicializado
    """
    return DQNAgent()

if __name__ == "__main__":
    # Crear y probar agente
    agent = create_agent()
    print(f"Agente DQN mejorado creado")
    print(f"Epsilon inicial: {agent.epsilon}")
    print(f"Memoria: {len(agent.memory)}/{DQNConfig.MEMORY_SIZE}")
    
    # Probar funciones básicas
    test_state = 0
    test_action = agent.act(test_state)
    q_values, probabilities = agent.get_action_probabilities(test_state)
    
    print(f"\nPrueba en estado {test_state}:")
    print(f"  Acción seleccionada: {test_action}")
    print(f"  Valores Q: {q_values}")
    print(f"  Probabilidades: {probabilities}")
    
    # Probar recompensa moldeada
    test_reward = agent.calculate_shaped_reward(0, 1, 4, 0, False, 1)
    print(f"  Recompensa moldeada (0→4): {test_reward}")
    
    # Mostrar distancias a la meta
    print(f"\nDistancias a la meta desde diferentes estados:")
    for state in [0, 1, 4, 5, 10, 15]:
        distance = agent.distance_map[state]
        row, col = divmod(state, 4)
        print(f"  Estado {state} ({row},{col}): {distance} pasos")