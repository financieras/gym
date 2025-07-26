"""
enhanced_agent.py - Agente DQN Mejorado con Recompensas Moldeadas

Este archivo contiene una versión mejorada del agente DQN que incluye:
- Recompensas moldeadas para incentivar rutas cortas
- Penalizaciones por pasos excesivos
- Bonificaciones por seguir rutas óptimas conocidas
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple

from config import *
from frozen_lake.dqn.v7.agent import QNetwork, ReplayBuffer

# Estructura para almacenar experiencias
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class EnhancedDQNAgent:
    """
    Agente DQN mejorado con recompensas moldeadas para encontrar rutas óptimas
    """
    
    def __init__(self):
        """Inicializar el agente DQN mejorado"""
        
        # Redes neuronales
        self.q_network_local = QNetwork().to(DEVICE)
        self.q_network_target = QNetwork().to(DEVICE)
        
        # Optimizador
        self.optimizer = optim.Adam(
            self.q_network_local.parameters(), 
            lr=NetworkConfig.LEARNING_RATE,
            weight_decay=NetworkConfig.WEIGHT_DECAY
        )
        
        # Buffer de experiencia
        self.memory = ReplayBuffer()
        
        # Parámetros de exploración
        self.epsilon = DQNConfig.EPSILON_START
        
        # Contador de pasos
        self.t_step = 0
        
        # Métricas
        self.losses = []
        self.q_values = []
        self.episode_steps = []  # Pasos por episodio
        self.optimal_episodes = []  # Episodios con rutas óptimas
        
        # Inicializar target network
        self.hard_update(self.q_network_local, self.q_network_target)
        
        # Mapa de distancias Manhattan para guidance
        self.distance_map = self._create_distance_map()
        
        # Contadores para análisis
        self.total_episodes = 0
        self.successful_episodes = 0
        self.optimal_solutions = 0
    
    def _create_distance_map(self):
        """Crear mapa de distancias Manhattan a la meta"""
        distance_map = {}
        goal_row, goal_col = 3, 3  # Meta en posición (3,3) = estado 15
        
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
    
    def calculate_shaped_reward(self, state, action, next_state, env_reward, done, steps):
        """
        Calcular recompensa moldeada para incentivar rutas óptimas
        
        Args:
            state (int): Estado actual
            action (int): Acción tomada
            next_state (int): Siguiente estado
            env_reward (float): Recompensa original del entorno
            done (bool): Si el episodio terminó
            steps (int): Número de pasos hasta ahora
            
        Returns:
            float: Recompensa moldeada
        """
        # Empezar con recompensa base del entorno
        reward = env_reward
        
        if done and env_reward > 0:  # Llegó a la meta exitosamente
            # Recompensa base por éxito
            base_success_reward = 10.0
            
            # Bonificación masiva por rutas óptimas
            if steps <= EvalConfig.OPTIMAL_STEPS:  # 6 pasos o menos
                optimal_bonus = 100.0
                reward = base_success_reward + optimal_bonus
                print(f"🎯 ¡RUTA ÓPTIMA! {steps} pasos - Recompensa: {reward}")
                self.optimal_solutions += 1
                self.optimal_episodes.append(self.total_episodes)
            elif steps <= 8:  # Casi óptima
                near_optimal_bonus = 50.0
                reward = base_success_reward + near_optimal_bonus
            elif steps <= 12:  # Buena
                good_bonus = 20.0
                reward = base_success_reward + good_bonus
            else:  # Regular
                # Penalización por ineficiencia
                inefficiency_penalty = min(steps - 12, 20) * 0.5
                reward = base_success_reward - inefficiency_penalty
            
            # Bonificación adicional por seguir rutas teóricas conocidas
            reward += self._check_theoretical_path_bonus(state, next_state)
            
        elif done and env_reward == 0:  # Cayó en agujero
            reward = -5.0
            
        else:  # Paso intermedio
            # Recompensa por acercarse a la meta
            current_distance = self.distance_map[state]
            next_distance = self.distance_map[next_state]
            
            if next_distance < current_distance:
                reward = 1.0  # Recompensa por progreso
            elif next_distance > current_distance:
                reward = -0.5  # Penalización por alejarse
            else:
                reward = -0.1  # Penalización pequeña por no progresar
            
            # Bonificación por estar en rutas óptimas conocidas
            if self._is_on_optimal_path(next_state):
                reward += 0.5
            
            # Penalización creciente por pasos excesivos
            if steps > 15:
                step_penalty = (steps - 15) * 0.1
                reward -= step_penalty
        
        return reward
    
    def _check_theoretical_path_bonus(self, current_state, next_state):
        """Verificar si sigue una ruta teórica conocida"""
        bonus = 0.0
        
        for path in OptimalPaths.PATHS_4X4:
            try:
                if current_state in path and next_state in path:
                    current_idx = path.index(current_state)
                    next_idx = path.index(next_state)
                    
                    # Si va en la dirección correcta en la ruta
                    if next_idx == current_idx + 1:
                        bonus = 5.0  # Bonificación por seguir ruta teórica
                        break
            except ValueError:
                continue
        
        return bonus
    
    def _is_on_optimal_path(self, state):
        """Verificar si el estado está en alguna ruta óptima conocida"""
        for path in OptimalPaths.PATHS_4X4:
            if state in path:
                return True
        return False
    
    def act(self, state, training=True):
        """Seleccionar acción usando epsilon-greedy mejorado"""
        state_tensor = torch.from_numpy(self.state_to_tensor(state)).float().unsqueeze(0).to(DEVICE)
        
        self.q_network_local.eval()
        with torch.no_grad():
            action_values = self.q_network_local(state_tensor)
        self.q_network_local.train()
        
        # Epsilon-greedy con exploración mantenida
        if training and random.random() < self.epsilon:
            # Exploración inteligente: favorecer acciones que nos acerquen a la meta
            if random.random() < 0.3:  # 30% exploración completamente aleatoria
                return random.choice(np.arange(EnvConfig.ACTION_SIZE))
            else:  # 70% exploración dirigida
                return self._get_guided_random_action(state)
        else:
            return np.argmax(action_values.cpu().data.numpy())
    
    def _get_guided_random_action(self, state):
        """Obtener acción aleatoria pero guiada hacia la meta"""
        current_distance = self.distance_map[state]
        
        # Evaluar todas las acciones posibles
        action_scores = []
        for action in range(EnvConfig.ACTION_SIZE):
            next_state = self._simulate_action(state, action)
            if next_state is not None:
                next_distance = self.distance_map[next_state]
                # Preferir acciones que nos acerquen
                score = current_distance - next_distance
                action_scores.append((action, score))
        
        if action_scores:
            # Elegir aleatoriamente entre las mejores acciones
            max_score = max(score for _, score in action_scores)
            best_actions = [action for action, score in action_scores if score == max_score]
            return random.choice(best_actions)
        else:
            return random.choice(np.arange(EnvConfig.ACTION_SIZE))
    
    def _simulate_action(self, state, action):
        """Simular el resultado de una acción (sin considerar slippery)"""
        row, col = divmod(state, 4)
        
        if action == 0:  # Izquierda
            new_col = max(0, col - 1)
        elif action == 1:  # Abajo
            new_row = min(3, row + 1)
            new_col = col
        elif action == 2:  # Derecha
            new_col = min(3, col + 1)
        elif action == 3:  # Arriba
            new_row = max(0, row - 1)
            new_col = col
        
        if action in [0, 2]:  # Movimiento horizontal
            new_row = row
        
        return new_row * 4 + new_col
    
    def step(self, state, action, reward, next_state, done, steps):
        """Paso de entrenamiento con recompensa moldeada"""
        # Calcular recompensa moldeada
        shaped_reward = self.calculate_shaped_reward(state, action, next_state, reward, done, steps)
        
        # Guardar experiencia
        self.memory.push(
            self.state_to_tensor(state),
            action,
            shaped_reward,  # Usar recompensa moldeada
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
        """Entrenar la red neuronal"""
        states, actions, rewards, next_states, dones = experiences
        
        # Valores Q objetivo usando target network
        Q_targets_next = self.q_network_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (DQNConfig.GAMMA * Q_targets_next * (1 - dones))
        
        # Valores Q esperados de la red local
        Q_expected = self.q_network_local(states).gather(1, actions)
        
        # Calcular pérdida
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Optimización
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network_local.parameters(), 1.0)
        self.optimizer.step()
        
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
        """Actualizar epsilon con decay más lento"""
        self.epsilon = max(DQNConfig.EPSILON_END, DQNConfig.EPSILON_DECAY * self.epsilon)
    
    def end_episode(self, episode_reward, steps):
        """Finalizar episodio y actualizar métricas"""
        self.total_episodes += 1
        self.episode_steps.append(steps)
        
        if episode_reward > 0:
            self.successful_episodes += 1
    
    def save(self, filepath=TrainingConfig.MODEL_PATH):
        """Guardar modelo mejorado"""
        torch.save({
            'q_network_local_state_dict': self.q_network_local.state_dict(),
            'q_network_target_state_dict': self.q_network_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'losses': self.losses,
            'q_values': self.q_values,
            'episode_steps': self.episode_steps,
            'optimal_episodes': self.optimal_episodes,
            'total_episodes': self.total_episodes,
            'successful_episodes': self.successful_episodes,
            'optimal_solutions': self.optimal_solutions
        }, filepath)
        print(f"{LogConfig.EMOJIS['SAVE']} Modelo mejorado guardado en {filepath}")
    
    def load(self, filepath=TrainingConfig.MODEL_PATH):
        """Cargar modelo mejorado"""
        try:
            checkpoint = torch.load(filepath, map_location=DEVICE)
            self.q_network_local.load_state_dict(checkpoint['q_network_local_state_dict'])
            self.q_network_target.load_state_dict(checkpoint['q_network_target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', DQNConfig.EPSILON_END)
            self.losses = checkpoint.get('losses', [])
            self.q_values = checkpoint.get('q_values', [])
            self.episode_steps = checkpoint.get('episode_steps', [])
            self.optimal_episodes = checkpoint.get('optimal_episodes', [])
            self.total_episodes = checkpoint.get('total_episodes', 0)
            self.successful_episodes = checkpoint.get('successful_episodes', 0)
            self.optimal_solutions = checkpoint.get('optimal_solutions', 0)
            print(f"{LogConfig.EMOJIS['SUCCESS']} Modelo mejorado cargado desde {filepath}")
        except FileNotFoundError:
            print(f"{LogConfig.EMOJIS['WARNING']} No se encontró el archivo {filepath}")
        except Exception as e:
            print(f"{LogConfig.EMOJIS['ERROR']} Error cargando modelo: {e}")
    
    def get_statistics(self):
        """Obtener estadísticas del entrenamiento"""
        success_rate = self.successful_episodes / max(self.total_episodes, 1)
        optimal_rate = self.optimal_solutions / max(self.successful_episodes, 1)
        avg_steps = np.mean(self.episode_steps) if self.episode_steps else 0
        
        return {
            'success_rate': success_rate,
            'optimal_rate': optimal_rate,
            'avg_steps': avg_steps,
            'total_episodes': self.total_episodes,
            'successful_episodes': self.successful_episodes,
            'optimal_solutions': self.optimal_solutions
        }

def create_enhanced_agent():
    """Crear agente DQN mejorado"""
    return EnhancedDQNAgent()

if __name__ == "__main__":
    # Prueba del agente mejorado
    agent = create_enhanced_agent()
    print("Agente DQN mejorado creado")
    print(f"Epsilon inicial: {agent.epsilon}")
    print(f"Distancia desde estado 0 a meta: {agent.distance_map[0]}")
    
    # Probar recompensa moldeada
    test_reward = agent.calculate_shaped_reward(0, 1, 4, 0, False, 1)
    print(f"Recompensa moldeada para paso hacia abajo: {test_reward}")