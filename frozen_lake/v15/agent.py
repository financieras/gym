"""
agent.py - Agente Q-Learning con Tabla Q para Frozen Lake

Este archivo implementa Q-Learning clásico con tabla Q en lugar de redes neuronales.
Para el espacio pequeño de Frozen Lake (16 estados), esto debería ser más efectivo
y resolver los problemas de generalización que teníamos con DQN.
"""

import numpy as np
import random
from collections import defaultdict
import pickle

from config import *

class QLearningAgent:
    """
    Agente Q-Learning con tabla Q y recompensas moldeadas
    
    Ventajas sobre DQN para espacios pequeños:
    - Convergencia garantizada
    - Sin problemas de generalización
    - Memoria perfecta de todas las experiencias
    - Interpretabilidad completa
    """
    
    def __init__(self):
        """Inicializar agente Q-Learning"""
        
        # Tabla Q: Q[estado][accion] = valor
        self.q_table = np.zeros((EnvConfig.STATE_SIZE, EnvConfig.ACTION_SIZE))
        
        # Parámetros de aprendizaje
        self.learning_rate = QLearningConfig.LEARNING_RATE
        self.discount_factor = QLearningConfig.DISCOUNT_FACTOR
        self.epsilon = QLearningConfig.EPSILON_START
        
        # Estadísticas y métricas
        self.episode_count = 0
        self.total_episodes = 0
        self.successful_episodes = 0
        self.optimal_solutions = 0
        
        # Métricas detalladas
        self.episode_rewards = []
        self.episode_steps = []
        self.optimal_episodes = []
        self.theoretical_paths_found = {i: 0 for i in range(len(OptimalPaths.PATHS_4X4))}
        
        # Contadores para análisis
        self.state_visits = np.zeros(EnvConfig.STATE_SIZE)  # Conteo de visitas por estado
        self.action_counts = np.zeros((EnvConfig.STATE_SIZE, EnvConfig.ACTION_SIZE))
        
        # Métricas de convergencia
        self.q_value_changes = []
        self.policy_changes = []
        
        print(f"{LogConfig.EMOJIS['QTABLE']} Agente Q-Learning inicializado")
        print(f"Tabla Q: {EnvConfig.STATE_SIZE} estados × {EnvConfig.ACTION_SIZE} acciones")
    
    def get_state_value(self, state):
        """Obtener el valor del estado (máximo valor Q)"""
        return np.max(self.q_table[state])
    
    def get_best_action(self, state):
        """Obtener la mejor acción para un estado"""
        return np.argmax(self.q_table[state])
    
    def get_action_probabilities(self, state):
        """Obtener probabilidades de acción usando softmax"""
        q_values = self.q_table[state]
        # Evitar overflow con normalización
        q_values_norm = q_values - np.max(q_values)
        exp_values = np.exp(q_values_norm)
        probabilities = exp_values / np.sum(exp_values)
        return probabilities
    
    def select_action(self, state, training=True):
        """
        Seleccionar acción usando epsilon-greedy mejorado
        
        Args:
            state (int): Estado actual
            training (bool): Si estamos en modo entrenamiento
            
        Returns:
            int: Acción seleccionada
        """
        # En evaluación, usar epsilon muy bajo
        effective_epsilon = self.epsilon if training else 0.01
        
        if random.random() < effective_epsilon:
            # Exploración inteligente
            if training and random.random() < 0.4:
                # 40% del tiempo usar acción óptima conocida si disponible
                optimal_action = OptimalPaths.get_optimal_action(state)
                if optimal_action is not None:
                    return optimal_action
            
            # 60% del tiempo o si no hay acción óptima: exploración dirigida
            return self._get_exploration_action(state)
        else:
            # Explotación: usar la mejor acción de la tabla Q
            return self.get_best_action(state)
    
    def _get_exploration_action(self, state):
        """
        Exploración dirigida hacia la meta y estados poco visitados
        """
        current_distance = get_manhattan_distance(state)
        
        # Evaluar todas las acciones
        action_scores = []
        for action in range(EnvConfig.ACTION_SIZE):
            next_state = self._simulate_action(state, action)
            if next_state is not None:
                # Puntuación basada en progreso + exploración
                next_distance = get_manhattan_distance(next_state)
                progress_score = current_distance - next_distance  # Positivo si nos acerca
                
                # Bonificación por explorar estados poco visitados
                exploration_score = 1.0 / (1.0 + self.state_visits[next_state])
                
                # Bonificación por estar en rutas óptimas
                optimal_score = 1.0 if OptimalPaths.is_on_optimal_path(next_state) else 0.0
                
                total_score = progress_score + 0.5 * exploration_score + 0.3 * optimal_score
                action_scores.append((action, total_score))
        
        if action_scores:
            # Selección estocástica basada en puntuaciones
            actions, scores = zip(*action_scores)
            scores = np.array(scores)
            
            # Convertir puntuaciones a probabilidades
            if np.max(scores) > np.min(scores):
                scores = scores - np.min(scores)  # Normalizar a valores positivos
                scores = scores / np.sum(scores)  # Convertir a probabilidades
            else:
                scores = np.ones(len(scores)) / len(scores)  # Uniforme si todas iguales
            
            return np.random.choice(actions, p=scores)
        else:
            return random.choice(range(EnvConfig.ACTION_SIZE))
    
    def _simulate_action(self, state, action):
        """Simular el resultado de una acción (determinístico)"""
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
    
    def calculate_reward(self, state, action, next_state, env_reward, done, steps, episode_path=None):
        """
        Calcular recompensa moldeada para Q-Learning
        
        Args:
            state (int): Estado actual
            action (int): Acción tomada
            next_state (int): Siguiente estado
            env_reward (float): Recompensa del entorno
            done (bool): Si el episodio terminó
            steps (int): Número de pasos
            episode_path (list): Camino completo del episodio
            
        Returns:
            float: Recompensa moldeada
        """
        if done and env_reward > 0:  # Llegó a la meta
            reward = RewardConfig.SUCCESS_REWARD
            
            # Bonificación masiva por rutas óptimas
            if steps <= EvalConfig.OPTIMAL_STEPS:
                reward += RewardConfig.OPTIMAL_BONUS
                self.optimal_solutions += 1
                self.optimal_episodes.append(self.total_episodes)
                
                print(f"{LogConfig.EMOJIS['OPTIMAL']} ¡RUTA ÓPTIMA! {steps} pasos")
                
                # Verificar si es ruta teórica exacta
                if episode_path is not None:
                    theoretical_index = OptimalPaths.is_optimal_path(episode_path)
                    if theoretical_index is not None:
                        reward += RewardConfig.THEORETICAL_BONUS
                        self.theoretical_paths_found[theoretical_index] += 1
                        print(f"{LogConfig.EMOJIS['THEORETICAL']} ¡RUTA TEÓRICA {theoretical_index + 1}!")
            
            return reward
            
        elif done and env_reward == 0:  # Cayó en agujero
            return RewardConfig.HOLE_PENALTY
            
        else:  # Paso intermedio
            reward = RewardConfig.STEP_PENALTY  # Penalización base por paso
            
            # Recompensa por progreso hacia la meta
            current_distance = get_manhattan_distance(state)
            next_distance = get_manhattan_distance(next_state)
            
            if next_distance < current_distance:
                reward += RewardConfig.PROGRESS_REWARD
            elif next_distance > current_distance:
                reward += RewardConfig.REGRESS_PENALTY
            
            # Bonificación por estar en rutas óptimas
            if OptimalPaths.is_on_optimal_path(next_state):
                reward += 0.2
            
            # Bonificación por exploración (estados poco visitados)
            if self.state_visits[next_state] < 10:
                reward += RewardConfig.EXPLORATION_BONUS
            
            return reward
    
    def update_q_table(self, state, action, reward, next_state, done):
        """
        Actualizar tabla Q usando la ecuación de Q-Learning
        
        Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        """
        # Guardar valor Q anterior para análisis de convergencia
        old_q_value = self.q_table[state, action]
        
        # Calcular valor objetivo
        if done:
            target = reward  # No hay estado futuro
        else:
            max_next_q = np.max(self.q_table[next_state])
            target = reward + self.discount_factor * max_next_q
        
        # Actualizar tabla Q
        self.q_table[state, action] += self.learning_rate * (target - old_q_value)
        
        # Registrar cambio en valor Q para análisis de convergencia
        q_change = abs(self.q_table[state, action] - old_q_value)
        self.q_value_changes.append(q_change)
        
        # Actualizar contadores
        self.state_visits[state] += 1
        self.action_counts[state, action] += 1
    
    def train_step(self, state, action, reward, next_state, done, steps, episode_path=None):
        """
        Paso completo de entrenamiento
        """
        # Calcular recompensa moldeada
        shaped_reward = self.calculate_reward(state, action, next_state, reward, done, steps, episode_path)
        
        # Actualizar tabla Q
        self.update_q_table(state, action, shaped_reward, next_state, done)
        
        # Actualizar parámetros de aprendizaje
        self.update_parameters()
    
    def update_parameters(self):
        """Actualizar parámetros de aprendizaje"""
        # Decaer epsilon gradualmente
        if self.epsilon > QLearningConfig.EPSILON_END:
            self.epsilon *= QLearningConfig.EPSILON_DECAY
        
        # Decaer learning rate gradualmente
        if self.learning_rate > QLearningConfig.MIN_LEARNING_RATE:
            self.learning_rate *= QLearningConfig.LR_DECAY
    
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
                'success_rate': 0.0, 'optimal_rate': 0.0, 'avg_steps': 0.0,
                'total_episodes': 0, 'successful_episodes': 0, 'optimal_solutions': 0,
                'theoretical_paths_found': 0, 'epsilon': self.epsilon, 'learning_rate': self.learning_rate
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
            'theoretical_paths_found': theoretical_found,
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate
        }
    
    def analyze_convergence(self):
        """Analizar convergencia del algoritmo"""
        if len(self.q_value_changes) == 0:
            return False
        
        # Verificar si los cambios en Q son pequeños (convergencia)
        recent_changes = self.q_value_changes[-1000:] if len(self.q_value_changes) >= 1000 else self.q_value_changes
        avg_change = np.mean(recent_changes)
        
        return avg_change < 0.01  # Umbral de convergencia
    
    def get_policy(self):
        """Obtener política óptima (mejor acción para cada estado)"""
        policy = {}
        action_names = ['←', '↓', '→', '↑']
        
        for state in range(EnvConfig.STATE_SIZE):
            best_action = self.get_best_action(state)
            policy[state] = {
                'action': best_action,
                'action_name': action_names[best_action],
                'value': self.get_state_value(state),
                'q_values': self.q_table[state].copy()
            }
        
        return policy
    
    def print_q_table(self):
        """Imprimir tabla Q formateada"""
        print(f"\n{LogConfig.EMOJIS['QTABLE']} TABLA Q:")
        print("Estado | ←      ↓      →      ↑    | Mejor | Valor")
        print("-" * 55)
        
        action_names = ['←', '↓', '→', '↑']
        for state in range(EnvConfig.STATE_SIZE):
            q_values = self.q_table[state]
            best_action = np.argmax(q_values)
            state_value = np.max(q_values)
            
            row, col = divmod(state, 4)
            q_str = " ".join([f"{q:6.2f}" for q in q_values])
            
            print(f"{state:2d}({row},{col}) | {q_str} | {action_names[best_action]:4s}  | {state_value:6.2f}")
    
    def save(self, filepath=None):
        """Guardar agente completo"""
        if filepath is None:
            filepath = TrainingConfig.MODEL_PATH
        
        data = {
            'q_table': self.q_table,
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon,
            'total_episodes': self.total_episodes,
            'successful_episodes': self.successful_episodes,
            'optimal_solutions': self.optimal_solutions,
            'episode_rewards': self.episode_rewards,
            'episode_steps': self.episode_steps,
            'optimal_episodes': self.optimal_episodes,
            'theoretical_paths_found': self.theoretical_paths_found,
            'state_visits': self.state_visits,
            'action_counts': self.action_counts,
            'q_value_changes': self.q_value_changes[-10000:]  # Guardar solo los últimos 10k
        }
        
        np.save(filepath, data)
        print(f"{LogConfig.EMOJIS['SAVE']} Agente Q-Learning guardado en {filepath}")
    
    def load(self, filepath=None):
        """Cargar agente previamente entrenado"""
        if filepath is None:
            filepath = TrainingConfig.MODEL_PATH
        
        try:
            data = np.load(filepath, allow_pickle=True).item()
            
            self.q_table = data['q_table']
            self.learning_rate = data.get('learning_rate', QLearningConfig.LEARNING_RATE)
            self.epsilon = data.get('epsilon', QLearningConfig.EPSILON_END)
            self.total_episodes = data.get('total_episodes', 0)
            self.successful_episodes = data.get('successful_episodes', 0)
            self.optimal_solutions = data.get('optimal_solutions', 0)
            self.episode_rewards = data.get('episode_rewards', [])
            self.episode_steps = data.get('episode_steps', [])
            self.optimal_episodes = data.get('optimal_episodes', [])
            self.theoretical_paths_found = data.get('theoretical_paths_found', 
                                                  {i: 0 for i in range(len(OptimalPaths.PATHS_4X4))})
            self.state_visits = data.get('state_visits', np.zeros(EnvConfig.STATE_SIZE))
            self.action_counts = data.get('action_counts', np.zeros((EnvConfig.STATE_SIZE, EnvConfig.ACTION_SIZE)))
            self.q_value_changes = data.get('q_value_changes', [])
            
            print(f"{LogConfig.EMOJIS['SUCCESS']} Agente Q-Learning cargado desde {filepath}")
            
            # Mostrar estadísticas del modelo cargado
            stats = self.get_statistics()
            print(f"  Episodios entrenados: {stats['total_episodes']}")
            print(f"  Tasa de éxito: {stats['success_rate']:.1%}")
            print(f"  Rutas óptimas: {stats['optimal_solutions']}")
            print(f"  Rutas teóricas encontradas: {stats['theoretical_paths_found']}/3")
            print(f"  Epsilon actual: {stats['epsilon']:.3f}")
            print(f"  Learning rate actual: {stats['learning_rate']:.4f}")
            
        except FileNotFoundError:
            print(f"{LogConfig.EMOJIS['WARNING']} No se encontró el archivo {filepath}")
        except Exception as e:
            print(f"{LogConfig.EMOJIS['ERROR']} Error cargando agente: {e}")
    
    def reset_statistics(self):
        """Reiniciar estadísticas para nueva evaluación"""
        self.total_episodes = 0
        self.successful_episodes = 0
        self.optimal_solutions = 0
        self.theoretical_paths_found = {i: 0 for i in range(len(OptimalPaths.PATHS_4X4))}
        self.episode_rewards = []
        self.episode_steps = []
        self.optimal_episodes = []

def create_agent():
    """
    Crear y retornar un nuevo agente Q-Learning
    
    Returns:
        QLearningAgent: Agente Q-Learning inicializado
    """
    return QLearningAgent()

if __name__ == "__main__":
    # Crear y probar agente
    agent = create_agent()
    
    print(f"\nPrueba del agente Q-Learning:")
    print(f"Tabla Q inicializada: {agent.q_table.shape}")
    print(f"Epsilon inicial: {agent.epsilon}")
    print(f"Learning rate inicial: {agent.learning_rate}")
    
    # Probar funciones básicas
    test_state = 0
    test_action = agent.select_action(test_state)
    state_value = agent.get_state_value(test_state)
    action_probs = agent.get_action_probabilities(test_state)
    
    print(f"\nPrueba en estado {test_state}:")
    print(f"  Acción seleccionada: {test_action}")
    print(f"  Valor del estado: {state_value:.3f}")
    print(f"  Probabilidades de acción: {action_probs}")
    print(f"  Valores Q: {agent.q_table[test_state]}")
    
    # Probar recompensa moldeada
    test_reward = agent.calculate_reward(0, 1, 4, 0, False, 1)
    print(f"  Recompensa moldeada (0→4): {test_reward}")
    
    # Mostrar distancias a la meta
    print(f"\nDistancias Manhattan a la meta:")
    for state in [0, 1, 4, 5, 10, 15]:
        distance = get_manhattan_distance(state)
        row, col = divmod(state, 4)
        on_optimal = OptimalPaths.is_on_optimal_path(state)
        print(f"  Estado {state} ({row},{col}): {distance} pasos {'(en ruta óptima)' if on_optimal else ''}")
    
    print(f"\n{LogConfig.EMOJIS['INFO']} Agente Q-Learning listo para entrenamiento")