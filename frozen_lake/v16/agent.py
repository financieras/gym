"""
agent.py - Agente Q-Learning OPTIMIZADO con Tabla Q para Frozen Lake

Esta versión optimizada incluye mejoras en la exploración, sistema de recompensas
y estrategias para aumentar la frecuencia de rutas óptimas basadas en el análisis
de los resultados de entrenamiento.
"""

import numpy as np
import random
from collections import defaultdict
import pickle

from config import *

class QLearningAgent:
    """
    Agente Q-Learning optimizado con tabla Q y recompensas moldeadas
    
    Mejoras implementadas:
    - Exploración más inteligente hacia rutas óptimas
    - Sistema de recompensas amplificado
    - Análisis de progreso en rutas teóricas
    - Bonificaciones de eficiencia graduadas
    """
    
    def __init__(self):
        """Inicializar agente Q-Learning optimizado"""
        
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
        
        # Contadores para análisis optimizado
        self.state_visits = np.zeros(EnvConfig.STATE_SIZE)
        self.action_counts = np.zeros((EnvConfig.STATE_SIZE, EnvConfig.ACTION_SIZE))
        self.optimal_action_counts = np.zeros((EnvConfig.STATE_SIZE, EnvConfig.ACTION_SIZE))  # Nuevo
        
        # Métricas de convergencia mejoradas
        self.q_value_changes = []
        self.policy_changes = []
        self.efficiency_history = []  # Nuevo: historial de eficiencia
        
        # Exploración inteligente
        self.exploration_temperature = 2.0  # Nuevo: para exploración softmax
        self.optimal_bias = 0.3  # Nuevo: sesgo hacia rutas óptimas
        
        print(f"{LogConfig.EMOJIS['QTABLE']} Agente Q-Learning OPTIMIZADO inicializado")
        print(f"Tabla Q: {EnvConfig.STATE_SIZE} estados × {EnvConfig.ACTION_SIZE} acciones")
        print(f"Optimizaciones: Exploración inteligente + Recompensas amplificadas")
    
    def get_state_value(self, state):
        """Obtener el valor del estado (máximo valor Q)"""
        return np.max(self.q_table[state])
    
    def get_best_action(self, state):
        """Obtener la mejor acción para un estado"""
        return np.argmax(self.q_table[state])
    
    def get_action_probabilities(self, state, temperature=None):
        """Obtener probabilidades de acción usando softmax con temperatura"""
        if temperature is None:
            temperature = self.exploration_temperature
            
        q_values = self.q_table[state]
        # Evitar overflow con normalización
        q_values_norm = q_values - np.max(q_values)
        exp_values = np.exp(q_values_norm / temperature)
        probabilities = exp_values / np.sum(exp_values)
        return probabilities
    
    def select_action(self, state, training=True):
        """
        Seleccionar acción usando exploración optimizada
        
        Args:
            state (int): Estado actual
            training (bool): Si estamos en modo entrenamiento
            
        Returns:
            int: Acción seleccionada
        """
        # En evaluación, usar política pura con pequeña exploración
        if not training:
            if random.random() < 0.02:  # 2% de exploración en evaluación
                return self._get_optimal_biased_action(state)
            return self.get_best_action(state)
        
        # Exploración adaptativa durante entrenamiento
        effective_epsilon = self._get_adaptive_epsilon(state)
        
        if random.random() < effective_epsilon:
            # Exploración inteligente con múltiples estrategias
            return self._get_intelligent_exploration_action(state)
        else:
            # Explotación: mejor acción aprendida
            return self.get_best_action(state)
    
    def _get_adaptive_epsilon(self, state):
        """Calcular epsilon adaptativo basado en el estado y progreso"""
        base_epsilon = self.epsilon
        
        # Reducir exploración en estados bien conocidos
        if self.state_visits[state] > 50:
            base_epsilon *= 0.5
        
        # Aumentar exploración en estados de rutas óptimas poco explorados
        if OptimalPaths.is_on_optimal_path(state) and self.state_visits[state] < 20:
            base_epsilon *= 1.5
        
        return min(base_epsilon, 0.8)  # Limitar epsilon máximo
    
    def _get_intelligent_exploration_action(self, state):
        """
        Exploración inteligente con múltiples estrategias
        """
        exploration_strategies = []
        
        # Estrategia 1: Acción óptima conocida (40% del tiempo)
        optimal_action = OptimalPaths.get_optimal_action(state)
        if optimal_action is not None:
            exploration_strategies.extend([optimal_action] * 4)
        
        # Estrategia 2: Exploración dirigida hacia progreso (30% del tiempo)
        progress_action = self._get_progress_action(state)
        if progress_action is not None:
            exploration_strategies.extend([progress_action] * 3)
        
        # Estrategia 3: Exploración softmax de Q-values (20% del tiempo)
        softmax_action = self._get_softmax_action(state)
        exploration_strategies.extend([softmax_action] * 2)
        
        # Estrategia 4: Exploración pura aleatoria (10% del tiempo)
        random_action = random.choice(range(EnvConfig.ACTION_SIZE))
        exploration_strategies.append(random_action)
        
        return random.choice(exploration_strategies)
    
    def _get_progress_action(self, state):
        """Obtener acción que hace progreso hacia la meta"""
        current_distance = get_manhattan_distance(state)
        best_action = None
        best_score = float('-inf')
        
        for action in range(EnvConfig.ACTION_SIZE):
            next_state = self._simulate_action(state, action)
            if next_state is not None:
                next_distance = get_manhattan_distance(next_state)
                
                # Puntuación basada en progreso + valor Q + exploración
                progress_score = current_distance - next_distance
                q_score = self.q_table[next_state].max() * 0.1
                exploration_score = 1.0 / (1.0 + self.state_visits[next_state])
                optimal_bonus = 2.0 if OptimalPaths.is_on_optimal_path(next_state) else 0.0
                
                total_score = progress_score + q_score + exploration_score + optimal_bonus
                
                if total_score > best_score:
                    best_score = total_score
                    best_action = action
        
        return best_action
    
    def _get_softmax_action(self, state):
        """Selección de acción usando softmax sobre Q-values"""
        probabilities = self.get_action_probabilities(state, temperature=self.exploration_temperature)
        return np.random.choice(range(EnvConfig.ACTION_SIZE), p=probabilities)
    
    def _get_optimal_biased_action(self, state):
        """Acción con sesgo hacia rutas óptimas (para evaluación)"""
        optimal_action = OptimalPaths.get_optimal_action(state)
        if optimal_action is not None and random.random() < self.optimal_bias:
            return optimal_action
        return self.get_best_action(state)
    
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
        Calcular recompensa moldeada optimizada para Q-Learning
        
        Args:
            state (int): Estado actual
            action (int): Acción tomada
            next_state (int): Siguiente estado
            env_reward (float): Recompensa del entorno
            done (bool): Si el episodio terminó
            steps (int): Número de pasos
            episode_path (list): Camino completo del episodio
            
        Returns:
            float: Recompensa moldeada optimizada
        """
        if done and env_reward > 0:  # Llegó a la meta
            reward = RewardConfig.SUCCESS_REWARD
            
            # Bonificación masiva por rutas óptimas
            if steps <= EvalConfig.OPTIMAL_STEPS:
                reward += RewardConfig.OPTIMAL_BONUS
                self.optimal_solutions += 1
                self.optimal_episodes.append(self.total_episodes)
                
                #print(f"{LogConfig.EMOJIS['OPTIMAL']} ¡RUTA ÓPTIMA! {steps} pasos")
                
                # Verificar si es ruta teórica exacta
                if episode_path is not None:
                    theoretical_index = OptimalPaths.is_optimal_path(episode_path)
                    if theoretical_index is not None:
                        reward += RewardConfig.THEORETICAL_BONUS
                        self.theoretical_paths_found[theoretical_index] += 1
                        #print(f"{LogConfig.EMOJIS['THEORETICAL']} ¡RUTA TEÓRICA {theoretical_index + 1}!")
            
            # Bonificación de eficiencia graduada
            efficiency_bonus = calculate_efficiency_bonus(steps)
            reward += efficiency_bonus
            
            # Registrar eficiencia
            efficiency = EvalConfig.OPTIMAL_STEPS / steps if steps > 0 else 0
            self.efficiency_history.append(efficiency)
            
            return reward
            
        elif done and env_reward == 0:  # Cayó en agujero
            return RewardConfig.HOLE_PENALTY
            
        else:  # Paso intermedio
            reward = RewardConfig.STEP_PENALTY
            
            # Recompensa por progreso hacia la meta
            current_distance = get_manhattan_distance(state)
            next_distance = get_manhattan_distance(next_state)
            
            if next_distance < current_distance:
                reward += RewardConfig.PROGRESS_REWARD
            elif next_distance > current_distance:
                reward += RewardConfig.REGRESS_PENALTY
            
            # Bonificación amplificada por estar en rutas óptimas
            if OptimalPaths.is_on_optimal_path(next_state):
                reward += RewardConfig.OPTIMAL_PATH_BONUS
                
                # Bonificación adicional por progreso en ruta óptima
                progress = OptimalPaths.get_path_progress(next_state)
                reward += progress * 2.0
            
            # Bonificación por exploración mejorada
            if self.state_visits[next_state] < 10:
                reward += RewardConfig.EXPLORATION_BONUS
            
            return reward
    
    def update_q_table(self, state, action, reward, next_state, done):
        """
        Actualizar tabla Q usando la ecuación de Q-Learning optimizada
        """
        # Guardar valor Q anterior para análisis de convergencia
        old_q_value = self.q_table[state, action]
        
        # Calcular valor objetivo
        if done:
            target = reward
        else:
            max_next_q = np.max(self.q_table[next_state])
            target = reward + self.discount_factor * max_next_q
        
        # Actualizar tabla Q con learning rate adaptativo
        adaptive_lr = self._get_adaptive_learning_rate(state, action)
        self.q_table[state, action] += adaptive_lr * (target - old_q_value)
        
        # Registrar cambio en valor Q para análisis de convergencia
        q_change = abs(self.q_table[state, action] - old_q_value)
        self.q_value_changes.append(q_change)
        
        # Actualizar contadores
        self.state_visits[state] += 1
        self.action_counts[state, action] += 1
        
        # Registrar acciones óptimas
        optimal_action = OptimalPaths.get_optimal_action(state)
        if optimal_action == action:
            self.optimal_action_counts[state, action] += 1
    
    def _get_adaptive_learning_rate(self, state, action):
        """Calcular learning rate adaptativo"""
        base_lr = self.learning_rate
        
        # Aumentar learning rate para estados poco visitados
        if self.state_visits[state] < 10:
            base_lr *= 1.5
        
        # Reducir learning rate para estados bien conocidos
        elif self.state_visits[state] > 100:
            base_lr *= 0.7
        
        return min(base_lr, 0.3)  # Limitar learning rate máximo
    
    def train_step(self, state, action, reward, next_state, done, steps, episode_path=None):
        """
        Paso completo de entrenamiento optimizado
        """
        # Calcular recompensa moldeada
        shaped_reward = self.calculate_reward(state, action, next_state, reward, done, steps, episode_path)
        
        # Actualizar tabla Q
        self.update_q_table(state, action, shaped_reward, next_state, done)
        
        # Actualizar parámetros de aprendizaje
        self.update_parameters()
    
    def update_parameters(self):
        """Actualizar parámetros de aprendizaje optimizados"""
        # Decaer epsilon gradualmente
        if self.epsilon > QLearningConfig.EPSILON_END:
            self.epsilon *= QLearningConfig.EPSILON_DECAY
        
        # Decaer learning rate gradualmente
        if self.learning_rate > QLearningConfig.MIN_LEARNING_RATE:
            self.learning_rate *= QLearningConfig.LR_DECAY
        
        # Actualizar temperatura de exploración
        if self.exploration_temperature > 0.5:
            self.exploration_temperature *= 0.9999
    
    def end_episode(self, episode_reward, steps):
        """Finalizar episodio y actualizar estadísticas"""
        self.total_episodes += 1
        self.episode_rewards.append(episode_reward)
        self.episode_steps.append(steps)
        
        if episode_reward > 0:
            self.successful_episodes += 1
    
    def get_statistics(self):
        """Obtener estadísticas completas del entrenamiento optimizado"""
        if self.total_episodes == 0:
            return {
                'success_rate': 0.0, 'optimal_rate': 0.0, 'avg_steps': 0.0,
                'total_episodes': 0, 'successful_episodes': 0, 'optimal_solutions': 0,
                'theoretical_paths_found': 0, 'epsilon': self.epsilon, 'learning_rate': self.learning_rate,
                'avg_efficiency': 0.0, 'exploration_temp': self.exploration_temperature
            }
        
        success_rate = self.successful_episodes / self.total_episodes
        optimal_rate = self.optimal_solutions / max(self.successful_episodes, 1)
        avg_steps = np.mean([steps for i, steps in enumerate(self.episode_steps) 
                           if self.episode_rewards[i] > 0]) if self.successful_episodes > 0 else 0
        theoretical_found = sum(1 for count in self.theoretical_paths_found.values() if count > 0)
        avg_efficiency = np.mean(self.efficiency_history) if self.efficiency_history else 0
        
        return {
            'success_rate': success_rate,
            'optimal_rate': optimal_rate,
            'avg_steps': avg_steps,
            'total_episodes': self.total_episodes,
            'successful_episodes': self.successful_episodes,
            'optimal_solutions': self.optimal_solutions,
            'theoretical_paths_found': theoretical_found,
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'avg_efficiency': avg_efficiency,
            'exploration_temp': self.exploration_temperature
        }
    
    def analyze_convergence(self):
        """Analizar convergencia del algoritmo optimizado"""
        if len(self.q_value_changes) == 0:
            return False
        
        # Verificar si los cambios en Q son pequeños (convergencia)
        recent_changes = self.q_value_changes[-2000:] if len(self.q_value_changes) >= 2000 else self.q_value_changes
        avg_change = np.mean(recent_changes)
        
        # Criterio de convergencia más estricto
        is_converged = avg_change < 0.005
        
        # Verificar también estabilidad en eficiencia
        if len(self.efficiency_history) >= 50:
            recent_efficiency = self.efficiency_history[-50:]
            efficiency_std = np.std(recent_efficiency)
            is_converged = is_converged and efficiency_std < 0.1
        
        return is_converged
    
    def get_policy(self):
        """Obtener política óptima mejorada con análisis"""
        policy = {}
        action_names = ['←', '↓', '→', '↑']
        
        for state in range(EnvConfig.STATE_SIZE):
            best_action = self.get_best_action(state)
            state_value = self.get_state_value(state)
            
            # Análisis adicional
            visits = int(self.state_visits[state])
            is_optimal_state = OptimalPaths.is_on_optimal_path(state)
            optimal_action = OptimalPaths.get_optimal_action(state)
            is_optimal_policy = (best_action == optimal_action) if optimal_action is not None else None
            
            policy[state] = {
                'action': best_action,
                'action_name': action_names[best_action],
                'value': state_value,
                'q_values': self.q_table[state].copy(),
                'visits': visits,
                'is_optimal_state': is_optimal_state,
                'is_optimal_policy': is_optimal_policy,
                'confidence': min(visits / 50.0, 1.0)  # Confianza basada en visitas
            }
        
        return policy
    
    def print_q_table(self, show_analysis=True):
        """Imprimir tabla Q formateada con análisis opcional"""
        print(f"\n{LogConfig.EMOJIS['QTABLE']} TABLA Q OPTIMIZADA:")
        
        if show_analysis:
            print("Estado | ←      ↓      →      ↑    | Mejor | Valor | Análisis")
            print("-" * 75)
        else:
            print("Estado | ←      ↓      →      ↑    | Mejor | Valor")
            print("-" * 55)
        
        action_names = ['←', '↓', '→', '↑']
        for state in range(EnvConfig.STATE_SIZE):
            q_values = self.q_table[state]
            best_action = np.argmax(q_values)
            state_value = np.max(q_values)
            
            row, col = divmod(state, 4)
            q_str = " ".join([f"{q:6.2f}" for q in q_values])
            
            if show_analysis:
                # Análisis del estado
                visits = int(self.state_visits[state])
                is_optimal = "✓" if OptimalPaths.is_on_optimal_path(state) else " "
                optimal_action = OptimalPaths.get_optimal_action(state)
                policy_optimal = "✓" if optimal_action == best_action and optimal_action is not None else " "
                
                analysis = f"V:{visits:3d} O:{is_optimal} P:{policy_optimal}"
                print(f"{state:2d}({row},{col}) | {q_str} | {action_names[best_action]:4s}  | {state_value:6.2f} | {analysis}")
            else:
                print(f"{state:2d}({row},{col}) | {q_str} | {action_names[best_action]:4s}  | {state_value:6.2f}")
        
        if show_analysis:
            print("\nLeyenda: V=Visitas, O=Estado Óptimo, P=Política Óptima")
    
    def get_optimization_metrics(self):
        """Obtener métricas específicas de optimización"""
        if self.total_episodes == 0:
            return {}
        
        # Calcular métricas de optimización
        optimal_state_visits = sum(self.state_visits[state] for state in range(16) 
                                 if OptimalPaths.is_on_optimal_path(state))
        total_visits = sum(self.state_visits)
        optimal_visit_ratio = optimal_state_visits / max(total_visits, 1)
        
        # Análisis de política óptima
        optimal_policy_states = 0
        total_optimal_states = 0
        for state in range(16):
            optimal_action = OptimalPaths.get_optimal_action(state)
            if optimal_action is not None:
                total_optimal_states += 1
                if self.get_best_action(state) == optimal_action:
                    optimal_policy_states += 1
        
        optimal_policy_ratio = optimal_policy_states / max(total_optimal_states, 1)
        
        # Eficiencia reciente
        recent_efficiency = np.mean(self.efficiency_history[-100:]) if len(self.efficiency_history) >= 100 else 0
        
        return {
            'optimal_visit_ratio': optimal_visit_ratio,
            'optimal_policy_ratio': optimal_policy_ratio,
            'recent_efficiency': recent_efficiency,
            'exploration_temperature': self.exploration_temperature,
            'q_value_stability': 1.0 / (1.0 + np.mean(self.q_value_changes[-1000:]) * 100) if len(self.q_value_changes) >= 1000 else 0
        }
    
    def save(self, filepath=None):
        """Guardar agente completo optimizado"""
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
            'optimal_action_counts': self.optimal_action_counts,
            'q_value_changes': self.q_value_changes[-10000:],  # Últimos 10k
            'efficiency_history': self.efficiency_history,
            'exploration_temperature': self.exploration_temperature,
            'optimal_bias': self.optimal_bias
        }
        
        np.save(filepath, data)
        print(f"{LogConfig.EMOJIS['SAVE']} Agente Q-Learning optimizado guardado en {filepath}")
    
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
            self.optimal_action_counts = data.get('optimal_action_counts', np.zeros((EnvConfig.STATE_SIZE, EnvConfig.ACTION_SIZE)))
            self.q_value_changes = data.get('q_value_changes', [])
            self.efficiency_history = data.get('efficiency_history', [])
            self.exploration_temperature = data.get('exploration_temperature', 2.0)
            self.optimal_bias = data.get('optimal_bias', 0.3)
            
            print(f"{LogConfig.EMOJIS['SUCCESS']} Agente Q-Learning optimizado cargado desde {filepath}")
            
            # Mostrar estadísticas del modelo cargado
            stats = self.get_statistics()
            print(f"  Episodios entrenados: {stats['total_episodes']}")
            print(f"  Tasa de éxito: {stats['success_rate']:.1%}")
            print(f"  Rutas óptimas: {stats['optimal_solutions']}")
            print(f"  Rutas teóricas encontradas: {stats['theoretical_paths_found']}/3")
            print(f"  Eficiencia promedio: {stats['avg_efficiency']:.1%}")
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
    Crear y retornar un nuevo agente Q-Learning optimizado
    
    Returns:
        QLearningAgent: Agente Q-Learning optimizado inicializado
    """
    return QLearningAgent()

if __name__ == "__main__":
    # Crear y probar agente optimizado
    agent = create_agent()
    
    print(f"\nPrueba del agente Q-Learning optimizado:")
    print(f"Tabla Q inicializada: {agent.q_table.shape}")
    print(f"Epsilon inicial: {agent.epsilon}")
    print(f"Learning rate inicial: {agent.learning_rate}")
    print(f"Temperatura de exploración: {agent.exploration_temperature}")
    print(f"Sesgo hacia rutas óptimas: {agent.optimal_bias}")
    
    # Probar funciones básicas optimizadas
    test_state = 0
    test_action = agent.select_action(test_state)
    state_value = agent.get_state_value(test_state)
    action_probs = agent.get_action_probabilities(test_state)
    
    print(f"\nPrueba en estado {test_state}:")
    print(f"  Acción seleccionada: {test_action}")
    print(f"  Valor del estado: {state_value:.3f}")
    print(f"  Probabilidades de acción: {action_probs}")
    print(f"  Valores Q: {agent.q_table[test_state]}")
    
    # Probar recompensa moldeada optimizada
    test_reward = agent.calculate_reward(0, 1, 4, 0, False, 1)
    print(f"  Recompensa moldeada (0→4): {test_reward}")
    
    # Probar exploración inteligente
    print(f"\nPrueba de exploración inteligente:")
    for i in range(5):
        action = agent._get_intelligent_exploration_action(0)
        print(f"  Exploración {i+1}: acción {action}")
    
    # Mostrar análisis de rutas óptimas
    print(f"\nAnálisis de rutas óptimas:")
    for state in [0, 1, 4, 8, 9, 10, 14, 15]:
        row, col = divmod(state, 4)
        distance = get_manhattan_distance(state)
        on_optimal = OptimalPaths.is_on_optimal_path(state)
        optimal_action = OptimalPaths.get_optimal_action(state)
        progress = OptimalPaths.get_path_progress(state)
        
        print(f"  Estado {state} ({row},{col}): distancia={distance}, óptimo={'Sí' if on_optimal else 'No'}, "
              f"acción_óptima={optimal_action}, progreso={progress:.1%}")
    
    print(f"\n{LogConfig.EMOJIS['SUCCESS']} Agente Q-Learning optimizado listo para entrenamiento mejorado")