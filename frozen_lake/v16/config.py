"""
config.py - Configuración OPTIMIZADA para Q-Learning con Q-Table en Frozen Lake

Esta versión optimizada mejora la convergencia y aumenta la frecuencia de rutas óptimas
basada en el análisis de los resultados de entrenamiento anteriores.
"""

import numpy as np

# ================================
# CONFIGURACIÓN DEL ENTORNO
# ================================
class EnvConfig:
    """Configuración del entorno Frozen Lake"""
    
    MAP_SIZE = "4x4"
    IS_SLIPPERY = True
    STATE_SIZE = 16  # Estados 0-15
    ACTION_SIZE = 4  # Izquierda, Abajo, Derecha, Arriba
    MAX_STEPS = 100
    RENDER_MODE = None

# ================================
# HIPERPARÁMETROS DE Q-LEARNING OPTIMIZADOS
# ================================
class QLearningConfig:
    """Configuración optimizada de Q-Learning con tabla Q"""
    
    # Parámetros de aprendizaje mejorados
    LEARNING_RATE = 0.15       # Alpha - incrementado para aprendizaje más rápido
    DISCOUNT_FACTOR = 0.95     # Gamma - reducido para valorar más recompensas inmediatas
    
    # Exploración optimizada (Epsilon-Greedy)
    EPSILON_START = 0.9        # Menos exploración inicial
    EPSILON_END = 0.05         # Más exploración final para encontrar rutas óptimas
    EPSILON_DECAY = 0.9992     # Decay más lento para mejor exploración
    
    # Configuración de convergencia
    MIN_LEARNING_RATE = 0.02   # Learning rate mínimo más alto
    LR_DECAY = 0.9998          # Decay más lento del learning rate

# ================================
# CONFIGURACIÓN DE ENTRENAMIENTO OPTIMIZADA
# ================================
class TrainingConfig:
    """Configuración optimizada del entrenamiento"""
    
    NUM_EPISODES = 12000       # Más episodios para convergencia completa
    PRINT_EVERY = 500          # Reportar cada 500 episodios
    SAVE_EVERY = 2500          # Guardar cada 2500 episodios
    
    # Criterios de convergencia mejorados
    SOLVE_SCORE = 0.75         # Objetivo más realista para entorno stochastic
    SOLVE_WINDOW = 200         # Ventana más grande para evaluación
    
    # Archivos
    MODEL_PATH = "frozen_lake_qtable_optimized.npy"
    RESULTS_PATH = "qtable_results_optimized.png"

# ================================
# CONFIGURACIÓN DE EVALUACIÓN
# ================================
class EvalConfig:
    """Configuración para evaluación"""
    
    EVAL_EPISODES = 2000       # Más episodios para evaluación estadísticamente significativa
    DEMO_EPISODES = 8          # Más demos para mostrar
    
    # Objetivos optimizados
    TARGET_SUCCESS_RATE = 0.7  # Objetivo más realista
    TARGET_OPTIMAL_RATE = 0.1  # 10% de rutas óptimas (más ambicioso)
    OPTIMAL_STEPS = 6

# ================================
# CONFIGURACIÓN DE RECOMPENSAS OPTIMIZADA
# ================================
class RewardConfig:
    """Sistema de recompensas optimizado para maximizar rutas óptimas"""
    
    # Recompensas principales amplificadas
    SUCCESS_REWARD = 1000.0         # Incrementado significativamente
    OPTIMAL_BONUS = 500.0           # Bonificación masiva por ruta óptima
    THEORETICAL_BONUS = 200.0       # Bonificación adicional por ruta teórica exacta
    
    # Recompensas de navegación refinadas
    PROGRESS_REWARD = 5.0           # Incrementado para motivar progreso
    REGRESS_PENALTY = -2.0          # Penalización más fuerte por retroceder
    HOLE_PENALTY = -50.0            # Penalización moderada por agujero
    STEP_PENALTY = -0.5             # Penalización por paso más significativa
    
    # Exploración dirigida mejorada
    EXPLORATION_BONUS = 2.0         # Bonificación aumentada por exploración
    OPTIMAL_PATH_BONUS = 3.0        # Nueva: bonificación por estar en ruta óptima
    
    # Nuevas recompensas para incentivar eficiencia
    EFFICIENCY_BONUS_BASE = 50.0    # Bonificación base por eficiencia
    EFFICIENCY_THRESHOLD = 10       # Umbral para bonificación de eficiencia

# ================================
# RUTAS ÓPTIMAS CONOCIDAS (sin cambios)
# ================================
class OptimalPaths:
    """Rutas óptimas conocidas y funciones de análisis"""
    
    PATHS_4X4 = [
        [0, 1, 2, 6, 10, 14, 15],    # Ruta Norte-Este
        [0, 4, 8, 9, 10, 14, 15],    # Ruta Sur-Este (variante 1)
        [0, 4, 8, 9, 13, 14, 15],    # Ruta Sur-Este (variante 2)
    ]
    
    @classmethod
    def get_optimal_action(cls, state):
        """
        Obtener acción óptima para un estado dado
        
        Args:
            state (int): Estado actual
            
        Returns:
            int or None: Acción óptima o None si no está en ruta óptima
        """
        for path in cls.PATHS_4X4:
            if state in path:
                current_idx = path.index(state)
                if current_idx < len(path) - 1:
                    next_state = path[current_idx + 1]
                    return cls._calculate_action(state, next_state)
        return None
    
    @classmethod
    def _calculate_action(cls, from_state, to_state):
        """Calcular acción necesaria para ir de un estado a otro"""
        from_row, from_col = divmod(from_state, 4)
        to_row, to_col = divmod(to_state, 4)
        
        if to_row > from_row:
            return 1  # Abajo
        elif to_row < from_row:
            return 3  # Arriba
        elif to_col > from_col:
            return 2  # Derecha
        elif to_col < from_col:
            return 0  # Izquierda
        
        return None
    
    @classmethod
    def is_optimal_path(cls, path):
        """Verificar si un camino es una ruta óptima teórica"""
        for i, theoretical_path in enumerate(cls.PATHS_4X4):
            if path == theoretical_path:
                return i
        return None
    
    @classmethod
    def is_on_optimal_path(cls, state):
        """Verificar si un estado está en alguna ruta óptima"""
        for path in cls.PATHS_4X4:
            if state in path:
                return True
        return False
    
    @classmethod
    def get_path_progress(cls, state):
        """Obtener progreso en rutas óptimas (nuevo)"""
        max_progress = 0
        for path in cls.PATHS_4X4:
            if state in path:
                progress = path.index(state) / (len(path) - 1)
                max_progress = max(max_progress, progress)
        return max_progress

# ================================
# CONFIGURACIÓN DE LOGGING (sin cambios)
# ================================
class LogConfig:
    """Configuración de logging y visualización"""
    
    COLORS = {
        'SUCCESS': '\033[92m',
        'WARNING': '\033[93m',
        'ERROR': '\033[91m',
        'INFO': '\033[94m',
        'BOLD': '\033[1m',
        'END': '\033[0m'
    }
    
    EMOJIS = {
        'SUCCESS': '✅',
        'OPTIMAL': '🎯',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'INFO': 'ℹ️',
        'TRAIN': '🚀',
        'EVAL': '🔬',
        'SAVE': '💾',
        'QTABLE': '📊',
        'THEORETICAL': '🏆',
        'EFFICIENCY': '⚡',
        'IMPROVEMENT': '📈'
    }

# ================================
# FUNCIONES DE UTILIDAD MEJORADAS
# ================================
def get_manhattan_distance(state, goal_state=15):
    """Calcular distancia Manhattan entre dos estados"""
    state_row, state_col = divmod(state, 4)
    goal_row, goal_col = divmod(goal_state, 4)
    return abs(state_row - goal_row) + abs(state_col - goal_col)

def calculate_efficiency_bonus(steps):
    """Calcular bonificación por eficiencia (nuevo)"""
    if steps <= EvalConfig.OPTIMAL_STEPS:
        return RewardConfig.EFFICIENCY_BONUS_BASE * 2  # Doble bonificación para rutas óptimas
    elif steps <= RewardConfig.EFFICIENCY_THRESHOLD:
        # Bonificación graduada para rutas eficientes
        efficiency_ratio = (RewardConfig.EFFICIENCY_THRESHOLD - steps) / RewardConfig.EFFICIENCY_THRESHOLD
        return RewardConfig.EFFICIENCY_BONUS_BASE * efficiency_ratio
    else:
        return 0

def print_config():
    """Imprimir configuración optimizada actual"""
    print(f"{LogConfig.COLORS['BOLD']}=== CONFIGURACIÓN Q-LEARNING OPTIMIZADA FROZEN LAKE ==={LogConfig.COLORS['END']}")
    print(f"Algoritmo: Q-Learning con Tabla Q (OPTIMIZADO)")
    print(f"Estados: {EnvConfig.STATE_SIZE}")
    print(f"Acciones: {EnvConfig.ACTION_SIZE}")
    print(f"Episodios: {TrainingConfig.NUM_EPISODES}")
    print(f"Learning Rate: {QLearningConfig.LEARNING_RATE} → {QLearningConfig.MIN_LEARNING_RATE}")
    print(f"Epsilon: {QLearningConfig.EPSILON_START} → {QLearningConfig.EPSILON_END}")
    print(f"Gamma: {QLearningConfig.DISCOUNT_FACTOR}")
    print(f"Recompensas optimizadas: ✅")
    print(f"Exploración dirigida mejorada: ✅")
    print(f"Bonificaciones de eficiencia: ✅")
    print("-" * 60)

def validate_config():
    """Validar configuración optimizada"""
    assert EnvConfig.STATE_SIZE == 16, "STATE_SIZE debe ser 16 para mapa 4x4"
    assert EnvConfig.ACTION_SIZE == 4, "ACTION_SIZE debe ser 4"
    assert 0 < QLearningConfig.LEARNING_RATE <= 1, "Learning rate debe estar entre 0 y 1"
    assert 0 < QLearningConfig.DISCOUNT_FACTOR <= 1, "Gamma debe estar entre 0 y 1"
    assert 0 <= QLearningConfig.EPSILON_END < QLearningConfig.EPSILON_START <= 1, "Epsilon inválido"
    
    # Validar rutas óptimas
    for i, path in enumerate(OptimalPaths.PATHS_4X4):
        assert len(path) == 7, f"Ruta {i+1} debe tener 7 estados"
        assert path[0] == 0 and path[-1] == 15, f"Ruta {i+1} debe ir de 0 a 15"
        assert all(0 <= state <= 15 for state in path), f"Ruta {i+1} contiene estados inválidos"
    
    print(f"{LogConfig.EMOJIS['SUCCESS']} Configuración optimizada validada correctamente")

def get_optimization_summary():
    """Obtener resumen de optimizaciones implementadas"""
    optimizations = [
        "🎯 Learning rate incrementado para convergencia más rápida",
        "🎯 Gamma reducido para valorar recompensas inmediatas",
        "🎯 Epsilon decay más lento para mejor exploración",
        "🎯 Recompensas amplificadas para rutas óptimas",
        "🎯 Bonificaciones de eficiencia graduadas",
        "🎯 Penalizaciones balanceadas para guiar aprendizaje",
        "🎯 Más episodios para convergencia completa",
        "🎯 Evaluación estadísticamente más robusta"
    ]
    return optimizations

if __name__ == "__main__":
    print_config()
    validate_config()
    
    print(f"\n{LogConfig.EMOJIS['OPTIMAL']} Rutas óptimas objetivo:")
    for i, path in enumerate(OptimalPaths.PATHS_4X4, 1):
        path_str = " → ".join(map(str, path))
        print(f"  {i}. {path_str}")
    
    print(f"\n{LogConfig.EMOJIS['IMPROVEMENT']} Optimizaciones implementadas:")
    for optimization in get_optimization_summary():
        print(f"  {optimization}")
    
    print(f"\n{LogConfig.EMOJIS['INFO']} Ventajas de la configuración optimizada:")
    print("  • Convergencia más rápida y estable")
    print("  • Mayor frecuencia de rutas óptimas")
    print("  • Exploración más inteligente")
    print("  • Recompensas balanceadas para eficiencia")
    print("  • Evaluación estadísticamente robusta")