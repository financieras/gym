"""
config.py - Configuración para Q-Learning con Q-Table en Frozen Lake

Este archivo contiene la configuración optimizada para Q-Learning clásico
con tabla Q en lugar de redes neuronales, para mejor convergencia en
espacios de estados pequeños como Frozen Lake 4x4.
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
# HIPERPARÁMETROS DE Q-LEARNING
# ================================
class QLearningConfig:
    """Configuración de Q-Learning con tabla Q"""
    
    # Parámetros de aprendizaje
    LEARNING_RATE = 0.1        # Alpha - tasa de aprendizaje
    DISCOUNT_FACTOR = 0.99     # Gamma - factor de descuento
    
    # Exploración (Epsilon-Greedy)
    EPSILON_START = 1.0        # Exploración inicial máxima
    EPSILON_END = 0.01         # Exploración mínima final
    EPSILON_DECAY = 0.9995     # Decay muy gradual
    
    # Configuración de convergencia
    MIN_LEARNING_RATE = 0.01   # Learning rate mínimo
    LR_DECAY = 0.9999          # Decay del learning rate

# ================================
# CONFIGURACIÓN DE ENTRENAMIENTO
# ================================
class TrainingConfig:
    """Configuración del entrenamiento"""
    
    NUM_EPISODES = 8000        # Más episodios para convergencia completa
    PRINT_EVERY = 400          # Reportar cada 400 episodios
    SAVE_EVERY = 2000          # Guardar cada 2000 episodios
    
    # Criterios de convergencia
    SOLVE_SCORE = 0.8          # Tasa de éxito objetivo
    SOLVE_WINDOW = 100         # Ventana para evaluar convergencia
    
    # Archivos
    MODEL_PATH = "frozen_lake_qtable.npy"
    RESULTS_PATH = "qtable_results.png"

# ================================
# CONFIGURACIÓN DE EVALUACIÓN
# ================================
class EvalConfig:
    """Configuración para evaluación"""
    
    EVAL_EPISODES = 1000
    DEMO_EPISODES = 5
    
    # Objetivos
    TARGET_SUCCESS_RATE = 0.8
    TARGET_OPTIMAL_RATE = 0.3    # 30% de rutas óptimas
    OPTIMAL_STEPS = 6

# ================================
# CONFIGURACIÓN DE RECOMPENSAS
# ================================
class RewardConfig:
    """Sistema de recompensas moldeadas para Q-Learning"""
    
    # Recompensas principales
    SUCCESS_REWARD = 100.0          # Llegar a la meta
    OPTIMAL_BONUS = 200.0           # Bonificación por ruta óptima
    THEORETICAL_BONUS = 50.0        # Bonificación por ruta teórica exacta
    
    # Recompensas de navegación
    PROGRESS_REWARD = 1.0           # Acercarse a la meta
    REGRESS_PENALTY = -0.5          # Alejarse de la meta
    HOLE_PENALTY = -10.0            # Caer en agujero
    STEP_PENALTY = -0.1             # Penalización por paso
    
    # Exploración dirigida
    EXPLORATION_BONUS = 0.5         # Bonificación por estados poco visitados

# ================================
# RUTAS ÓPTIMAS CONOCIDAS
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

# ================================
# CONFIGURACIÓN DE LOGGING
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
        'THEORETICAL': '🏆'
    }

# ================================
# FUNCIONES DE UTILIDAD
# ================================
def get_manhattan_distance(state, goal_state=15):
    """Calcular distancia Manhattan entre dos estados"""
    state_row, state_col = divmod(state, 4)
    goal_row, goal_col = divmod(goal_state, 4)
    return abs(state_row - goal_row) + abs(state_col - goal_col)

def print_config():
    """Imprimir configuración actual"""
    print(f"{LogConfig.COLORS['BOLD']}=== CONFIGURACIÓN Q-LEARNING FROZEN LAKE ==={LogConfig.COLORS['END']}")
    print(f"Algoritmo: Q-Learning con Tabla Q")
    print(f"Estados: {EnvConfig.STATE_SIZE}")
    print(f"Acciones: {EnvConfig.ACTION_SIZE}")
    print(f"Episodios: {TrainingConfig.NUM_EPISODES}")
    print(f"Learning Rate: {QLearningConfig.LEARNING_RATE} → {QLearningConfig.MIN_LEARNING_RATE}")
    print(f"Epsilon: {QLearningConfig.EPSILON_START} → {QLearningConfig.EPSILON_END}")
    print(f"Gamma: {QLearningConfig.DISCOUNT_FACTOR}")
    print(f"Recompensas moldeadas: ✅")
    print(f"Exploración dirigida: ✅")
    print("-" * 50)

def validate_config():
    """Validar configuración"""
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
    
    print(f"{LogConfig.EMOJIS['SUCCESS']} Configuración validada correctamente")

if __name__ == "__main__":
    print_config()
    validate_config()
    
    print(f"\n{LogConfig.EMOJIS['OPTIMAL']} Rutas óptimas objetivo:")
    for i, path in enumerate(OptimalPaths.PATHS_4X4, 1):
        path_str = " → ".join(map(str, path))
        print(f"  {i}. {path_str}")
    
    print(f"\n{LogConfig.EMOJIS['INFO']} Ventajas de Q-Table vs DQN:")
    print("  • Convergencia garantizada en espacios pequeños")
    print("  • Sin problemas de generalización")
    print("  • Interpretabilidad completa")
    print("  • Memoria perfecta de todas las experiencias")
    print("  • No hay overfitting")