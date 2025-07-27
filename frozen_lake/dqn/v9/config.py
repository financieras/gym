"""
config.py - Configuración e Hiperparámetros para DQN en Frozen Lake

Este archivo contiene todas las configuraciones y hiperparámetros necesarios
para entrenar el agente DQN mejorado en el entorno Frozen Lake.
"""

import torch

# ================================
# CONFIGURACIÓN DEL DISPOSITIVO
# ================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================
# CONFIGURACIÓN DEL ENTORNO
# ================================
class EnvConfig:
    """Configuración del entorno Frozen Lake"""
    
    # Configuración del mapa
    MAP_SIZE = "4x4"  # "4x4" o "8x8"
    IS_SLIPPERY = True  # True para entorno estocástico, False para determinístico
    
    # Espacios de estado y acción
    STATE_SIZE = 16  # Para mapa 4x4
    ACTION_SIZE = 4  # 4 direcciones: izquierda, abajo, derecha, arriba
    
    # Número máximo de pasos por episodio
    MAX_STEPS = 100
    
    # Configuración de renderizado
    RENDER_MODE = None  # None para entrenamiento, "human" para visualización

# ================================
# HIPERPARÁMETROS DE LA RED NEURONAL
# ================================
class NetworkConfig:
    """Configuración de la red neuronal Q-Network"""
    
    # Arquitectura de la red
    INPUT_SIZE = EnvConfig.STATE_SIZE
    HIDDEN_SIZES = [128, 128, 64]  # Red más profunda para mejor aprendizaje
    OUTPUT_SIZE = EnvConfig.ACTION_SIZE
    
    # Configuración de entrenamiento
    LEARNING_RATE = 0.0005  # Learning rate más conservador
    BATCH_SIZE = 64  # Batch size mayor para estabilidad
    
    # Regularización
    DROPOUT_RATE = 0.1  # Dropout ligero
    WEIGHT_DECAY = 1e-5

# ================================
# HIPERPARÁMETROS DEL AGENTE DQN
# ================================
class DQNConfig:
    """Configuración del agente DQN optimizada para rutas óptimas"""
    
    # Exploración (Epsilon-Greedy) - AJUSTADO PARA MEJOR EVALUACIÓN
    EPSILON_START = 1.0      # Exploración inicial máxima
    EPSILON_END = 0.05       # Exploración mínima MÁS BAJA para mejor evaluación
    EPSILON_DECAY = 0.9997   # Factor de decaimiento MÁS LENTO
    
    # Q-Learning
    GAMMA = 0.99            # Factor de descuento
    TAU = 0.005             # Soft update para target network
    
    # Experience Replay
    MEMORY_SIZE = 50000     # Buffer más grande para más experiencias
    MIN_MEMORY_SIZE = 2000  # Mínimo más alto para mejor estabilidad
    
    # Frecuencia de actualización
    UPDATE_EVERY = 4        # Actualizar cada N pasos
    TARGET_UPDATE_EVERY = 100  # Actualizar target network cada N episodios

# ================================
# CONFIGURACIÓN DE ENTRENAMIENTO
# ================================
class TrainingConfig:
    """Configuración del proceso de entrenamiento extendido"""
    
    # Duración del entrenamiento - MÁS LARGO PARA MEJOR CONVERGENCIA
    NUM_EPISODES = 6000     # Número total de episodios (aumentado)
    
    # Logging y guardado
    PRINT_EVERY = 200       # Imprimir progreso cada N episodios
    SAVE_EVERY = 1000       # Guardar modelo cada N episodios
    
    # Métricas para evaluar convergencia
    SOLVE_SCORE = 0.8       # Tasa de éxito para considerar resuelto
    SOLVE_WINDOW = 100      # Ventana de episodios para evaluar
    
    # Paths de archivos
    MODEL_PATH = "frozen_lake_dqn.pth"
    RESULTS_PATH = "training_results.png"
    LOG_PATH = "training_log.txt"

# ================================
# CONFIGURACIÓN DE EVALUACIÓN
# ================================
class EvalConfig:
    """Configuración para evaluación del agente"""
    
    # Número de episodios de evaluación
    EVAL_EPISODES = 1000
    
    # Configuración de renderizado para demos
    DEMO_EPISODES = 5
    RENDER_DEMO = True
    
    # Métricas objetivo
    TARGET_SUCCESS_RATE = 0.8     # Tasa de éxito objetivo
    TARGET_OPTIMAL_RATE = 0.2     # Tasa de rutas óptimas objetivo (20%)
    OPTIMAL_STEPS = 6             # Número de pasos óptimo

# ================================
# CONFIGURACIÓN DE RECOMPENSAS
# ================================
class RewardConfig:
    """Configuración del sistema de recompensas moldeadas"""
    
    # Recompensas por éxito
    BASE_SUCCESS_REWARD = 10.0      # Recompensa base por llegar a la meta
    OPTIMAL_BONUS = 100.0           # Bonificación masiva por rutas óptimas (≤6 pasos)
    NEAR_OPTIMAL_BONUS = 50.0       # Bonificación por rutas casi-óptimas (≤8 pasos)
    GOOD_BONUS = 20.0               # Bonificación por rutas buenas (≤12 pasos)
    
    # Bonificaciones por seguir rutas teóricas
    THEORETICAL_PATH_BONUS = 5.0    # Bonificación por seguir ruta teórica exacta
    OPTIMAL_PATH_STEP_BONUS = 0.5   # Bonificación por estar en ruta óptima
    
    # Recompensas de navegación
    PROGRESS_REWARD = 1.0           # Recompensa por acercarse a la meta
    REGRESS_PENALTY = -0.5          # Penalización por alejarse
    NO_PROGRESS_PENALTY = -0.1      # Penalización por no progresar
    
    # Penalizaciones
    HOLE_PENALTY = -5.0             # Penalización por caer en agujero
    STEP_PENALTY_THRESHOLD = 15     # Umbral para penalización por pasos excesivos
    STEP_PENALTY_RATE = 0.1         # Penalización por paso excesivo

# ================================
# RUTAS ÓPTIMAS CONOCIDAS
# ================================
class OptimalPaths:
    """Rutas óptimas conocidas para el mapa 4x4"""
    
    PATHS_4X4 = [
        [0, 1, 2, 6, 10, 14, 15],    # Ruta Norte-Este
        [0, 4, 8, 9, 10, 14, 15],    # Ruta Sur-Este (variante 1)
        [0, 4, 8, 9, 13, 14, 15],    # Ruta Sur-Este (variante 2)
    ]
    
    @classmethod
    def get_optimal_actions(cls, state, path):
        """
        Obtener la acción óptima para un estado dado en una ruta específica
        
        Args:
            state (int): Estado actual
            path (list): Ruta óptima como lista de estados
            
        Returns:
            int or None: Acción óptima (0=izq, 1=abajo, 2=der, 3=arriba) o None
        """
        if state not in path:
            return None
            
        current_idx = path.index(state)
        if current_idx >= len(path) - 1:
            return None
            
        next_state = path[current_idx + 1]
        
        # Calcular acción basada en la diferencia de posiciones
        current_row, current_col = divmod(state, 4)
        next_row, next_col = divmod(next_state, 4)
        
        if next_row > current_row:
            return 1  # Abajo
        elif next_row < current_row:
            return 3  # Arriba
        elif next_col > current_col:
            return 2  # Derecha
        elif next_col < current_col:
            return 0  # Izquierda
        
        return None
    
    @classmethod
    def is_on_optimal_path(cls, state):
        """Verificar si un estado está en alguna ruta óptima"""
        for path in cls.PATHS_4X4:
            if state in path:
                return True
        return False
    
    @classmethod
    def check_path_match(cls, full_path):
        """
        Verificar si un camino completo coincide con alguna ruta teórica
        
        Args:
            full_path (list): Camino completo como lista de estados
            
        Returns:
            int or None: Índice de la ruta teórica coincidente o None
        """
        for i, theoretical_path in enumerate(cls.PATHS_4X4):
            if full_path == theoretical_path:
                return i
        return None

# ================================
# CONFIGURACIÓN DE LOGGING
# ================================
class LogConfig:
    """Configuración de logging y visualización"""
    
    # Colores para terminal
    COLORS = {
        'SUCCESS': '\033[92m',    # Verde
        'WARNING': '\033[93m',    # Amarillo
        'ERROR': '\033[91m',      # Rojo
        'INFO': '\033[94m',       # Azul
        'BOLD': '\033[1m',        # Negrita
        'END': '\033[0m'          # Fin de color
    }
    
    # Emojis para diferentes tipos de mensajes
    EMOJIS = {
        'SUCCESS': '✅',
        'OPTIMAL': '🎯',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'INFO': 'ℹ️',
        'TRAIN': '🚀',
        'EVAL': '🔬',
        'SAVE': '💾',
        'THEORETICAL': '🏆'
    }

# ================================
# UTILIDADES DE CONFIGURACIÓN
# ================================
def print_config():
    """Imprimir la configuración actual"""
    print(f"{LogConfig.COLORS['BOLD']}=== CONFIGURACIÓN DQN FROZEN LAKE MEJORADO ==={LogConfig.COLORS['END']}")
    print(f"Dispositivo: {DEVICE}")
    print(f"Mapa: {EnvConfig.MAP_SIZE} ({'resbaladizo' if EnvConfig.IS_SLIPPERY else 'determinístico'})")
    print(f"Episodios: {TrainingConfig.NUM_EPISODES}")
    print(f"Arquitectura: {NetworkConfig.INPUT_SIZE} → {' → '.join(map(str, NetworkConfig.HIDDEN_SIZES))} → {NetworkConfig.OUTPUT_SIZE}")
    print(f"Learning Rate: {NetworkConfig.LEARNING_RATE}")
    print(f"Epsilon: {DQNConfig.EPSILON_START} → {DQNConfig.EPSILON_END} (decay: {DQNConfig.EPSILON_DECAY})")
    print(f"Gamma: {DQNConfig.GAMMA}")
    print(f"Batch Size: {NetworkConfig.BATCH_SIZE}")
    print(f"Recompensas moldeadas: ✅")
    print(f"Exploración dirigida: ✅")
    print("-" * 50)

def validate_config():
    """Validar que la configuración sea coherente"""
    assert EnvConfig.STATE_SIZE == 16, "STATE_SIZE debe ser 16 para mapa 4x4"
    assert EnvConfig.ACTION_SIZE == 4, "ACTION_SIZE debe ser 4"
    assert 0 < DQNConfig.EPSILON_DECAY < 1, "EPSILON_DECAY debe estar entre 0 y 1"
    assert 0 < DQNConfig.GAMMA <= 1, "GAMMA debe estar entre 0 y 1"
    assert NetworkConfig.BATCH_SIZE <= DQNConfig.MEMORY_SIZE, "BATCH_SIZE no puede ser mayor que MEMORY_SIZE"
    assert DQNConfig.MIN_MEMORY_SIZE <= DQNConfig.MEMORY_SIZE, "MIN_MEMORY_SIZE no puede ser mayor que MEMORY_SIZE"
    assert EvalConfig.OPTIMAL_STEPS <= EnvConfig.MAX_STEPS, "OPTIMAL_STEPS debe ser menor que MAX_STEPS"
    
    # Validar que las rutas óptimas sean válidas
    for i, path in enumerate(OptimalPaths.PATHS_4X4):
        assert len(path) == 7, f"Ruta {i+1} debe tener 7 estados (6 pasos)"
        assert path[0] == 0, f"Ruta {i+1} debe empezar en estado 0"
        assert path[-1] == 15, f"Ruta {i+1} debe terminar en estado 15"
        assert all(0 <= state <= 15 for state in path), f"Ruta {i+1} contiene estados inválidos"
    
    print(f"{LogConfig.EMOJIS['SUCCESS']} Configuración validada correctamente")

def get_distance_to_goal(state):
    """Calcular distancia Manhattan desde un estado a la meta (estado 15)"""
    goal_row, goal_col = 3, 3
    state_row, state_col = divmod(state, 4)
    return abs(state_row - goal_row) + abs(state_col - goal_col)

if __name__ == "__main__":
    print_config()
    validate_config()
    
    # Mostrar rutas óptimas
    print(f"\n{LogConfig.EMOJIS['OPTIMAL']} Rutas óptimas objetivo:")
    for i, path in enumerate(OptimalPaths.PATHS_4X4, 1):
        path_str = " → ".join(map(str, path))
        print(f"  {i}. {path_str}")
    
    print(f"\n{LogConfig.EMOJIS['INFO']} Configuración de recompensas:")
    print(f"  Ruta óptima (≤6 pasos): +{RewardConfig.BASE_SUCCESS_REWARD + RewardConfig.OPTIMAL_BONUS}")
    print(f"  Ruta casi-óptima (≤8 pasos): +{RewardConfig.BASE_SUCCESS_REWARD + RewardConfig.NEAR_OPTIMAL_BONUS}")
    print(f"  Seguir ruta teórica: +{RewardConfig.THEORETICAL_PATH_BONUS}")
    print(f"  Progreso hacia meta: +{RewardConfig.PROGRESS_REWARD}")