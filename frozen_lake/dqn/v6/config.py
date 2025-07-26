"""
config.py - Configuración e Hiperparámetros para DQN en Frozen Lake

Este archivo contiene todas las configuraciones y hiperparámetros necesarios
para entrenar el agente DQN en el entorno Frozen Lake.
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
    HIDDEN_SIZES = [64, 64]  # Capas ocultas más simples
    OUTPUT_SIZE = EnvConfig.ACTION_SIZE
    
    # Configuración de entrenamiento
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    
    # Regularización
    DROPOUT_RATE = 0.0  # Sin dropout inicialmente
    WEIGHT_DECAY = 1e-5

# ================================
# HIPERPARÁMETROS DEL AGENTE DQN
# ================================
class DQNConfig:
    """Configuración del agente DQN"""
    
    # Exploración (Epsilon-Greedy)
    EPSILON_START = 1.0      # Exploración inicial máxima
    EPSILON_END = 0.01       # Exploración mínima
    EPSILON_DECAY = 0.995    # Factor de decaimiento por episodio
    
    # Q-Learning
    GAMMA = 0.99            # Factor de descuento
    TAU = 0.005             # Soft update para target network
    
    # Experience Replay
    MEMORY_SIZE = 10000     # Tamaño del buffer de memoria
    MIN_MEMORY_SIZE = 1000  # Mínimo para comenzar entrenamiento
    
    # Frecuencia de actualización
    UPDATE_EVERY = 4        # Actualizar cada N pasos
    TARGET_UPDATE_EVERY = 100  # Actualizar target network cada N episodios

# ================================
# CONFIGURACIÓN DE ENTRENAMIENTO
# ================================
class TrainingConfig:
    """Configuración del proceso de entrenamiento"""
    
    # Duración del entrenamiento
    NUM_EPISODES = 2000     # Número total de episodios
    
    # Logging y guardado
    PRINT_EVERY = 100       # Imprimir progreso cada N episodios
    SAVE_EVERY = 500        # Guardar modelo cada N episodios
    
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
    TARGET_OPTIMAL_RATE = 0.3     # Tasa de rutas óptimas objetivo
    OPTIMAL_STEPS = 6             # Número de pasos óptimo

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
        'SAVE': '💾'
    }

# ================================
# UTILIDADES DE CONFIGURACIÓN
# ================================
def print_config():
    """Imprimir la configuración actual"""
    print(f"{LogConfig.COLORS['BOLD']}=== CONFIGURACIÓN DQN FROZEN LAKE ==={LogConfig.COLORS['END']}")
    print(f"Dispositivo: {DEVICE}")
    print(f"Mapa: {EnvConfig.MAP_SIZE} ({'resbaladizo' if EnvConfig.IS_SLIPPERY else 'determinístico'})")
    print(f"Episodios: {TrainingConfig.NUM_EPISODES}")
    print(f"Arquitectura: {NetworkConfig.INPUT_SIZE} → {' → '.join(map(str, NetworkConfig.HIDDEN_SIZES))} → {NetworkConfig.OUTPUT_SIZE}")
    print(f"Learning Rate: {NetworkConfig.LEARNING_RATE}")
    print(f"Epsilon: {DQNConfig.EPSILON_START} → {DQNConfig.EPSILON_END} (decay: {DQNConfig.EPSILON_DECAY})")
    print(f"Gamma: {DQNConfig.GAMMA}")
    print(f"Batch Size: {NetworkConfig.BATCH_SIZE}")
    print("-" * 50)

def validate_config():
    """Validar que la configuración sea coherente"""
    assert EnvConfig.STATE_SIZE == 16, "STATE_SIZE debe ser 16 para mapa 4x4"
    assert EnvConfig.ACTION_SIZE == 4, "ACTION_SIZE debe ser 4"
    assert 0 < DQNConfig.EPSILON_DECAY < 1, "EPSILON_DECAY debe estar entre 0 y 1"
    assert 0 < DQNConfig.GAMMA <= 1, "GAMMA debe estar entre 0 y 1"
    assert NetworkConfig.BATCH_SIZE <= DQNConfig.MEMORY_SIZE, "BATCH_SIZE no puede ser mayor que MEMORY_SIZE"
    assert DQNConfig.MIN_MEMORY_SIZE <= DQNConfig.MEMORY_SIZE, "MIN_MEMORY_SIZE no puede ser mayor que MEMORY_SIZE"
    
    print(f"{LogConfig.EMOJIS['SUCCESS']} Configuración validada correctamente")

if __name__ == "__main__":
    print_config()
    validate_config()