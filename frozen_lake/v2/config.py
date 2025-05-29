# config.py
# Configuraciones para Q-Learning en FrozenLake

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import numpy as np

@dataclass
class QLearningConfig:
    """Configuración base para Q-Learning"""
    # Parámetros de Q-Learning
    learning_rate: float = 0.8
    discount_factor: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.9995
    
    # Parámetros de entrenamiento
    n_episodes: int = 5000
    max_steps_per_episode: int = 200
    progress_report_interval: int = 500
    
    # Parámetros de prueba
    n_test_episodes: int = 10
    max_test_steps: int = 50

@dataclass 
class EnvironmentConfig:
    """Configuración del ambiente FrozenLake"""
    # Parámetros del mapa
    map_name: str = "4x4"  # "4x4" o "8x8"
    is_slippery: bool = False
    
    # Propiedades del mapa (se calculan automáticamente)
    grid_size: int = 4
    n_states: int = 16
    n_actions: int = 4
    start_state: int = 0
    goal_state: int = 15
    hole_states: Set[int] = None
    
    # Representación visual del mapa
    map_layout: List[str] = None
    
    def __post_init__(self):
        """Calcula automáticamente las propiedades basadas en map_name"""
        if self.map_name == "4x4":
            self.grid_size = 4
            self.n_states = 16
            self.start_state = 0
            self.goal_state = 15
            self.hole_states = {5, 7, 11, 12}
            self.map_layout = [
                "SFFF",
                "FHFH", 
                "FFFH",
                "HFFG"
            ]
        elif self.map_name == "8x8":
            self.grid_size = 8
            self.n_states = 64
            self.start_state = 0
            self.goal_state = 63
            self.hole_states = {19, 29, 35, 41, 42, 46, 49, 52, 54, 59}
            self.map_layout = [
                "SFFFFFFF",
                "FFFFFFFF", 
                "FFFHFFFF",
                "FFFFFHFF",
                "FFFHFFFF",
                "FHHFFFHF",
                "FHFFHFHF",
                "FFFHFFFG"
            ]
        else:
            raise ValueError(f"Mapa '{self.map_name}' no soportado. Use '4x4' o '8x8'")
    
    def state_to_coordinates(self, state: int) -> Tuple[int, int]:
        """Convierte estado a coordenadas (fila, columna)"""
        row = state // self.grid_size
        col = state % self.grid_size
        return row, col
    
    def coordinates_to_state(self, row: int, col: int) -> int:
        """Convierte coordenadas a estado"""
        return row * self.grid_size + col
    
    def is_valid_state(self, state: int) -> bool:
        """Verifica si un estado es válido (no es agujero)"""
        return state not in self.hole_states
    
    def get_action_names(self) -> Dict[int, str]:
        """Retorna mapeo de acciones a nombres"""
        return {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
    
    def get_action_names_spanish(self) -> Dict[int, str]:
        """Retorna mapeo de acciones a nombres en español"""
        return {0: "Izquierda", 1: "Abajo", 2: "Derecha", 3: "Arriba"}
    
    def get_policy_symbols(self) -> Dict[int, str]:
        """Retorna símbolos para mostrar la política"""
        return {0: "←", 1: "↓", 2: "→", 3: "↑"}

@dataclass
class FrozenLakeConfig:
    """Configuración completa para FrozenLake Q-Learning"""
    environment: EnvironmentConfig
    qlearning: QLearningConfig
    
    def __init__(self, map_name: str = "4x4", **kwargs):
        """
        Inicializa configuración completa
        
        Args:
            map_name: "4x4" o "8x8" 
            **kwargs: Parámetros adicionales para sobreescribir defaults
        """
        self.environment = EnvironmentConfig(map_name=map_name)
        self.qlearning = QLearningConfig()
        
        # Aplicar kwargs para sobreescribir defaults
        for key, value in kwargs.items():
            if hasattr(self.qlearning, key):
                setattr(self.qlearning, key, value)
            elif hasattr(self.environment, key):
                setattr(self.environment, key, value)
            else:
                raise ValueError(f"Parámetro '{key}' no reconocido")
    
    def get_gym_kwargs(self) -> Dict:
        """Retorna argumentos para gym.make()"""
        kwargs = {
            'is_slippery': self.environment.is_slippery
        }
        if self.environment.map_name != "4x4":
            kwargs['map_name'] = self.environment.map_name
        return kwargs
    
    def summary(self) -> str:
        """Retorna resumen de la configuración"""
        return f"""
=== CONFIGURACIÓN FROZENLAKE ===
Ambiente:
  - Mapa: {self.environment.map_name} ({self.environment.n_states} estados)
  - Resbaladizo: {self.environment.is_slippery}
  - Estados agujero: {len(self.environment.hole_states)}
  
Q-Learning:
  - Learning rate: {self.qlearning.learning_rate}
  - Discount factor: {self.qlearning.discount_factor}
  - Epsilon: {self.qlearning.epsilon_start} → {self.qlearning.epsilon_end}
  - Episodios: {self.qlearning.n_episodes}
  - Max pasos/episodio: {self.qlearning.max_steps_per_episode}
"""

# ========== CONFIGURACIONES PREDEFINIDAS ==========

class Presets:
    """Configuraciones predefinidas para diferentes escenarios"""
    
    @staticmethod
    def frozen_lake_4x4_fast():
        """Configuración rápida para 4x4 (desarrollo/testing)"""
        return FrozenLakeConfig(
            map_name="4x4",
            n_episodes=2000,
            epsilon_decay=0.995,
            learning_rate=0.8,
            max_steps_per_episode=100
        )
    
    @staticmethod
    def frozen_lake_4x4_optimal():
        """Configuración óptima para 4x4 (mejor rendimiento)"""
        return FrozenLakeConfig(
            map_name="4x4",
            n_episodes=5000,
            epsilon_decay=0.9995,
            learning_rate=0.8,
            max_steps_per_episode=200
        )
    
    @staticmethod
    def frozen_lake_8x8_fast():
        """Configuración rápida para 8x8 (desarrollo/testing)"""
        return FrozenLakeConfig(
            map_name="8x8",
            n_episodes=8000,
            epsilon_decay=0.9997,
            learning_rate=0.6,
            max_steps_per_episode=300,
            progress_report_interval=1000
        )
    
    @staticmethod
    def frozen_lake_8x8_optimal():
        """Configuración óptima para 8x8 (mejor rendimiento)"""
        return FrozenLakeConfig(
            map_name="8x8",
            n_episodes=20000,
            epsilon_decay=0.9998,
            learning_rate=0.5,
            max_steps_per_episode=500,
            progress_report_interval=2000
        )
    
    @staticmethod
    def frozen_lake_slippery_4x4():
        """Configuración para 4x4 resbaladizo (más difícil)"""
        return FrozenLakeConfig(
            map_name="4x4",
            is_slippery=True,
            n_episodes=10000,
            epsilon_decay=0.9998,
            learning_rate=0.3,
            max_steps_per_episode=300
        )

# ========== FUNCIONES DE CONVENIENCIA ==========

def get_config(preset_name: str) -> FrozenLakeConfig:
    """
    Obtiene una configuración predefinida
    
    Args:
        preset_name: Nombre del preset
            - "4x4_fast": 4x4 rápido
            - "4x4_optimal": 4x4 óptimo  
            - "8x8_fast": 8x8 rápido
            - "8x8_optimal": 8x8 óptimo
            - "4x4_slippery": 4x4 resbaladizo
    
    Returns:
        FrozenLakeConfig configurado
    """
    presets = {
        "4x4_fast": Presets.frozen_lake_4x4_fast,
        "4x4_optimal": Presets.frozen_lake_4x4_optimal,
        "8x8_fast": Presets.frozen_lake_8x8_fast,
        "8x8_optimal": Presets.frozen_lake_8x8_optimal,
        "4x4_slippery": Presets.frozen_lake_slippery_4x4
    }
    
    if preset_name not in presets:
        available = ", ".join(presets.keys())
        raise ValueError(f"Preset '{preset_name}' no existe. Disponibles: {available}")
    
    return presets[preset_name]()

def list_presets() -> None:
    """Muestra todos los presets disponibles"""
    print("=== PRESETS DISPONIBLES ===")
    presets = [
        ("4x4_fast", "4x4 rápido para desarrollo/testing"),
        ("4x4_optimal", "4x4 óptimo para mejor rendimiento"),
        ("8x8_fast", "8x8 rápido para desarrollo/testing"), 
        ("8x8_optimal", "8x8 óptimo para mejor rendimiento"),
        ("4x4_slippery", "4x4 resbaladizo (más difícil)")
    ]
    
    for name, description in presets:
        print(f"  {name:15s} - {description}")

# ========== EJEMPLO DE USO ==========
if __name__ == "__main__":
    # Mostrar presets disponibles
    list_presets()
    
    # Crear configuración personalizada
    custom_config = FrozenLakeConfig(
        map_name="4x4",
        n_episodes=3000,
        learning_rate=0.9
    )
    print(custom_config.summary())
    
    # Usar preset
    preset_config = get_config("8x8_optimal")
    print(preset_config.summary())