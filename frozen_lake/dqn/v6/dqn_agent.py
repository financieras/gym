"""
dqn_agent.py - Implementación del Agente DQN para Frozen Lake

Este archivo contiene la implementación del agente DQN, incluyendo la red neuronal,
el buffer de experiencia, y toda la lógica de entrenamiento.
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
    Red neuronal para aproximar la función Q(s,a)
    
    Esta red toma un estado (codificado como one-hot) y devuelve
    los valores Q para todas las acciones posibles.
    """
    
    def __init__(self, state_size=NetworkConfig.INPUT_SIZE, 
                 action_size=NetworkConfig.OUTPUT_SIZE, 
                 hidden_sizes=NetworkConfig.HIDDEN_SIZES):
        """
        Inicializar la red neuronal
        
        Args:
            state_size (int): Tamaño del espacio de estados
            action_size (int): Tamaño del espacio de acciones  
            hidden_sizes (list): Lista con tamaños de capas ocultas
        """
        super(QNetwork, self).__init__()
        
        # Construir capas dinámicamente
        layers = []
        
        # Capa de entrada
        layers.append(nn.Linear(state_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        
        # Capas ocultas
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
        
        # Capa de salida
        layers.append(nn.Linear(hidden_sizes[-1], action_size))
        
        self.network = nn.Sequential(*layers)
        
        # Inicialización de pesos
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializar pesos de la red neuronal"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state):
        """
        Propagación hacia adelante
        
        Args:
            state (torch.Tensor): Estado de entrada (one-hot encoded)
            
        Returns:
            torch.Tensor: Valores Q para todas las acciones
        """
        return self.network(state)

class ReplayBuffer:
    """
    Buffer de experiencia para Experience Replay
    
    Almacena experiencias pasadas y permite muestreo aleatorio
    para romper la correlación temporal en el entrenamiento.
    """
    
    def __init__(self, capacity=DQNConfig.MEMORY_SIZE):
        """
        Inicializar buffer de experiencia
        
        Args:
            capacity (int): Capacidad máxima del buffer
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Añadir experiencia al buffer
        
        Args:
            state (int): Estado actual
            action (int): Acción tomada
            reward (float): Recompensa recibida
            next_state (int): Siguiente estado
            done (bool): Si el episodio terminó
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        Muestrear un batch aleatorio de experiencias
        
        Args:
            batch_size (int): Tamaño del batch
            
        Returns:
            tuple: Batch de experiencias separadas por componente
        """
        experiences = random.sample(self.buffer, batch_size)
        
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
    Agente DQN para resolver Frozen Lake
    
    Implementa el algoritmo Deep Q-Network con Experience Replay
    y target network para estabilizar el entrenamiento.
    """
    
    def __init__(self):
        """Inicializar el agente DQN"""
        
        # Redes neuronales
        self.q_network_local = QNetwork().to(DEVICE)    # Red principal (entrenamiento)
        self.q_network_target = QNetwork().to(DEVICE)   # Red objetivo (estabilidad)
        
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
        
        # Contador de pasos para actualización
        self.t_step = 0
        
        # Métricas de entrenamiento
        self.losses = []
        self.q_values = []
        
        # Inicializar target network con los mismos pesos
        self.hard_update(self.q_network_local, self.q_network_target)
    
    def state_to_tensor(self, state):
        """
        Convertir estado entero a tensor one-hot
        
        Args:
            state (int): Estado como entero (0-15)
            
        Returns:
            np.ndarray: Estado como vector one-hot
        """
        one_hot = np.zeros(EnvConfig.STATE_SIZE)
        one_hot[state] = 1.0
        return one_hot
    
    def act(self, state, training=True):
        """
        Seleccionar acción usando política epsilon-greedy
        
        Args:
            state (int): Estado actual
            training (bool): Si estamos en modo entrenamiento
            
        Returns:
            int: Acción seleccionada (0-3)
        """
        state_tensor = torch.from_numpy(self.state_to_tensor(state)).float().unsqueeze(0).to(DEVICE)
        
        # Cambiar a modo evaluación
        self.q_network_local.eval()
        with torch.no_grad():
            action_values = self.q_network_local(state_tensor)
        # Volver a modo entrenamiento
        self.q_network_local.train()
        
        # Epsilon-greedy para exploración
        if training and random.random() < self.epsilon:
            return random.choice(np.arange(EnvConfig.ACTION_SIZE))
        else:
            return np.argmax(action_values.cpu().data.numpy())
    
    def step(self, state, action, reward, next_state, done):
        """
        Guardar experiencia y entrenar si es necesario
        
        Args:
            state (int): Estado actual
            action (int): Acción tomada
            reward (float): Recompensa recibida
            next_state (int): Siguiente estado  
            done (bool): Si el episodio terminó
        """
        # Guardar experiencia en el buffer
        self.memory.push(
            self.state_to_tensor(state),
            action,
            reward,
            self.state_to_tensor(next_state),
            done
        )
        
        # Entrenar cada UPDATE_EVERY pasos
        self.t_step = (self.t_step + 1) % DQNConfig.UPDATE_EVERY
        if self.t_step == 0:
            # Solo entrenar si tenemos suficientes experiencias
            if len(self.memory) > DQNConfig.MIN_MEMORY_SIZE:
                experiences = self.memory.sample(NetworkConfig.BATCH_SIZE)
                self.learn(experiences)
    
    def learn(self, experiences):
        """
        Actualizar parámetros usando un batch de experiencias
        
        Args:
            experiences (tuple): Batch de experiencias (s, a, r, s', done)
        """
        states, actions, rewards, next_states, dones = experiences
        
        # Obtener valores Q actuales para las acciones tomadas
        Q_targets_next = self.q_network_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Calcular valores Q objetivo usando ecuación de Bellman
        Q_targets = rewards + (DQNConfig.GAMMA * Q_targets_next * (1 - dones))
        
        # Obtener valores Q esperados de la red local
        Q_expected = self.q_network_local(states).gather(1, actions)
        
        # Calcular pérdida
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimizar pérdida
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping para estabilidad
        torch.nn.utils.clip_grad_norm_(self.q_network_local.parameters(), 1.0)
        self.optimizer.step()
        
        # Guardar métricas
        self.losses.append(loss.item())
        self.q_values.append(Q_expected.mean().item())
        
        # Actualizar target network
        self.soft_update(self.q_network_local, self.q_network_target, DQNConfig.TAU)
    
    def soft_update(self, local_model, target_model, tau):
        """
        Actualización suave de target network: θ_target = τ*θ_local + (1 - τ)*θ_target
        
        Args:
            local_model (nn.Module): Red local (fuente de pesos)
            target_model (nn.Module): Red objetivo (destino de pesos)
            tau (float): Parámetro de interpolación
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def hard_update(self, local_model, target_model):
        """
        Copia completa de pesos de la red local a la target
        
        Args:
            local_model (nn.Module): Red local (fuente)
            target_model (nn.Module): Red objetivo (destino)
        """
        target_model.load_state_dict(local_model.state_dict())
    
    def update_epsilon(self):
        """Actualizar epsilon para reducir exploración gradualmente"""
        self.epsilon = max(DQNConfig.EPSILON_END, DQNConfig.EPSILON_DECAY * self.epsilon)
    
    def save(self, filepath=TrainingConfig.MODEL_PATH):
        """
        Guardar el modelo entrenado
        
        Args:
            filepath (str): Ruta donde guardar el modelo
        """
        torch.save({
            'q_network_local_state_dict': self.q_network_local.state_dict(),
            'q_network_target_state_dict': self.q_network_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'losses': self.losses,
            'q_values': self.q_values
        }, filepath)
        print(f"{LogConfig.EMOJIS['SAVE']} Modelo guardado en {filepath}")
    
    def load(self, filepath=TrainingConfig.MODEL_PATH):
        """
        Cargar modelo previamente entrenado
        
        Args:
            filepath (str): Ruta del modelo a cargar
        """
        try:
            checkpoint = torch.load(filepath, map_location=DEVICE)
            self.q_network_local.load_state_dict(checkpoint['q_network_local_state_dict'])
            self.q_network_target.load_state_dict(checkpoint['q_network_target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', DQNConfig.EPSILON_END)
            self.losses = checkpoint.get('losses', [])
            self.q_values = checkpoint.get('q_values', [])
            print(f"{LogConfig.EMOJIS['SUCCESS']} Modelo cargado desde {filepath}")
        except FileNotFoundError:
            print(f"{LogConfig.EMOJIS['WARNING']} No se encontró el archivo {filepath}")
        except Exception as e:
            print(f"{LogConfig.EMOJIS['ERROR']} Error cargando modelo: {e}")
    
    def get_action_probabilities(self, state):
        """
        Obtener probabilidades de acción para análisis
        
        Args:
            state (int): Estado actual
            
        Returns:
            np.ndarray: Probabilidades de cada acción
        """
        state_tensor = torch.from_numpy(self.state_to_tensor(state)).float().unsqueeze(0).to(DEVICE)
        
        self.q_network_local.eval()
        with torch.no_grad():
            q_values = self.q_network_local(state_tensor)
            probabilities = F.softmax(q_values, dim=1)
        self.q_network_local.train()
        
        return probabilities.cpu().data.numpy().flatten()
    
    def get_q_values(self, state):
        """
        Obtener valores Q para un estado
        
        Args:
            state (int): Estado actual
            
        Returns:
            np.ndarray: Valores Q para cada acción
        """
        state_tensor = torch.from_numpy(self.state_to_tensor(state)).float().unsqueeze(0).to(DEVICE)
        
        self.q_network_local.eval()
        with torch.no_grad():
            q_values = self.q_network_local(state_tensor)
        self.q_network_local.train()
        
        return q_values.cpu().data.numpy().flatten()

# Función de utilidad para crear agente
def create_agent():
    """
    Crear y retornar un nuevo agente DQN
    
    Returns:
        DQNAgent: Agente DQN inicializado
    """
    return DQNAgent()

if __name__ == "__main__":
    # Crear agente para pruebas
    agent = create_agent()
    print(f"Agente DQN creado con éxito")
    print(f"Epsilon inicial: {agent.epsilon}")
    print(f"Tamaño de memoria: {len(agent.memory)}")
    
    # Probar funciones básicas
    test_state = 0
    test_action = agent.act(test_state)
    print(f"Acción para estado {test_state}: {test_action}")
    
    q_values = agent.get_q_values(test_state)
    print(f"Valores Q para estado {test_state}: {q_values}")