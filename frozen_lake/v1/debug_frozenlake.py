# debug_frozenlake.py
# Diagnóstico completo del problema

import gymnasium as gym
import numpy as np

def debug_environment():
    """Debuggear el ambiente paso a paso"""
    print("=== DEBUG DEL AMBIENTE FROZENLAKE ===\n")
    
    # Crear ambiente
    env = gym.make('FrozenLake-v1', is_slippery=False)
    
    print("1. Información del ambiente:")
    print(f"   Estados: {env.observation_space.n}")
    print(f"   Acciones: {env.action_space.n}")
    
    # Mapeo de acciones
    action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
    
    print("\n2. Prueba manual de todas las acciones desde el estado inicial:")
    
    for action in range(4):
        state, info = env.reset()
        print(f"\n   Estado inicial: {state}")
        print(f"   Acción: {action} ({action_names[action]})")
        
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        print(f"   Resultado: estado {next_state}, recompensa {reward}, terminado {done}")
    
    print("\n3. Encontrar un camino manual al objetivo:")
    
    # Camino conocido: 0→1→2→3→7→11→15 (debería funcionar)
    manual_path = [2, 2, 2, 1, 1, 1, 1]  # RIGHT, RIGHT, RIGHT, DOWN, DOWN, DOWN, DOWN
    
    state, info = env.reset()
    print(f"   Inicio: estado {state}")
    
    for step, action in enumerate(manual_path):
        print(f"   Paso {step+1}: acción {action} ({action_names[action]})")
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        print(f"   → Estado {next_state}, recompensa {reward}, terminado {done}")
        
        if done:
            if reward > 0:
                print(f"   ¡ÉXITO! Llegó al objetivo")
            else:
                print(f"   FALLÓ: cayó en agujero")
            break
        
        state = next_state
    
    print("\n4. Verificar el mapa real del ambiente:")
    
    # Explorar todos los estados posibles
    print("   Mapa de transiciones desde cada estado:")
    
    for start_state in range(16):
        if start_state in {5, 7, 11, 12, 15}:  # Skip holes and goal
            continue
            
        print(f"\n   Estado {start_state}:")
        
        for action in range(4):
            # Reset to specific state (this is tricky with Gym)
            env.reset()
            
            # Manually set state (if possible)
            try:
                env.unwrapped.s = start_state
                next_state, reward, terminated, truncated, info = env.step(action)
                print(f"     {action_names[action]}: {start_state} → {next_state}")
            except:
                print(f"     No se puede establecer estado {start_state} manualmente")
                break
    
    env.close()

def test_q_learning_update():
    """Probar la actualización de Q-Learning aisladamente"""
    print("\n=== TEST DE ACTUALIZACIÓN Q-LEARNING ===\n")
    
    # Simular escenarios
    q_table = np.zeros((16, 4))
    learning_rate = 0.1
    discount_factor = 0.9
    
    print("Escenario 1: Movimiento normal (recompensa 0)")
    state, action, reward, next_state, done = 0, 2, 0.0, 1, False
    
    current_q = q_table[state, action]
    if done:
        target = reward
    else:
        max_next_q = np.max(q_table[next_state])
        target = reward + discount_factor * max_next_q
    
    new_q = current_q + learning_rate * (target - current_q)
    q_table[state, action] = new_q
    
    print(f"   Q({state},{action}): {current_q} → {new_q}")
    print(f"   Target: {target}")
    
    print("\nEscenario 2: Llegar al objetivo (recompensa 1)")
    state, action, reward, next_state, done = 14, 2, 1.0, 15, True
    
    current_q = q_table[state, action]
    if done:
        target = reward
    else:
        max_next_q = np.max(q_table[next_state])
        target = reward + discount_factor * max_next_q
    
    new_q = current_q + learning_rate * (target - current_q)
    q_table[state, action] = new_q
    
    print(f"   Q({state},{action}): {current_q} → {new_q}")
    print(f"   Target: {target}")
    
    print(f"\nQ-Table actualizada:")
    print(f"   Estado 0: {q_table[0]}")
    print(f"   Estado 14: {q_table[14]}")

if __name__ == "__main__":
    debug_environment()
    test_q_learning_update()