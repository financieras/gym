# frozenlake_qlearning.py
# Q-Learning en FrozenLake (versión no resbaladiza)

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("=== Q-LEARNING EN FROZENLAKE ===\n")
    
    # ========== PASO 1: CONFIGURACIÓN DEL AMBIENTE ==========
    print("PASO 1: Configuración del ambiente")
    env = gym.make('FrozenLake-v1', is_slippery=False)
    
    print(f"Espacios de estado: {env.observation_space.n}")
    print(f"Espacios de acción: {env.action_space.n}")
    
    action_names = {0: "Izquierda", 1: "Abajo", 2: "Derecha", 3: "Arriba"}
    print(f"Acciones: {action_names}")
    
    # ========== VISUALIZACIÓN DEL MAPA ==========
    def visualize_map(size="4x4"):
        """Visualiza el mapa de FrozenLake de forma clara"""
        maps = {
            "4x4": [
                "SFFF",
                "FHFH", 
                "FFFH",
                "HFFG"
            ],
            "8x8": [
                "SFFFFFFF",
                "FFFFFFFF", 
                "FFFHFFFF",
                "FFFFFHFF",
                "FFFHFFFF",
                "FHHFFFHF",
                "FHFFHFHF",
                "FFFHFFFG"
            ]
        }
        
        if size not in maps:
            print(f"Tamaño {size} no soportado")
            return
        
        print(f"\nMapa visual {size}:")
        print("S=Inicio, F=Hielo, H=Agujero, G=Meta")
        
        # Mostrar el mapa con números de estado
        map_grid = maps[size]
        grid_size = int(size[0])  # 4 o 8
        
        # Línea de números de columna
        print("   " + " ".join(f"{i:2d}" for i in range(grid_size)))
        
        for row_idx, row in enumerate(map_grid):
            # Número de fila + contenido
            print(f"{row_idx:2d} ", end="")
            for col_idx, cell in enumerate(row):
                state_num = row_idx * grid_size + col_idx
                print(f"{cell:2s} ", end="")
            
            # Mostrar números de estado al lado
            print("  [", end="")
            for col_idx in range(grid_size):
                state_num = row_idx * grid_size + col_idx
                print(f"{state_num:2d}", end="")
                if col_idx < grid_size - 1:
                    print(",", end="")
            print("]")
        
        return map_grid
    
    # Visualizar el mapa actual (4x4)
    current_map = visualize_map("4x4")
    
    print(f"\nMapa en formato de matriz:")
    for i, row in enumerate(current_map):
        print(f"  \"{row}\"{',' if i < len(current_map)-1 else ''}")
    
    # También mostrar el 8x8 como ejemplo
    print(f"\nEjemplo de mapa 8x8:")
    visualize_map("8x8")
    
    # ========== PASO 2: MAPEO DE ESTADOS ==========
    print(f"\nPASO 2: Funciones de mapeo")
    
    def state_to_coordinates(state):
        """Convierte estado a coordenadas (fila, columna)"""
        row = state // 4
        col = state % 4
        return row, col
    
    def coordinates_to_state(row, col):
        """Convierte coordenadas a estado"""
        return row * 4 + col
    
    # ========== PASO 3: INICIALIZACIÓN Q-TABLE ==========
    print(f"\nPASO 3: Inicialización de Q-Table")
    
    n_states = env.observation_space.n  # 16
    n_actions = env.action_space.n      # 4
    
    # Crear Q-Table vacía
    q_table = np.zeros((n_states, n_actions))
    print(f"Q-Table creada: {q_table.shape}")
    
    # Parámetros de Q-Learning
    learning_rate = 0.1    # α
    discount_factor = 0.9  # γ  
    epsilon = 0.1          # ε
    
    print(f"Parámetros: α={learning_rate}, γ={discount_factor}, ε={epsilon}")
    
    # Estados especiales
    start_state = 0
    goal_state = 15
    hole_states = {5, 7, 11, 12}
    
    print(f"Inicio: {start_state}, Meta: {goal_state}, Agujeros: {hole_states}")
    
    # Mostrar Q-Table inicial (sample)
    print(f"\nQ-Table inicial (muestra):")
    print("Estado  [Izq, Abajo, Der, Arriba]")
    for i in range(0, 16, 4):  # Mostrar cada 4 estados
        print(f"{i:2d}:     {q_table[i]}")
    
    # ========== PASO 4: ESTRATEGIA EPSILON-GREEDY ==========
    print(f"\nPASO 4: Implementación de epsilon-greedy")
    
    def choose_action(state, q_table, epsilon):
        """
        Estrategia epsilon-greedy para elegir acción
        - Con probabilidad epsilon: acción aleatoria (exploración)
        - Con probabilidad (1-epsilon): mejor acción conocida (explotación)
        """
        if np.random.random() < epsilon:
            # Exploración: acción aleatoria
            action = env.action_space.sample()
            print(f"  Exploración: acción aleatoria {action}")
        else:
            # Explotación: mejor acción según Q-Table
            action = np.argmax(q_table[state])
            print(f"  Explotación: mejor acción {action}")
        
        return action
    
    # ========== PASO 5: ECUACIÓN DE BELLMAN ==========
    print(f"\nPASO 5: Implementación de la ecuación de Bellman")
    
    def update_q_table(q_table, state, action, reward, next_state, learning_rate, discount_factor, done):
        """
        Actualiza Q-Table usando la ecuación de Bellman:
        Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        IMPORTANTE: Si el episodio terminó, no hay estado futuro
        """
        # Valor Q actual
        current_q = q_table[state, action]
        
        # Si el episodio terminó, no hay valor futuro
        if done:
            target = reward
        else:
            # Mejor valor Q del siguiente estado
            max_next_q = np.max(q_table[next_state])
            target = reward + discount_factor * max_next_q
        
        # Calcular nuevo valor Q
        new_q = current_q + learning_rate * (target - current_q)
        
        # Actualizar Q-Table
        q_table[state, action] = new_q
        
        print(f"  Actualización Q({state},{action}): {current_q:.3f} → {new_q:.3f} (done={done})")
        return q_table
    
    # ========== PASO 6: DEMOSTRACIÓN CON UN EPISODIO ==========
    print(f"\nPASO 6: Demostración con un episodio de entrenamiento")
    
    # Reiniciar ambiente
    env = gym.make('FrozenLake-v1', is_slippery=False)
    state, info = env.reset()
    
    print(f"\nEpisodio de demostración:")
    print(f"Estado inicial: {state}")
    
    step = 0
    max_steps = 10  # Limitar para demostración
    
    while step < max_steps:
        print(f"\n--- Paso {step + 1} ---")
        print(f"Estado actual: {state}")
        
        # Elegir acción usando epsilon-greedy
        action = choose_action(state, q_table, epsilon)
        print(f"Acción elegida: {action} ({action_names[action]})")
        
        # Ejecutar acción
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        print(f"Resultado: estado {next_state}, recompensa {reward}")
        
        # Actualizar Q-Table
        q_table = update_q_table(q_table, state, action, reward, next_state, 
                                learning_rate, discount_factor, done)
        
        # Actualizar estado
        state = next_state
        step += 1
        
        # Verificar si terminó el episodio
        if done:
            if reward > 0:
                print(f"¡Éxito! Llegó al objetivo en {step} pasos")
            else:
                print(f"Falló: cayó en un agujero en el paso {step}")
            break
    
    if step >= max_steps:
        print(f"Demostración limitada a {max_steps} pasos")
    
    # Mostrar Q-Table después de la demostración
    print(f"\nQ-Table después de 1 episodio (muestra):")
    print("Estado  [Izq, Abajo, Der, Arriba]")
    for i in range(0, 16, 4):
        print(f"{i:2d}:     {q_table[i]}")
    
    env.close()
    
    # ========== PASO 7: ENTRENAMIENTO COMPLETO ==========
    print(f"\nPASO 7: Entrenamiento completo del agente")
    
    def train_agent(n_episodes=5000):  # Más episodios
        """Entrena el agente por múltiples episodios"""
        env = gym.make('FrozenLake-v1', is_slippery=False)
        
        # Reinicializar Q-Table para entrenamiento limpio
        q_table = np.zeros((n_states, n_actions))
        
        # Métricas de entrenamiento
        rewards_per_episode = []
        success_rate_window = []
        
        # Parámetros de entrenamiento optimizados
        epsilon_start = 1.0      # 100% exploración al inicio
        epsilon_end = 0.01       # 1% exploración al final
        epsilon_decay = 0.9995   # Decay más lento
        current_epsilon = epsilon_start
        
        # Parámetros de aprendizaje
        learning_rate = 0.8      # Aprendizaje más agresivo
        discount_factor = 0.95   # Más importancia al futuro
        
        print(f"Iniciando entrenamiento por {n_episodes} episodios...")
        print(f"Epsilon: {epsilon_start} → {epsilon_end}")
        print(f"Learning rate: {learning_rate}, Discount: {discount_factor}")
        
        for episode in range(n_episodes):
            state, info = env.reset()
            total_reward = 0
            step_count = 0
            max_steps_per_episode = 200  # Más pasos permitidos
            
            while step_count < max_steps_per_episode:
                # Elegir acción con epsilon dinámico
                if np.random.random() < current_epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(q_table[state])
                
                # Ejecutar acción
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Actualizar Q-Table con manejo correcto de estados terminales
                current_q = q_table[state, action]
                if done:
                    target = reward
                else:
                    max_next_q = np.max(q_table[next_state])
                    target = reward + discount_factor * max_next_q
                
                new_q = current_q + learning_rate * (target - current_q)
                q_table[state, action] = new_q
                
                # Actualizar métricas
                total_reward += reward
                state = next_state
                step_count += 1
                
                if done:
                    break
            
            # Decay epsilon más lento
            current_epsilon = max(epsilon_end, current_epsilon * epsilon_decay)
            
            # Guardar recompensa del episodio
            rewards_per_episode.append(total_reward)
            
            # Mostrar progreso cada 500 episodios
            if (episode + 1) % 500 == 0:
                recent_avg = np.mean(rewards_per_episode[-100:]) if len(rewards_per_episode) >= 100 else np.mean(rewards_per_episode)
                print(f"Episodio {episode + 1}: Tasa éxito últimos 100 = {recent_avg:.2%}, Epsilon = {current_epsilon:.3f}")
        
        env.close()
        return q_table, rewards_per_episode, success_rate_window
    
    # Ejecutar entrenamiento
    trained_q_table, rewards_history, success_history = train_agent(5000)
    
    # ========== VERIFICACIÓN: CAMINO CONOCIDO ==========
    print(f"\nVERIFICACIÓN: Probando camino conocido al objetivo")
    print("Camino: 0→4→8→9→10→14→15 (DOWN,DOWN,RIGHT,RIGHT,DOWN,RIGHT)")
    
    env = gym.make('FrozenLake-v1', is_slippery=False)
    state, info = env.reset()
    known_path = [1, 1, 2, 2, 1, 2]  # DOWN, DOWN, RIGHT, RIGHT, DOWN, RIGHT
    
    print(f"Inicio: {state}", end="")
    success = True
    
    for i, action in enumerate(known_path):
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f" → {next_state}", end="")
        state = next_state
        
        if done:
            if reward > 0:
                print(f" ¡LLEGÓ AL OBJETIVO!")
            else:
                print(f" (cayó en agujero)")
                success = False
            break
    
    env.close()
    if success:
        print("✓ Camino válido confirmado")
    else:
        print("✗ Camino no funciona")
    
    # ========== PASO 8: ANÁLISIS DE RESULTADOS ==========
    print(f"\nPASO 8: Análisis de resultados del entrenamiento")
    
    # Estadísticas finales
    total_episodes = len(rewards_history)
    successful_episodes = sum(rewards_history)
    final_success_rate = successful_episodes / total_episodes
    
    print(f"Estadísticas finales:")
    print(f"Total episodios: {total_episodes}")
    print(f"Episodios exitosos: {int(successful_episodes)}")
    print(f"Tasa de éxito final: {final_success_rate:.2%}")
    
    # Tasa de éxito en los últimos 100 episodios
    if len(rewards_history) >= 100:
        recent_success_rate = np.mean(rewards_history[-100:])
        print(f"Tasa de éxito últimos 100 episodios: {recent_success_rate:.2%}")
    
    # Mostrar Q-Table entrenada
    print(f"\nQ-Table después del entrenamiento:")
    print("Estado  [Izq, Abajo, Der, Arriba]")
    for i in range(n_states):
        best_action = np.argmax(trained_q_table[i])
        print(f"{i:2d}:     {trained_q_table[i]} → Mejor: {action_names[best_action]}")
    
    # ========== PASO 9: POLÍTICA APRENDIDA ==========
    print(f"\nPASO 9: Política óptima aprendida")
    
    def show_policy(q_table):
        """Muestra la política aprendida en formato visual"""
        policy_symbols = {0: "←", 1: "↓", 2: "→", 3: "↑"}
        
        # Mapa base
        base_map = [
            "SFFF",
            "FHFH", 
            "FFFH",
            "HFFG"
        ]
        
        print("Política aprendida (flechas muestran mejor acción):")
        print("S=Inicio, G=Meta, H=Agujero, ←↓→↑=Mejor acción")
        
        # Línea de números de columna
        print("   " + " ".join(f"{i:2d}" for i in range(4)))
        
        for row in range(4):
            print(f"{row:2d} ", end="")
            for col in range(4):
                state = row * 4 + col
                base_cell = base_map[row][col]
                
                if base_cell in ['S', 'G', 'H']:
                    symbol = base_cell
                else:
                    best_action = np.argmax(q_table[state])
                    symbol = policy_symbols[best_action]
                
                print(f"{symbol:2s} ", end="")
            
            # Mostrar valores Q más altos de la fila
            print("  [", end="")
            for col in range(4):
                state = row * 4 + col
                max_q = np.max(q_table[state])
                print(f"{max_q:4.2f}", end="")
                if col < 3:
                    print(",", end="")
            print("]")
        
        print(f"\nInterpretación:")
        print(f"- Las flechas indican la mejor acción desde cada estado")
        print(f"- Los números entre [ ] son los valores Q máximos por estado")
        print(f"- Valores más altos = estados más valiosos (cerca del objetivo)")
    
    show_policy(trained_q_table)
    
    # ========== PASO 10: PRUEBA DEL AGENTE ENTRENADO ==========
    print(f"\nPASO 10: Prueba del agente entrenado (sin exploración)")
    
    def test_trained_agent(q_table, n_test_episodes=10):
        """Prueba el agente entrenado sin exploración"""
        env = gym.make('FrozenLake-v1', is_slippery=False)
        successes = 0
        
        for episode in range(n_test_episodes):
            state, info = env.reset()
            step_count = 0
            path = [state]
            
            print(f"\nEpisodio de prueba {episode + 1}:")
            print(f"Camino: {state}", end="")
            
            while step_count < 20:  # Máximo 20 pasos
                # Solo explotación (sin exploración)
                action = np.argmax(q_table[state])
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                print(f" → {next_state}", end="")
                path.append(next_state)
                state = next_state
                step_count += 1
                
                if done:
                    if reward > 0:
                        print(f" ¡ÉXITO!")
                        successes += 1
                    else:
                        print(f" (cayó en agujero)")
                    break
            
            if step_count >= 20:
                print(f" (demasiados pasos)")
        
        env.close()
        success_rate = successes / n_test_episodes
        print(f"\nResultados de prueba: {successes}/{n_test_episodes} éxitos ({success_rate:.1%})")
        return success_rate
    
    test_success_rate = test_trained_agent(trained_q_table)
    
    print(f"\n¡Entrenamiento completo!")
    print(f"El agente aprendió a jugar FrozenLake con {test_success_rate:.1%} de éxito")
    
    # ========== ANÁLISIS DE CAMINOS ALTERNATIVOS ==========
    print(f"\n=== ANÁLISIS DE CAMINOS ALTERNATIVOS ===")
    
    def find_all_optimal_paths():
        """Encuentra todos los caminos óptimos al objetivo"""
        action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
        
        # Definir el mapa manualmente basado en nuestro debug
        transitions = {
            0: {0: 0, 1: 4, 2: 1, 3: 0},
            1: {0: 0, 1: 5, 2: 2, 3: 1},  # 5 es agujero
            2: {0: 1, 1: 6, 2: 3, 3: 2},
            3: {0: 2, 1: 7, 2: 3, 3: 3},  # 7 es agujero
            4: {0: 4, 1: 8, 2: 5, 3: 0},  # 5 es agujero
            6: {0: 5, 1: 10, 2: 7, 3: 2}, # 5,7 son agujeros
            8: {0: 8, 1: 12, 2: 9, 3: 4}, # 12 es agujero
            9: {0: 8, 1: 13, 2: 10, 3: 5}, # 5 es agujero
            10: {0: 9, 1: 14, 2: 11, 3: 6}, # 11 es agujero
            13: {0: 12, 1: 13, 2: 14, 3: 9}, # 12 es agujero
            14: {0: 13, 1: 14, 2: 15, 3: 10}  # 15 es objetivo
        }
        
        holes = {5, 7, 11, 12}
        goal = 15
        
        def dfs_paths(current_state, target_state, path, visited, all_paths, max_depth=8):
            """Búsqueda en profundidad para encontrar todos los caminos"""
            if len(path) > max_depth:  # Evitar caminos demasiado largos
                return
                
            if current_state == target_state:
                # Convertir path a secuencia de estados para evitar duplicados
                state_sequence = [0] + [step[2] for step in path]
                if state_sequence not in [([0] + [s[2] for s in p]) for p in all_paths]:
                    all_paths.append(path.copy())
                return
            
            if current_state in holes or current_state in visited:
                return
            
            visited.add(current_state)
            
            if current_state in transitions:
                for action, next_state in transitions[current_state].items():
                    if next_state not in visited and next_state not in holes:
                        path.append((current_state, action, next_state))
                        dfs_paths(next_state, target_state, path, visited, all_paths, max_depth)
                        path.pop()
            
            visited.remove(current_state)
        
        # Encontrar todos los caminos desde el estado 0 al 15
        all_paths = []
        dfs_paths(0, goal, [], set(), all_paths)
        
        return all_paths
    
    def analyze_paths():
        """Analiza y muestra los caminos encontrados"""
        paths = find_all_optimal_paths()
        action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
        
        print(f"Caminos válidos encontrados: {len(paths)}")
        
        if len(paths) == 0:
            print("No se encontraron caminos válidos.")
            return
        
        # Ordenar por longitud
        paths.sort(key=len)
        min_length = len(paths[0]) if paths else 0
        
        # Mostrar TODOS los caminos encontrados
        print(f"\nTodos los caminos encontrados:")
        for i, path in enumerate(paths):
            states = [0] + [step[2] for step in path]
            actions = [action_names[step[1]] for step in path]
            
            is_optimal = len(path) == min_length
            status = "ÓPTIMO" if is_optimal else f"LARGO ({len(path)} pasos)"
            
            print(f"\nCamino {i+1} ({status}):")
            print(f"  Estados: {' → '.join(map(str, states))}")
            print(f"  Acciones: {' → '.join(actions)}")
        
        # Contar óptimos
        optimal_count = sum(1 for path in paths if len(path) == min_length)
        longer_count = len(paths) - optimal_count
        
        print(f"\nResumen:")
        print(f"  - Caminos óptimos ({min_length} pasos): {optimal_count}")
        print(f"  - Caminos más largos: {longer_count}")
        print(f"  - Total: {len(paths)}")
        
        return paths
    
    # Ejecutar análisis
    optimal_paths = analyze_paths()
    
    # ========== COMPARACIÓN CON LA POLÍTICA APRENDIDA ==========
    print(f"\n=== COMPARACIÓN CON POLÍTICA APRENDIDA ===")
    
    def get_learned_path(q_table, max_steps=10):
        """Obtiene el camino que seguiría la política aprendida"""
        env = gym.make('FrozenLake-v1', is_slippery=False)
        state, info = env.reset()
        path = [state]
        actions = []
        
        for _ in range(max_steps):
            action = np.argmax(q_table[state])
            actions.append(action)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            path.append(next_state)
            state = next_state
            
            if terminated or truncated:
                break
        
        env.close()
        return path, actions
    
    learned_path, learned_actions = get_learned_path(trained_q_table)
    action_names_full = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
    
    print(f"Camino aprendido por Q-Learning:")
    print(f"  Estados: {' → '.join(map(str, learned_path))}")
    print(f"  Acciones: {' → '.join([action_names_full[a] for a in learned_actions])}")
    print(f"  Longitud: {len(learned_actions)} pasos")
    
    # Verificar si es óptimo
    is_optimal = False
    if optimal_paths:
        min_optimal_length = len(optimal_paths[0])
        is_optimal = len(learned_actions) == min_optimal_length and learned_path[-1] == 15
        print(f"  ¿Es óptimo? {'✓ Sí' if is_optimal else '✗ No'}")
    
    # ========== VISUALIZACIÓN DE CAMINOS EN EL MAPA ==========
    print(f"\n=== CAMINOS VISUALIZADOS EN EL MAPA ===")
    
    def visualize_path_on_map(path_states, path_name="Camino"):
        """Visualiza un camino específico en el mapa"""
        base_map = [
            "SFFF",
            "FHFH", 
            "FFFH",
            "HFFG"
        ]
        
        print(f"\n{path_name}: {' → '.join(map(str, path_states))}")
        print("   " + " ".join(f"{i:2d}" for i in range(4)))
        
        for row in range(4):
            print(f"{row:2d} ", end="")
            for col in range(4):
                state = row * 4 + col
                base_cell = base_map[row][col]
                
                if state in path_states:
                    position = path_states.index(state)
                    if state == 0:
                        symbol = "S"  # Inicio
                    elif state == 15:
                        symbol = "G"  # Meta
                    else:
                        symbol = str(position)  # Número de paso
                elif base_cell == 'H':
                    symbol = "H"  # Agujero
                else:
                    symbol = "·"  # Espacio vacío
                
                print(f"{symbol:2s} ", end="")
            print()
    
    # Mostrar los caminos óptimos en el mapa
    if optimal_paths:
        paths_by_length = {}
        for path in optimal_paths:
            states = [0] + [step[2] for step in path]
            length = len(path)
            if length not in paths_by_length:
                paths_by_length[length] = []
            paths_by_length[length].append(states)
        
        min_length = min(paths_by_length.keys())
        optimal_paths_states = paths_by_length[min_length]
        
        for i, path_states in enumerate(optimal_paths_states):
            visualize_path_on_map(path_states, f"Camino óptimo {i+1}")
        
        # Mostrar el camino aprendido
        visualize_path_on_map(learned_path, "Camino aprendido por Q-Learning")
    
    print(f"\n{'='*50}")
    print(f"CONCLUSIÓN:")
    print(f"Q-Learning encontró {'un camino óptimo' if is_optimal else 'un camino'}")
    print(f"Existen múltiples soluciones óptimas de igual longitud")
    print(f"El algoritmo converge a una de ellas (comportamiento normal)")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()