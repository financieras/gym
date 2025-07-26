"""
main.py - Entrenamiento y Evaluación del Agente DQN

Este archivo contiene las funciones principales para entrenar y evaluar
el agente DQN en el entorno Frozen Lake, junto con visualización de resultados.
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
import os

from config import *
from dqn_agent import create_agent

def create_environment():
    """
    Crear el entorno Frozen Lake
    
    Returns:
        gym.Env: Entorno de Gymnasium configurado
    """
    return gym.make(
        'FrozenLake-v1',
        desc=None,
        map_name=EnvConfig.MAP_SIZE,
        is_slippery=EnvConfig.IS_SLIPPERY,
        render_mode=EnvConfig.RENDER_MODE
    )

def train_agent(agent, env, num_episodes=TrainingConfig.NUM_EPISODES):
    """
    Entrenar el agente DQN en el entorno
    
    Args:
        agent: Agente DQN
        env: Entorno de Gymnasium
        num_episodes (int): Número de episodios de entrenamiento
        
    Returns:
        tuple: (scores, moving_averages, optimal_counts)
    """
    print(f"{LogConfig.EMOJIS['TRAIN']} Iniciando entrenamiento de {num_episodes} episodios...")
    print_config()
    
    # Métricas de entrenamiento
    scores = []                     # Recompensas por episodio
    moving_averages = []            # Promedio móvil de recompensas
    optimal_counts = []             # Contador de rutas óptimas
    steps_to_goal = []              # Pasos necesarios para llegar a la meta
    scores_window = deque(maxlen=TrainingConfig.SOLVE_WINDOW)
    
    # Contadores
    optimal_solutions = 0
    successful_episodes = 0
    
    start_time = time.time()
    
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        score = 0
        steps = 0
        
        for step in range(EnvConfig.MAX_STEPS):
            # Seleccionar acción
            action = agent.act(state, training=True)
            
            # Ejecutar acción
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Guardar experiencia y entrenar
            agent.step(state, action, reward, next_state, done)
            
            # Actualizar estado y score
            state = next_state
            score += reward
            steps += 1
            
            if done:
                break
        
        # Actualizar epsilon
        agent.update_epsilon()
        
        # Guardar métricas
        scores.append(score)
        scores_window.append(score)
        
        # Contar episodios exitosos y rutas óptimas
        if score > 0:
            successful_episodes += 1
            steps_to_goal.append(steps)
            
            # Verificar si es ruta óptima
            if steps <= EvalConfig.OPTIMAL_STEPS:
                optimal_solutions += 1
                print(f"{LogConfig.EMOJIS['OPTIMAL']} ¡Ruta óptima en episodio {episode}! ({steps} pasos)")
        
        optimal_counts.append(optimal_solutions)
        
        # Calcular promedio móvil
        if len(scores_window) >= TrainingConfig.SOLVE_WINDOW:
            moving_avg = np.mean(scores_window)
            moving_averages.append(moving_avg)
            
            # Verificar si el problema está resuelto
            if moving_avg >= TrainingConfig.SOLVE_SCORE:
                print(f"\n{LogConfig.EMOJIS['SUCCESS']} ¡Entorno resuelto en {episode} episodios!")
                print(f"Promedio de últimos {TrainingConfig.SOLVE_WINDOW} episodios: {moving_avg:.3f}")
                break
        else:
            moving_averages.append(np.mean(scores))
        
        # Imprimir progreso
        if episode % TrainingConfig.PRINT_EVERY == 0:
            elapsed_time = time.time() - start_time
            success_rate = successful_episodes / episode
            optimal_rate = optimal_solutions / max(successful_episodes, 1)
            avg_steps = np.mean(steps_to_goal) if steps_to_goal else 0
            
            print(f"\nEpisodio {episode:4d}/{num_episodes}")
            print(f"  Promedio últimos {len(scores_window):3d}: {np.mean(scores_window):.3f}")
            print(f"  Tasa de éxito: {success_rate:.1%}")
            print(f"  Rutas óptimas: {optimal_solutions} ({optimal_rate:.1%} de éxitos)")
            print(f"  Pasos promedio: {avg_steps:.1f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Tiempo: {elapsed_time:.1f}s")
        
        # Guardar modelo periódicamente
        if episode % TrainingConfig.SAVE_EVERY == 0:
            agent.save(f"checkpoint_episode_{episode}.pth")
    
    # Guardar modelo final
    agent.save()
    
    # Estadísticas finales
    total_time = time.time() - start_time
    final_success_rate = successful_episodes / num_episodes
    final_optimal_rate = optimal_solutions / max(successful_episodes, 1)
    
    print(f"\n{LogConfig.EMOJIS['SUCCESS']} ¡Entrenamiento completado!")
    print(f"Tiempo total: {total_time:.1f}s ({total_time/60:.1f} minutos)")
    print(f"Episodios exitosos: {successful_episodes}/{num_episodes} ({final_success_rate:.1%})")
    print(f"Rutas óptimas: {optimal_solutions} ({final_optimal_rate:.1%} de éxitos)")
    if steps_to_goal:
        print(f"Pasos promedio a la meta: {np.mean(steps_to_goal):.1f}")
        print(f"Mínimo pasos: {min(steps_to_goal)}")
    
    return scores, moving_averages, optimal_counts

def evaluate_agent(agent, env, num_episodes=EvalConfig.EVAL_EPISODES):
    """
    Evaluar el agente entrenado
    
    Args:
        agent: Agente DQN entrenado
        env: Entorno de evaluación
        num_episodes (int): Número de episodios de evaluación
        
    Returns:
        dict: Diccionario con métricas de evaluación
    """
    print(f"\n{LogConfig.EMOJIS['EVAL']} Evaluando agente durante {num_episodes} episodios...")
    
    # Métricas de evaluación
    successful_episodes = 0
    optimal_episodes = 0
    total_steps = []
    optimal_paths_found = []
    
    # Contadores para cada ruta óptima
    path_counts = {i: 0 for i in range(len(OptimalPaths.PATHS_4X4))}
    unique_optimal_paths = set()
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        path = [state]
        steps = 0
        total_reward = 0
        
        for step in range(EnvConfig.MAX_STEPS):
            action = agent.act(state, training=False)  # Sin exploración
            state, reward, terminated, truncated, _ = env.step(action)
            
            path.append(state)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        # Analizar resultados del episodio
        if total_reward > 0:  # Episodio exitoso
            successful_episodes += 1
            total_steps.append(steps)
            
            # Verificar si es ruta óptima
            if steps <= EvalConfig.OPTIMAL_STEPS:
                optimal_episodes += 1
                path_str = " → ".join(map(str, path))
                unique_optimal_paths.add(path_str)
                optimal_paths_found.append(path)
                
                # Verificar si coincide con rutas teóricas
                for i, optimal_path in enumerate(OptimalPaths.PATHS_4X4):
                    if path == optimal_path:
                        path_counts[i] += 1
                        break
        
        # Progreso cada 200 episodios
        if (episode + 1) % 200 == 0:
            current_success_rate = successful_episodes / (episode + 1)
            current_optimal_rate = optimal_episodes / (episode + 1)
            print(f"  Progreso: {episode + 1}/{num_episodes} - "
                  f"Éxito: {current_success_rate:.1%} - "
                  f"Óptimas: {current_optimal_rate:.1%}")
    
    # Calcular métricas finales
    success_rate = successful_episodes / num_episodes
    optimal_rate = optimal_episodes / num_episodes
    optimal_rate_among_successful = optimal_episodes / max(successful_episodes, 1)
    avg_steps = np.mean(total_steps) if total_steps else 0
    
    # Resultados
    results = {
        'success_rate': success_rate,
        'optimal_rate': optimal_rate,
        'optimal_rate_among_successful': optimal_rate_among_successful,
        'avg_steps': avg_steps,
        'successful_episodes': successful_episodes,
        'optimal_episodes': optimal_episodes,
        'total_steps': total_steps,
        'unique_optimal_paths': len(unique_optimal_paths),
        'path_counts': path_counts,
        'optimal_paths_found': optimal_paths_found
    }
    
    # Imprimir resultados
    print(f"\n{LogConfig.EMOJIS['SUCCESS']} === RESULTADOS DE EVALUACIÓN ===")
    print(f"Episodios evaluados: {num_episodes}")
    print(f"Tasa de éxito: {success_rate:.1%} ({successful_episodes}/{num_episodes})")
    print(f"Tasa de rutas óptimas: {optimal_rate:.1%} ({optimal_episodes}/{num_episodes})")
    print(f"Rutas óptimas entre éxitos: {optimal_rate_among_successful:.1%}")
    
    if total_steps:
        print(f"Pasos promedio a la meta: {avg_steps:.1f}")
        print(f"Rango de pasos: {min(total_steps)} - {max(total_steps)}")
    
    print(f"Rutas óptimas únicas encontradas: {len(unique_optimal_paths)}")
    
    # Análisis de rutas teóricas
    print(f"\n{LogConfig.EMOJIS['INFO']} Análisis de rutas teóricas:")
    theoretical_found = 0
    for i, (path, count) in enumerate(path_counts.items()):
        path_str = " → ".join(map(str, OptimalPaths.PATHS_4X4[i]))
        if count > 0:
            theoretical_found += 1
            print(f"  ✅ Ruta {i+1}: {path_str} (encontrada {count} veces)")
        else:
            print(f"  ❌ Ruta {i+1}: {path_str} (no encontrada)")
    
    results['theoretical_paths_found'] = theoretical_found
    
    return results

def demonstrate_agent(agent, env, num_demos=EvalConfig.DEMO_EPISODES):
    """
    Demostrar el agente entrenado con visualización
    
    Args:
        agent: Agente DQN entrenado
        env: Entorno con renderizado
        num_demos (int): Número de demostraciones
    """
    print(f"\n{LogConfig.EMOJIS['INFO']} === DEMOSTRACIÓN DEL AGENTE ===")
    print("Leyenda del mapa:")
    print("  S = Start (Inicio)")
    print("  F = Frozen (Hielo)")
    print("  H = Hole (Agujero)")
    print("  G = Goal (Meta)")
    print("  ← ↓ → ↑ = Acciones")
    
    action_names = ["←", "↓", "→", "↑"]
    successful_demos = 0
    optimal_demos = 0
    
    for demo in range(num_demos):
        print(f"\n--- Demo {demo + 1} ---")
        
        state, _ = env.reset()
        path = [state]
        actions_taken = []
        total_reward = 0
        steps = 0
        
        print(f"Estado inicial: {state} (posición {state//4}, {state%4})")
        
        for step in range(EnvConfig.MAX_STEPS):
            action = agent.act(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            actions_taken.append(action)
            path.append(next_state)
            total_reward += reward
            steps += 1
            
            # Mostrar paso
            current_pos = f"({state//4}, {state%4})"
            next_pos = f"({next_state//4}, {next_state%4})"
            print(f"  Paso {steps}: {current_pos} --{action_names[action]}--> {next_pos}")
            
            state = next_state
            
            if terminated or truncated:
                break
        
        # Analizar resultado
        if total_reward > 0:
            successful_demos += 1
            result = "ÉXITO"
            if steps <= EvalConfig.OPTIMAL_STEPS:
                optimal_demos += 1
                result += " (ÓPTIMO)"
        else:
            result = "FALLO"
        
        print(f"Resultado: {result}")
        print(f"Pasos: {steps}, Recompensa: {total_reward}")
        print(f"Camino completo: {' → '.join(map(str, path))}")
        print(f"Acciones: {' → '.join([action_names[a] for a in actions_taken])}")
        
        # Pausa entre demos
        if demo < num_demos - 1:
            input("Presiona Enter para la siguiente demostración...")
    
    print(f"\n{LogConfig.EMOJIS['SUCCESS']} Resumen de demostraciones:")
    print(f"Exitosas: {successful_demos}/{num_demos} ({successful_demos/num_demos:.1%})")
    print(f"Óptimas: {optimal_demos}/{num_demos} ({optimal_demos/num_demos:.1%})")

def plot_training_results(scores, moving_averages, optimal_counts, save_path=TrainingConfig.RESULTS_PATH):
    """
    Visualizar resultados del entrenamiento
    
    Args:
        scores (list): Recompensas por episodio
        moving_averages (list): Promedios móviles
        optimal_counts (list): Contador de rutas óptimas
        save_path (str): Ruta para guardar la gráfica
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    episodes = range(1, len(scores) + 1)
    
    # Gráfica 1: Recompensas por episodio
    ax1.plot(episodes, scores, alpha=0.6, color='lightblue', linewidth=0.8)
    ax1.plot(episodes, moving_averages, color='red', linewidth=2, label=f'Promedio móvil ({TrainingConfig.SOLVE_WINDOW})')
    ax1.axhline(y=TrainingConfig.SOLVE_SCORE, color='green', linestyle='--', label='Objetivo')
    ax1.set_xlabel('Episodio')
    ax1.set_ylabel('Recompensa')
    ax1.set_title('Progreso del Entrenamiento')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfica 2: Rutas óptimas acumuladas
    ax2.plot(episodes, optimal_counts, color='orange', linewidth=2)
    ax2.set_xlabel('Episodio')
    ax2.set_ylabel('Rutas Óptimas Acumuladas')
    ax2.set_title('Descubrimiento de Rutas Óptimas')
    ax2.grid(True, alpha=0.3)
    
    # Gráfica 3: Distribución de recompensas
    ax3.hist(scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(x=np.mean(scores), color='red', linestyle='--', label=f'Promedio: {np.mean(scores):.3f}')
    ax3.set_xlabel('Recompensa')
    ax3.set_ylabel('Frecuencia')
    ax3.set_title('Distribución de Recompensas')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Gráfica 4: Tasa de éxito por ventana
    window_size = 100
    success_rates = []
    for i in range(window_size, len(scores) + 1):
        window_scores = scores[i-window_size:i]
        success_rate = sum(1 for s in window_scores if s > 0) / window_size
        success_rates.append(success_rate)
    
    if success_rates:
        ax4.plot(range(window_size, len(scores) + 1), success_rates, color='green', linewidth=2)
        ax4.axhline(y=EvalConfig.TARGET_SUCCESS_RATE, color='red', linestyle='--', label='Objetivo')
        ax4.set_xlabel('Episodio')
        ax4.set_ylabel('Tasa de Éxito')
        ax4.set_title(f'Tasa de Éxito (ventana de {window_size})')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"{LogConfig.EMOJIS['SAVE']} Gráfica guardada en {save_path}")
    plt.show()

def main():
    """Función principal"""
    print(f"{LogConfig.COLORS['BOLD']}🧊 === FROZEN LAKE DQN - IMPLEMENTACIÓN SIMPLIFICADA ==={LogConfig.COLORS['END']}")
    
    # Validar configuración
    validate_config()
    
    # Crear entorno y agente
    env = create_environment()
    agent = create_agent()
    
    print(f"\n{LogConfig.EMOJIS['INFO']} Configuración del entorno:")
    print(f"  Espacio de observación: {env.observation_space}")
    print(f"  Espacio de acción: {env.action_space}")
    print(f"  Mapa: {EnvConfig.MAP_SIZE}")
    print(f"  Resbaladizo: {'Sí' if EnvConfig.IS_SLIPPERY else 'No'}")
    
    # Menú de opciones
    print(f"\n{LogConfig.EMOJIS['INFO']} Opciones disponibles:")
    print("1. Entrenar desde cero")
    print("2. Cargar modelo y evaluar")
    print("3. Cargar modelo y demostrar")
    print("4. Entrenar, evaluar y demostrar (completo)")
    
    choice = input("\nSelecciona una opción (1-4): ").strip()
    
    if choice == "1":
        # Solo entrenar
        scores, moving_averages, optimal_counts = train_agent(agent, env)
        plot_training_results(scores, moving_averages, optimal_counts)
        
    elif choice == "2":
        # Cargar y evaluar
        agent.load()
        results = evaluate_agent(agent, env)
        
    elif choice == "3":
        # Cargar y demostrar
        agent.load()
        if EvalConfig.RENDER_DEMO:
            demo_env = gym.make('FrozenLake-v1', map_name=EnvConfig.MAP_SIZE, 
                             is_slippery=EnvConfig.IS_SLIPPERY, render_mode='human')
            demonstrate_agent(agent, demo_env)
            demo_env.close()
        else:
            demonstrate_agent(agent, env)
            
    elif choice == "4":
        # Flujo completo
        print(f"\n{LogConfig.EMOJIS['TRAIN']} Iniciando flujo completo...")
        
        # Entrenar
        scores, moving_averages, optimal_counts = train_agent(agent, env)
        plot_training_results(scores, moving_averages, optimal_counts)
        
        # Evaluar
        results = evaluate_agent(agent, env)
        
        # Demostrar si el rendimiento es bueno
        if results['success_rate'] >= 0.5:
            print(f"\n{LogConfig.EMOJIS['SUCCESS']} ¡Buen rendimiento! Mostrando demostraciones...")
            demonstrate_agent(agent, env)
        else:
            print(f"\n{LogConfig.EMOJIS['WARNING']} Rendimiento bajo. Considera entrenar más episodios.")
            
        # Resumen final
        print(f"\n{LogConfig.EMOJIS['SUCCESS']} === RESUMEN FINAL ===")
        print(f"Tasa de éxito: {results['success_rate']:.1%}")
        print(f"Rutas óptimas: {results['optimal_rate']:.1%}")
        print(f"Rutas teóricas encontradas: {results['theoretical_paths_found']}/3")
        
        if results['success_rate'] >= EvalConfig.TARGET_SUCCESS_RATE:
            print(f"{LogConfig.EMOJIS['SUCCESS']} ¡Objetivo de tasa de éxito alcanzado!")
        
        if results['optimal_rate'] >= EvalConfig.TARGET_OPTIMAL_RATE:
            print(f"{LogConfig.EMOJIS['OPTIMAL']} ¡Objetivo de rutas óptimas alcanzado!")
            
    else:
        print(f"{LogConfig.EMOJIS['ERROR']} Opción no válida.")
    
    env.close()
    print(f"\n{LogConfig.EMOJIS['SUCCESS']} ¡Programa terminado!")

if __name__ == "__main__":
    main()