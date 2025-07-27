"""
main.py - Entrenamiento y Evaluación del Agente DQN Mejorado

Este archivo contiene las funciones principales para entrenar y evaluar
el agente DQN mejorado en el entorno Frozen Lake, con enfoque en encontrar
rutas óptimas de 6 pasos.
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
import os

from config import *
from agent import create_agent

def create_environment():
    """Crear el entorno Frozen Lake"""
    return gym.make(
        'FrozenLake-v1',
        desc=None,
        map_name=EnvConfig.MAP_SIZE,
        is_slippery=EnvConfig.IS_SLIPPERY,
        render_mode=EnvConfig.RENDER_MODE
    )

def train_agent(agent, env, num_episodes=TrainingConfig.NUM_EPISODES):
    """
    Entrenar el agente DQN mejorado con recompensas moldeadas
    
    Args:
        agent: Agente DQN mejorado
        env: Entorno de Gymnasium
        num_episodes (int): Número de episodios de entrenamiento
        
    Returns:
        tuple: (scores, moving_averages, optimal_counts, steps_per_episode)
    """
    print(f"{LogConfig.EMOJIS['TRAIN']} Iniciando entrenamiento optimizado para rutas óptimas")
    print(f"🎯 Objetivo: Encontrar las 3 rutas teóricas de 6 pasos")
    print_config()
    
    # Métricas de entrenamiento
    scores = []
    moving_averages = []
    optimal_counts = []
    steps_per_episode = []
    scores_window = deque(maxlen=TrainingConfig.SOLVE_WINDOW)
    
    start_time = time.time()
    
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        score = 0
        steps = 0
        episode_path = [state]
        
        for step in range(EnvConfig.MAX_STEPS):
            # Seleccionar acción
            action = agent.act(state, training=True)
            
            # Ejecutar acción
            next_state, env_reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            episode_path.append(next_state)
            
            # Entrenar agente con recompensa moldeada
            full_path = episode_path if done else None
            agent.step(state, action, env_reward, next_state, done, steps, full_path)
            
            # Actualizar estado y score
            state = next_state
            score += env_reward  # Usar recompensa original para métricas
            
            if done:
                break
        
        # Finalizar episodio
        agent.end_episode(score, steps)
        agent.update_epsilon()
        
        # Guardar métricas
        scores.append(score)
        scores_window.append(score)
        steps_per_episode.append(steps)
        optimal_counts.append(agent.optimal_solutions)
        
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
        
        # Imprimir progreso cada PRINT_EVERY episodios
        if episode % TrainingConfig.PRINT_EVERY == 0:
            elapsed_time = time.time() - start_time
            stats = agent.get_statistics()
            
            print(f"\nEpisodio {episode:4d}/{num_episodes}")
            print(f"  Promedio últimos {len(scores_window):3d}: {np.mean(scores_window):.3f}")
            print(f"  Tasa de éxito: {stats['success_rate']:.1%}")
            print(f"  Rutas óptimas: {agent.optimal_solutions} ({stats['optimal_rate']:.1%} de éxitos)")
            print(f"  Rutas teóricas: {stats['theoretical_paths_found']}/3")
            print(f"  Pasos promedio: {stats['avg_steps']:.1f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Tiempo: {elapsed_time:.1f}s")
            
            # Mostrar rutas teóricas encontradas
            if any(count > 0 for count in agent.theoretical_paths_found.values()):
                found_paths = []
                for i, count in agent.theoretical_paths_found.items():
                    if count > 0:
                        found_paths.append(f"Ruta {i+1}({count}x)")
                print(f"  Encontradas: {', '.join(found_paths)}")
                
            # Parada temprana si ya encontró todas las rutas múltiples veces
            if (episode > 2000 and 
                stats['theoretical_paths_found'] == 3 and 
                all(count >= 3 for count in agent.theoretical_paths_found.values()) and
                stats['optimal_rate'] >= 0.05):  # 5% de rutas óptimas
                print(f"\n🎯 ¡CONVERGENCIA ÓPTIMA ALCANZADA!")
                print(f"Todas las rutas encontradas múltiples veces con buena tasa")
                break
        
        # Guardar modelo periódicamente
        if episode % TrainingConfig.SAVE_EVERY == 0:
            agent.save(f"checkpoint_episode_{episode}.pth")
    
    # Guardar modelo final
    agent.save()
    
    # Estadísticas finales
    total_time = time.time() - start_time
    final_stats = agent.get_statistics()
    
    print(f"\n{LogConfig.EMOJIS['SUCCESS']} ¡Entrenamiento completado!")
    print(f"Tiempo total: {total_time:.1f}s ({total_time/60:.1f} minutos)")
    print(f"Episodios exitosos: {agent.successful_episodes}/{episode} ({final_stats['success_rate']:.1%})")
    print(f"Rutas óptimas encontradas: {agent.optimal_solutions}")
    print(f"Tasa de rutas óptimas: {final_stats['optimal_rate']:.1%} de éxitos")
    print(f"Rutas teóricas encontradas: {final_stats['theoretical_paths_found']}/3")
    print(f"Pasos promedio: {final_stats['avg_steps']:.1f}")
    
    # Detalles de rutas teóricas
    print(f"\nDetalle de rutas teóricas:")
    for i, count in agent.theoretical_paths_found.items():
        path_str = " → ".join(map(str, OptimalPaths.PATHS_4X4[i]))
        status = "✅" if count > 0 else "❌"
        print(f"  {status} Ruta {i+1}: {path_str} ({count} veces)")
    
    # Distribución temporal de descubrimientos
    if agent.optimal_episodes:
        early_optimal = sum(1 for ep in agent.optimal_episodes if ep <= episode // 3)
        mid_optimal = sum(1 for ep in agent.optimal_episodes if episode // 3 < ep <= 2 * episode // 3)
        late_optimal = sum(1 for ep in agent.optimal_episodes if ep > 2 * episode // 3)
        
        print(f"\nDistribución temporal de rutas óptimas:")
        print(f"  Fase inicial (1-{episode//3}): {early_optimal}")
        print(f"  Fase media ({episode//3+1}-{2*episode//3}): {mid_optimal}")
        print(f"  Fase final ({2*episode//3+1}-{episode}): {late_optimal}")
    
    return scores, moving_averages, optimal_counts, steps_per_episode

def evaluate_agent(agent, env, num_episodes=EvalConfig.EVAL_EPISODES):
    """
    Evaluar el agente entrenado con análisis detallado de rutas óptimas
    
    Args:
        agent: Agente DQN entrenado
        env: Entorno de evaluación
        num_episodes (int): Número de episodios de evaluación
        
    Returns:
        dict: Diccionario con métricas de evaluación detalladas
    """
    print(f"\n{LogConfig.EMOJIS['EVAL']} Evaluando agente durante {num_episodes} episodios...")
    
    # Reiniciar estadísticas para evaluación limpia
    initial_stats = agent.get_statistics()
    agent.reset_statistics()
    
    # Métricas de evaluación
    successful_episodes = 0
    optimal_episodes = 0
    total_steps = []
    all_paths = []
    unique_optimal_paths = set()
    
    # Contadores específicos para rutas teóricas
    theoretical_found = {i: 0 for i in range(len(OptimalPaths.PATHS_4X4))}
    
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
        
        # Actualizar estadísticas del agente para evaluación
        agent.end_episode(total_reward, steps)
        
        # Analizar resultados del episodio
        if total_reward > 0:  # Episodio exitoso
            successful_episodes += 1
            total_steps.append(steps)
            all_paths.append(path)
            
            # Verificar si es ruta óptima
            if steps <= EvalConfig.OPTIMAL_STEPS:
                optimal_episodes += 1
                path_str = " → ".join(map(str, path))
                unique_optimal_paths.add(path_str)
                
                # Verificar si coincide con rutas teóricas
                theoretical_index = OptimalPaths.check_path_match(path)
                if theoretical_index is not None:
                    theoretical_found[theoretical_index] += 1
                    agent.theoretical_paths_found[theoretical_index] += 1
                    print(f"{LogConfig.EMOJIS['THEORETICAL']} Ruta teórica {theoretical_index+1} encontrada en episodio {episode+1}!")
        
        # Progreso cada 250 episodios
        if (episode + 1) % 250 == 0:
            current_success = successful_episodes / (episode + 1)
            current_optimal = optimal_episodes / (episode + 1)
            unique_count = len(unique_optimal_paths)
            theoretical_count = sum(1 for count in theoretical_found.values() if count > 0)
            
            print(f"  Progreso: {episode + 1}/{num_episodes} - "
                  f"Éxito: {current_success:.1%} - "
                  f"Óptimas: {current_optimal:.1%} - "
                  f"Únicas: {unique_count} - "
                  f"Teóricas: {theoretical_count}/3")
    
    # Calcular métricas finales
    success_rate = successful_episodes / num_episodes
    optimal_rate = optimal_episodes / num_episodes
    optimal_rate_among_successful = optimal_episodes / max(successful_episodes, 1)
    avg_steps = np.mean(total_steps) if total_steps else 0
    theoretical_paths_found = sum(1 for count in theoretical_found.values() if count > 0)
    
    # Mostrar resultados detallados
    print(f"\n{LogConfig.EMOJIS['SUCCESS']} === RESULTADOS DE EVALUACIÓN ===")
    print(f"Episodios evaluados: {num_episodes}")
    print(f"Tasa de éxito: {success_rate:.1%} ({successful_episodes}/{num_episodes})")
    print(f"Tasa de rutas óptimas: {optimal_rate:.1%} ({optimal_episodes}/{num_episodes})")
    print(f"Rutas óptimas entre éxitos: {optimal_rate_among_successful:.1%}")
    
    if total_steps:
        print(f"Pasos promedio a la meta: {avg_steps:.1f}")
        print(f"Rango de pasos: {min(total_steps)} - {max(total_steps)}")
    
    print(f"Rutas óptimas únicas: {len(unique_optimal_paths)}")
    print(f"Rutas teóricas encontradas: {theoretical_paths_found}/3")
    
    # Mostrar rutas únicas encontradas
    if unique_optimal_paths:
        print(f"\n{LogConfig.EMOJIS['OPTIMAL']} RUTAS ÓPTIMAS ÚNICAS DESCUBIERTAS:")
        for i, path_str in enumerate(sorted(unique_optimal_paths), 1):
            # Contar frecuencia de esta ruta
            path_list = [int(x) for x in path_str.split(" → ")]
            count = sum(1 for path in all_paths if path == path_list and len(path) <= EvalConfig.OPTIMAL_STEPS + 1)
            frequency = count / optimal_episodes if optimal_episodes > 0 else 0
            
            print(f"  {i}. {path_str}")
            print(f"     Encontrada {count} veces ({frequency:.1%} de rutas óptimas)")
    
    # Análisis detallado de rutas teóricas
    print(f"\n{LogConfig.EMOJIS['INFO']} VERIFICACIÓN DE RUTAS TEÓRICAS:")
    for i, count in theoretical_found.items():
        path_str = " → ".join(map(str, OptimalPaths.PATHS_4X4[i]))
        if count > 0:
            print(f"  ✅ Ruta teórica {i+1}: {path_str} (encontrada {count} veces)")
        else:
            print(f"  ❌ Ruta teórica {i+1}: {path_str} (no encontrada)")
    
    # Distribución de pasos
    if total_steps:
        steps_dist = {}
        for steps in total_steps:
            steps_dist[steps] = steps_dist.get(steps, 0) + 1
        
        print(f"\n📊 DISTRIBUCIÓN DE EFICIENCIA:")
        sorted_steps = sorted(steps_dist.keys())
        for steps in sorted_steps[:20]:  # Mostrar primeros 20
            count = steps_dist[steps]
            pct = count / successful_episodes * 100
            if steps <= EvalConfig.OPTIMAL_STEPS:
                icon = "🎯"
            elif steps <= 10:
                icon = "✅"
            else:
                icon = "⚠️"
            print(f"  {icon} {steps:2d} pasos: {count:3d} veces ({pct:5.1f}%)")
        
        if len(sorted_steps) > 20:
            remaining = sum(count for steps, count in steps_dist.items() if steps > sorted_steps[19])
            print(f"  ⚠️  >20 pasos: {remaining:3d} veces")
    
    # Comparación con entrenamiento
    print(f"\n📈 COMPARACIÓN ENTRENAMIENTO vs EVALUACIÓN:")
    print(f"  Entrenamiento - Éxito: {initial_stats['success_rate']:.1%}, Óptimas: {initial_stats['optimal_solutions']}")
    print(f"  Evaluación   - Éxito: {success_rate:.1%}, Óptimas: {optimal_episodes}")
    
    # Restaurar estadísticas del agente
    agent.total_episodes = initial_stats['total_episodes']
    agent.successful_episodes = initial_stats['successful_episodes'] 
    agent.optimal_solutions = initial_stats['optimal_solutions']
    
    return {
        'success_rate': success_rate,
        'optimal_rate': optimal_rate,
        'optimal_rate_among_successful': optimal_rate_among_successful,
        'avg_steps': avg_steps,
        'successful_episodes': successful_episodes,
        'optimal_episodes': optimal_episodes,
        'unique_optimal_paths': len(unique_optimal_paths),
        'theoretical_paths_found': theoretical_paths_found,
        'total_steps': total_steps,
        'theoretical_details': theoretical_found
    }

def demonstrate_agent(agent, env, num_demos=EvalConfig.DEMO_EPISODES):
    """
    Demostrar el agente entrenado con análisis detallado
    """
    print(f"\n{LogConfig.EMOJIS['INFO']} === DEMOSTRACIÓN DEL AGENTE ===")
    print("Mapa de Frozen Lake 4x4:")
    print("┌─────┬─────┬─────┬─────┐")
    print("│  S  │  F  │  F  │  F  │  Fila 0")
    print("├─────┼─────┼─────┼─────┤")
    print("│  F  │  H  │  F  │  H  │  Fila 1")
    print("├─────┼─────┼─────┼─────┤")
    print("│  F  │  F  │  F  │  H  │  Fila 2")
    print("├─────┼─────┼─────┼─────┤")
    print("│  H  │  F  │  F  │  G  │  Fila 3")
    print("└─────┴─────┴─────┴─────┘")
    print("S=Start(0), F=Frozen, H=Hole, G=Goal(15)")
    
    action_names = ["←", "↓", "→", "↑"]
    
    # Recolectar múltiples ejecuciones para encontrar las mejores
    all_demos = []
    
    print(f"\nBuscando las mejores demostraciones entre {num_demos * 10} intentos...")
    
    for attempt in range(num_demos * 10):
        state, _ = env.reset()
        path = [state]
        actions_taken = []
        total_reward = 0
        steps = 0
        
        for step in range(EnvConfig.MAX_STEPS):
            action = agent.act(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            actions_taken.append(action)
            path.append(next_state)
            total_reward += reward
            steps += 1
            state = next_state
            
            if terminated or truncated:
                break
        
        if total_reward > 0:  # Solo demos exitosas
            demo_info = {
                'steps': steps,
                'path': path,
                'actions': actions_taken,
                'reward': total_reward,
                'is_optimal': steps <= EvalConfig.OPTIMAL_STEPS,
                'theoretical_index': OptimalPaths.check_path_match(path)
            }
            all_demos.append(demo_info)
    
    if not all_demos:
        print("❌ No se encontraron demostraciones exitosas")
        return
    
    # Ordenar demos: primero teóricas, luego óptimas, luego por pasos
    all_demos.sort(key=lambda x: (
        x['theoretical_index'] is None,  # Teóricas primero
        not x['is_optimal'],             # Luego óptimas
        x['steps']                       # Luego por menos pasos
    ))
    
    # Seleccionar las mejores demos para mostrar
    demos_to_show = all_demos[:num_demos]
    
    optimal_count = sum(1 for demo in demos_to_show if demo['is_optimal'])
    theoretical_count = sum(1 for demo in demos_to_show if demo['theoretical_index'] is not None)
    
    print(f"Encontradas {len(all_demos)} demostraciones exitosas")
    print(f"Mostrando las {len(demos_to_show)} mejores:")
    print(f"  Óptimas: {optimal_count}/{len(demos_to_show)}")
    print(f"  Teóricas: {theoretical_count}/{len(demos_to_show)}")
    
    for i, demo in enumerate(demos_to_show):
        print(f"\n--- Demo {i+1} ---")
        
        # Determinar tipo de demo
        if demo['theoretical_index'] is not None:
            status = f"🏆 TEÓRICA {demo['theoretical_index'] + 1}"
        elif demo['is_optimal']:
            status = "🎯 ÓPTIMA"
        else:
            status = "✅ EXITOSA"
        
        print(f"Resultado: {status} ({demo['steps']} pasos)")
        
        # Mostrar camino con coordenadas
        path_with_coords = []
        for state in demo['path']:
            row, col = divmod(state, 4)
            path_with_coords.append(f"{state}({row},{col})")
        
        print(f"Camino: {' → '.join(path_with_coords)}")
        
        # Mostrar secuencia de acciones
        action_sequence = " → ".join([action_names[a] for a in demo['actions']])
        print(f"Acciones: {action_sequence}")
        
        # Análisis detallado para demos óptimas o teóricas
        if demo['is_optimal'] or demo['theoretical_index'] is not None:
            print("Pasos detallados:")
            for step_idx, (action, next_state) in enumerate(zip(demo['actions'], demo['path'][1:])):
                current_state = demo['path'][step_idx]
                current_pos = f"({current_state//4},{current_state%4})"
                next_pos = f"({next_state//4},{next_state%4})"
                step_num = step_idx + 1
                print(f"  {step_num}. Estado {current_state}{current_pos} --{action_names[action]}--> Estado {next_state}{next_pos}")
        
        # Verificar eficiencia
        optimal_distance = 6  # Distancia mínima conocida
        efficiency = optimal_distance / demo['steps'] if demo['steps'] > 0 else 0
        print(f"Eficiencia: {efficiency:.1%} (óptimo: 6 pasos)")
        
        if i < len(demos_to_show) - 1:
            input("Presiona Enter para la siguiente demostración...")
    
    # Resumen final
    print(f"\n{LogConfig.EMOJIS['SUCCESS']} RESUMEN DE DEMOSTRACIONES:")
    print(f"Demos exitosas mostradas: {len(demos_to_show)}")
    print(f"Demos óptimas: {optimal_count} ({optimal_count/len(demos_to_show):.1%})")
    print(f"Demos teóricas: {theoretical_count} ({theoretical_count/len(demos_to_show):.1%})")
    
    if demos_to_show:
        avg_steps = np.mean([demo['steps'] for demo in demos_to_show])
        min_steps = min(demo['steps'] for demo in demos_to_show)
        print(f"Pasos promedio: {avg_steps:.1f}")
        print(f"Mínimo pasos: {min_steps}")
        
        # Mostrar qué rutas teóricas se encontraron
        found_theoretical = set(demo['theoretical_index'] for demo in demos_to_show 
                              if demo['theoretical_index'] is not None)
        if found_theoretical:
            found_list = [f"Ruta {i+1}" for i in sorted(found_theoretical)]
            print(f"Rutas teóricas mostradas: {', '.join(found_list)}")

def plot_training_results(scores, moving_averages, optimal_counts, steps_per_episode):
    """
    Visualizar resultados del entrenamiento con análisis avanzado
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    episodes = range(1, len(scores) + 1)
    
    # Gráfica 1: Recompensas y convergencia
    ax1.plot(episodes, scores, alpha=0.6, color='lightblue', linewidth=0.8, label='Recompensas')
    ax1.plot(episodes, moving_averages, color='red', linewidth=2, label=f'Promedio móvil ({TrainingConfig.SOLVE_WINDOW})')
    ax1.axhline(y=TrainingConfig.SOLVE_SCORE, color='green', linestyle='--', label='Objetivo')
    ax1.set_xlabel('Episodio')
    ax1.set_ylabel('Recompensa')
    ax1.set_title('Progreso del Entrenamiento')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfica 2: Descubrimiento de rutas óptimas
    ax2.plot(episodes, optimal_counts, color='orange', linewidth=2, marker='o', markersize=1)
    ax2.set_xlabel('Episodio')
    ax2.set_ylabel('Rutas Óptimas Acumuladas')
    ax2.set_title('Descubrimiento de Rutas Óptimas')
    ax2.grid(True, alpha=0.3)
    
    # Añadir líneas verticales para marcar descubrimientos importantes
    if len(optimal_counts) > 0:
        significant_discoveries = []
        for i, count in enumerate(optimal_counts):
            if i > 0 and count > optimal_counts[i-1]:
                significant_discoveries.append(i+1)
        
        for ep in significant_discoveries[:10]:  # Máximo 10 líneas
            ax2.axvline(x=ep, color='red', alpha=0.3, linestyle=':')
    
    # Gráfica 3: Distribución de pasos (solo episodios exitosos)
    successful_steps = [steps for i, steps in enumerate(steps_per_episode) if scores[i] > 0]
    if successful_steps:
        bins = min(30, max(successful_steps) - min(successful_steps) + 1)
        ax3.hist(successful_steps, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(x=EvalConfig.OPTIMAL_STEPS, color='red', linestyle='--', linewidth=2, 
                   label=f'Óptimo: {EvalConfig.OPTIMAL_STEPS} pasos')
        ax3.axvline(x=np.mean(successful_steps), color='orange', linestyle='--', linewidth=2,
                   label=f'Promedio: {np.mean(successful_steps):.1f}')
        ax3.set_xlabel('Pasos a la Meta')
        ax3.set_ylabel('Frecuencia')
        ax3.set_title('Distribución de Pasos (Episodios Exitosos)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Gráfica 4: Evolución de la eficiencia
    window_size = 200
    efficiency_over_time = []
    
    for i in range(window_size, len(scores) + 1):
        window_scores = scores[i-window_size:i]
        window_steps = steps_per_episode[i-window_size:i]
        
        # Calcular porcentaje de episodios exitosos eficientes (≤10 pasos)
        efficient_episodes = sum(1 for j, score in enumerate(window_scores) 
                               if score > 0 and window_steps[j] <= 10)
        total_successful = sum(1 for score in window_scores if score > 0)
        
        if total_successful > 0:
            efficiency = efficient_episodes / total_successful
        else:
            efficiency = 0
        
        efficiency_over_time.append(efficiency)
    
    if efficiency_over_time:
        ax4.plot(range(window_size, len(scores) + 1), efficiency_over_time, 
                color='green', linewidth=2, label='Eficiencia')
        ax4.axhline(y=0.5, color='red', linestyle='--', label='50% eficiencia')
        ax4.axhline(y=0.3, color='orange', linestyle='--', label='30% eficiencia')
        ax4.set_xlabel('Episodio')
        ax4.set_ylabel('Tasa de Eficiencia (≤10 pasos)')
        ax4.set_title(f'Evolución de Eficiencia (Ventana {window_size})')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(TrainingConfig.RESULTS_PATH, dpi=300, bbox_inches='tight')
    print(f"{LogConfig.EMOJIS['SAVE']} Gráfica guardada en {TrainingConfig.RESULTS_PATH}")
    
    try:
        plt.show()
    except:
        print("(Nota: Gráfica guardada pero no se puede mostrar en este entorno)")

def analyze_agent_behavior(agent):
    """
    Analizar el comportamiento aprendido del agente
    """
    print(f"\n{LogConfig.EMOJIS['INFO']} === ANÁLISIS DE COMPORTAMIENTO ===")
    
    # Analizar valores Q para estados clave
    key_states = [0, 1, 4, 5, 8, 9, 10, 13, 14, 15]
    action_names = ["←", "↓", "→", "↑"]
    
    print("Valores Q para estados clave:")
    for state in key_states:
        if state < 16:  # Verificar que el estado sea válido
            q_values, probabilities = agent.get_action_probabilities(state)
            best_action = np.argmax(q_values)
            row, col = divmod(state, 4)
            
            print(f"  Estado {state} ({row},{col}): Mejor acción = {action_names[best_action]}")
            print(f"    Q-values: {q_values}")
            print(f"    Prob: {probabilities}")
    
    # Mostrar estadísticas del agente
    agent.analyze_performance()

def main():
    """Función principal del sistema de entrenamiento DQN"""
    print(f"{LogConfig.COLORS['BOLD']}🧊 === FROZEN LAKE DQN OPTIMIZADO PARA RUTAS ÓPTIMAS ==={LogConfig.COLORS['END']}")
    print("🎯 Versión especializada en encontrar las 3 rutas teóricas de 6 pasos")
    print("🔧 Características: Recompensas moldeadas + Exploración dirigida + Análisis avanzado")
    
    # Validar configuración
    validate_config()
    
    # Crear entorno y agente
    env = create_environment()
    agent = create_agent()
    
    print(f"\n{LogConfig.EMOJIS['INFO']} Configuración del sistema:")
    print(f"  Entorno: {env.spec.id}")
    print(f"  Estados: {env.observation_space.n}")
    print(f"  Acciones: {env.action_space.n}")
    print(f"  Episodios de entrenamiento: {TrainingConfig.NUM_EPISODES}")
    print(f"  Episodios de evaluación: {EvalConfig.EVAL_EPISODES}")
    
    # Mostrar rutas objetivo
    print(f"\n🎯 Rutas óptimas objetivo:")
    for i, path in enumerate(OptimalPaths.PATHS_4X4, 1):
        path_str = " → ".join(map(str, path))
        print(f"  {i}. {path_str}")
    
    # Menú de opciones
    print(f"\n{LogConfig.EMOJIS['INFO']} Opciones disponibles:")
    print("1. Entrenar desde cero")
    print("2. Cargar modelo y evaluar")
    print("3. Cargar modelo y demostrar")
    print("4. Cargar modelo y analizar comportamiento")
    print("5. Flujo completo (entrenar + evaluar + demostrar)")
    
    choice = input("\nSelecciona una opción (1-5): ").strip()
    
    if choice == "1":
        # Solo entrenar
        print(f"\n{LogConfig.EMOJIS['TRAIN']} Iniciando entrenamiento optimizado...")
        scores, moving_averages, optimal_counts, steps_per_episode = train_agent(agent, env)
        plot_training_results(scores, moving_averages, optimal_counts, steps_per_episode)
        
    elif choice == "2":
        # Cargar y evaluar
        agent.load()
        results = evaluate_agent(agent, env)
        
        # Mostrar comparación con objetivos
        print(f"\n📊 COMPARACIÓN CON OBJETIVOS:")
        print(f"  Tasa de éxito: {results['success_rate']:.1%} vs {EvalConfig.TARGET_SUCCESS_RATE:.1%} objetivo")
        print(f"  Rutas óptimas: {results['optimal_rate']:.1%} vs {EvalConfig.TARGET_OPTIMAL_RATE:.1%} objetivo")
        
    elif choice == "3":
        # Cargar y demostrar
        agent.load()
        demonstrate_agent(agent, env)
        
    elif choice == "4":
        # Cargar y analizar
        agent.load()
        analyze_agent_behavior(agent)
        
    elif choice == "5":
        # Flujo completo
        print(f"\n{LogConfig.EMOJIS['TRAIN']} Ejecutando flujo completo optimizado...")
        
        # 1. Entrenar
        print("=" * 60)
        print("FASE 1: ENTRENAMIENTO")
        print("=" * 60)
        scores, moving_averages, optimal_counts, steps_per_episode = train_agent(agent, env)
        plot_training_results(scores, moving_averages, optimal_counts, steps_per_episode)
        
        # 2. Evaluar
        print("=" * 60)
        print("FASE 2: EVALUACIÓN")
        print("=" * 60)
        results = evaluate_agent(agent, env)
        
        # 3. Demostrar si hay buenos resultados
        if results['optimal_rate'] > 0 or results['success_rate'] > 0.5:
            print("=" * 60)
            print("FASE 3: DEMOSTRACIÓN")
            print("=" * 60)
            demonstrate_agent(agent, env)
        
        # 4. Análisis de comportamiento
        print("=" * 60)
        print("FASE 4: ANÁLISIS")
        print("=" * 60)
        analyze_agent_behavior(agent)
        
        # 5. Resumen final
        print("=" * 60)
        print("RESUMEN FINAL")
        print("=" * 60)
        
        final_stats = agent.get_statistics()
        
        print(f"📈 MÉTRICAS FINALES:")
        print(f"  Episodios entrenados: {final_stats['total_episodes']}")
        print(f"  Tasa de éxito: {results['success_rate']:.1%}")
        print(f"  Rutas óptimas encontradas: {results['optimal_episodes']}")
        print(f"  Tasa de rutas óptimas: {results['optimal_rate']:.1%}")
        print(f"  Rutas teóricas encontradas: {results['theoretical_paths_found']}/3")
        print(f"  Pasos promedio: {results['avg_steps']:.1f}")
        
        # Evaluación del éxito
        success_criteria = {
            'Tasa de éxito ≥80%': results['success_rate'] >= 0.8,
            'Rutas óptimas ≥20%': results['optimal_rate'] >= 0.2,
            'Al menos 1 ruta teórica': results['theoretical_paths_found'] >= 1,
            'Todas las rutas teóricas': results['theoretical_paths_found'] == 3,
            'Pasos promedio ≤15': results['avg_steps'] <= 15
        }
        
        print(f"\n🎯 EVALUACIÓN DE OBJETIVOS:")
        for criterion, achieved in success_criteria.items():
            status = "✅" if achieved else "❌"
            print(f"  {status} {criterion}")
        
        achieved_count = sum(success_criteria.values())
        total_criteria = len(success_criteria)
        
        print(f"\nPuntuación final: {achieved_count}/{total_criteria} criterios cumplidos")
        
        if achieved_count >= 4:
            print(f"{LogConfig.EMOJIS['SUCCESS']} ¡EXCELENTE! Objetivos principales alcanzados")
        elif achieved_count >= 2:
            print(f"{LogConfig.EMOJIS['INFO']} BUENO. Progreso significativo logrado")
        else:
            print(f"{LogConfig.EMOJIS['WARNING']} Necesita más optimización")
            
    else:
        print(f"{LogConfig.EMOJIS['ERROR']} Opción no válida.")
    
    env.close()
    print(f"\n{LogConfig.EMOJIS['SUCCESS']} ¡Programa terminado!")

if __name__ == "__main__":
    main()