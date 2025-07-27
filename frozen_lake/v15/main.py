"""
main.py - Q-Learning con Tabla Q para Frozen Lake

Este archivo implementa el entrenamiento y evaluación usando Q-Learning clásico
con tabla Q en lugar de redes neuronales. Esto debería resolver los problemas
de generalización y conseguir mejores resultados en rutas óptimas.
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
import argparse
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

def train_q_learning_agent(agent, env, num_episodes=TrainingConfig.NUM_EPISODES):
    """
    Entrenar agente Q-Learning con tabla Q
    
    Args:
        agent: Agente Q-Learning
        env: Entorno de Gymnasium
        num_episodes (int): Número de episodios de entrenamiento
        
    Returns:
        tuple: (scores, moving_averages, optimal_counts, convergence_data)
    """
    print(f"{LogConfig.EMOJIS['TRAIN']} Entrenamiento Q-Learning con Tabla Q")
    print(f"🎯 Objetivo: Convergencia completa a rutas óptimas")
    print_config()
    
    # Métricas de entrenamiento
    scores = []
    moving_averages = []
    optimal_counts = []
    steps_per_episode = []
    convergence_data = []
    scores_window = deque(maxlen=TrainingConfig.SOLVE_WINDOW)
    
    start_time = time.time()
    
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        score = 0
        steps = 0
        episode_path = [state]
        
        for step in range(EnvConfig.MAX_STEPS):
            # Seleccionar acción
            action = agent.select_action(state, training=True)
            
            # Ejecutar acción
            next_state, env_reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            episode_path.append(next_state)
            
            # Entrenar agente Q-Learning
            full_path = episode_path if done else None
            agent.train_step(state, action, env_reward, next_state, done, steps, full_path)
            
            # Actualizar estado y score
            state = next_state
            score += env_reward
            
            if done:
                break
        
        # Finalizar episodio
        agent.end_episode(score, steps)
        
        # Guardar métricas
        scores.append(score)
        scores_window.append(score)
        steps_per_episode.append(steps)
        optimal_counts.append(agent.optimal_solutions)
        
        # Análisis de convergencia
        is_converged = agent.analyze_convergence()
        convergence_data.append(is_converged)
        
        # Calcular promedio móvil
        if len(scores_window) >= TrainingConfig.SOLVE_WINDOW:
            moving_avg = np.mean(scores_window)
            moving_averages.append(moving_avg)
            
            # Verificar si el problema está resuelto
            if moving_avg >= TrainingConfig.SOLVE_SCORE:
                print(f"\n{LogConfig.EMOJIS['SUCCESS']} ¡Entorno resuelto en {episode} episodios!")
                print(f"Promedio de últimos {TrainingConfig.SOLVE_WINDOW} episodios: {moving_avg:.3f}")
                
                # Verificar también convergencia y rutas óptimas
                stats = agent.get_statistics()
                if (is_converged and 
                    stats['theoretical_paths_found'] >= 2 and 
                    stats['optimal_rate'] >= 0.1):
                    print(f"{LogConfig.EMOJIS['OPTIMAL']} ¡Convergencia óptima completa!")
                    break
        else:
            moving_averages.append(np.mean(scores))
        
        # Imprimir progreso
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
            print(f"  Learning rate: {agent.learning_rate:.4f}")
            print(f"  Convergencia: {'Sí' if is_converged else 'No'}")
            print(f"  Tiempo: {elapsed_time:.1f}s")
            
            # Mostrar rutas teóricas encontradas
            if any(count > 0 for count in agent.theoretical_paths_found.values()):
                found_paths = []
                for i, count in agent.theoretical_paths_found.items():
                    if count > 0:
                        found_paths.append(f"Ruta {i+1}({count}x)")
                print(f"  Encontradas: {', '.join(found_paths)}")
            
            # Parada temprana si convergencia total
            if (episode > 2000 and is_converged and 
                stats['theoretical_paths_found'] == 3 and
                all(count >= 5 for count in agent.theoretical_paths_found.values())):
                print(f"\n{LogConfig.EMOJIS['SUCCESS']} ¡CONVERGENCIA TOTAL ALCANZADA!")
                break
        
        # Guardar modelo periódicamente
        if episode % TrainingConfig.SAVE_EVERY == 0:
            agent.save(f"qtable_checkpoint_episode_{episode}.npy")
    
    # Guardar modelo final
    agent.save()
    
    # Estadísticas finales
    total_time = time.time() - start_time
    final_stats = agent.get_statistics()
    
    print(f"\n{LogConfig.EMOJIS['SUCCESS']} ¡Entrenamiento Q-Learning completado!")
    print(f"Tiempo total: {total_time:.1f}s ({total_time/60:.1f} minutos)")
    print(f"Episodios exitosos: {agent.successful_episodes}/{episode} ({final_stats['success_rate']:.1%})")
    print(f"Rutas óptimas encontradas: {agent.optimal_solutions}")
    print(f"Tasa de rutas óptimas: {final_stats['optimal_rate']:.1%} de éxitos")
    print(f"Rutas teóricas encontradas: {final_stats['theoretical_paths_found']}/3")
    print(f"Pasos promedio: {final_stats['avg_steps']:.1f}")
    print(f"Convergencia final: {'Sí' if agent.analyze_convergence() else 'No'}")
    
    # Detalles de rutas teóricas
    print(f"\nDetalle de rutas teóricas encontradas:")
    for i, count in agent.theoretical_paths_found.items():
        path_str = " → ".join(map(str, OptimalPaths.PATHS_4X4[i]))
        status = "✅" if count > 0 else "❌"
        print(f"  {status} Ruta {i+1}: {path_str} ({count} veces)")
    
    return scores, moving_averages, optimal_counts, convergence_data

def evaluate_q_learning_agent(agent, env, num_episodes=EvalConfig.EVAL_EPISODES):
    """
    Evaluar agente Q-Learning entrenado
    
    Args:
        agent: Agente Q-Learning entrenado
        env: Entorno de evaluación
        num_episodes (int): Número de episodios de evaluación
        
    Returns:
        dict: Métricas de evaluación detalladas
    """
    print(f"\n{LogConfig.EMOJIS['EVAL']} Evaluando agente Q-Learning - {num_episodes} episodios...")
    
    # Resetear estadísticas para evaluación limpia
    initial_stats = agent.get_statistics()
    agent.reset_statistics()
    
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
            action = agent.select_action(state, training=False)  # Sin exploración
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
                theoretical_index = OptimalPaths.is_optimal_path(path)
                if theoretical_index is not None:
                    theoretical_found[theoretical_index] += 1
                    agent.theoretical_paths_found[theoretical_index] += 1
                    print(f"{LogConfig.EMOJIS['THEORETICAL']} Ruta teórica {theoretical_index+1} encontrada en episodio {episode+1}!")
        
        # Progreso cada 200 episodios
        if (episode + 1) % 200 == 0:
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
    print(f"\n{LogConfig.EMOJIS['SUCCESS']} === RESULTADOS DE EVALUACIÓN Q-LEARNING ===")
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
        for steps in sorted_steps[:15]:  # Mostrar primeros 15
            count = steps_dist[steps]
            pct = count / successful_episodes * 100
            if steps <= EvalConfig.OPTIMAL_STEPS:
                icon = "🎯"
            elif steps <= 10:
                icon = "✅"
            else:
                icon = "⚠️"
            print(f"  {icon} {steps:2d} pasos: {count:3d} veces ({pct:5.1f}%)")
        
        if len(sorted_steps) > 15:
            remaining = sum(count for steps, count in steps_dist.items() if steps > sorted_steps[14])
            print(f"  ⚠️  >15 pasos: {remaining:3d} veces")
    
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

def demonstrate_q_learning_agent(agent, env, num_demos=EvalConfig.DEMO_EPISODES):
    """
    Demostrar agente Q-Learning con análisis de política
    """
    print(f"\n{LogConfig.EMOJIS['INFO']} === DEMOSTRACIÓN DEL AGENTE Q-LEARNING ===")
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
    
    # Mostrar política aprendida
    print(f"\n{LogConfig.EMOJIS['QTABLE']} POLÍTICA APRENDIDA:")
    policy = agent.get_policy()
    print("Estado | Mejor Acción | Valor | Q-Values")
    print("-" * 45)
    for state in range(16):
        row, col = divmod(state, 4)
        pol = policy[state]
        q_str = " ".join([f"{q:5.1f}" for q in pol['q_values']])
        print(f"{state:2d}({row},{col}) | {pol['action_name']:11s} | {pol['value']:5.1f} | {q_str}")
    
    # Recolectar demostraciones
    all_demos = []
    
    print(f"\nBuscando demostraciones entre {num_demos * 10} intentos...")
    
    for attempt in range(num_demos * 10):
        state, _ = env.reset()
        path = [state]
        actions_taken = []
        total_reward = 0
        steps = 0
        
        for step in range(EnvConfig.MAX_STEPS):
            action = agent.select_action(state, training=False)
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
                'theoretical_index': OptimalPaths.is_optimal_path(path)
            }
            all_demos.append(demo_info)
    
    if not all_demos:
        print("❌ No se encontraron demostraciones exitosas")
        return
    
    # Ordenar demos por calidad
    all_demos.sort(key=lambda x: (
        x['theoretical_index'] is None,
        not x['is_optimal'],
        x['steps']
    ))
    
    demos_to_show = all_demos[:num_demos]
    optimal_count = sum(1 for demo in demos_to_show if demo['is_optimal'])
    theoretical_count = sum(1 for demo in demos_to_show if demo['theoretical_index'] is not None)
    
    print(f"Encontradas {len(all_demos)} demostraciones exitosas")
    print(f"Mostrando las {len(demos_to_show)} mejores:")
    print(f"  Óptimas: {optimal_count}/{len(demos_to_show)}")
    print(f"  Teóricas: {theoretical_count}/{len(demos_to_show)}")
    
    for i, demo in enumerate(demos_to_show):
        print(f"\n--- Demo {i+1} ---")
        
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
        print(f"Acciones: {' → '.join([action_names[a] for a in demo['actions']])}")
        
        # Eficiencia
        efficiency = EvalConfig.OPTIMAL_STEPS / demo['steps'] if demo['steps'] > 0 else 0
        print(f"Eficiencia: {efficiency:.1%} (óptimo: {EvalConfig.OPTIMAL_STEPS} pasos)")
        
        if i < len(demos_to_show) - 1:
            input("Presiona Enter para la siguiente demostración...")
    
    print(f"\n{LogConfig.EMOJIS['SUCCESS']} RESUMEN DE DEMOSTRACIONES:")
    print(f"Demos exitosas mostradas: {len(demos_to_show)}")
    print(f"Demos óptimas: {optimal_count} ({optimal_count/len(demos_to_show):.1%})")
    print(f"Demos teóricas: {theoretical_count} ({theoretical_count/len(demos_to_show):.1%})")
    
    if demos_to_show:
        avg_steps = np.mean([demo['steps'] for demo in demos_to_show])
        min_steps = min(demo['steps'] for demo in demos_to_show)
        print(f"Pasos promedio: {avg_steps:.1f}")
        print(f"Mínimo pasos: {min_steps}")

def plot_q_learning_results(scores, moving_averages, optimal_counts, convergence_data):
    """
    Visualizar resultados del entrenamiento Q-Learning
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    episodes = range(1, len(scores) + 1)
    
    # Gráfica 1: Recompensas y convergencia
    ax1.plot(episodes, scores, alpha=0.6, color='lightblue', linewidth=0.8, label='Recompensas')
    ax1.plot(episodes, moving_averages, color='red', linewidth=2, label=f'Promedio móvil ({TrainingConfig.SOLVE_WINDOW})')
    ax1.axhline(y=TrainingConfig.SOLVE_SCORE, color='green', linestyle='--', label='Objetivo')
    ax1.set_xlabel('Episodio')
    ax1.set_ylabel('Recompensa')
    ax1.set_title('Progreso del Entrenamiento Q-Learning')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfica 2: Descubrimiento de rutas óptimas
    ax2.plot(episodes, optimal_counts, color='orange', linewidth=2, marker='o', markersize=1)
    ax2.set_xlabel('Episodio')
    ax2.set_ylabel('Rutas Óptimas Acumuladas')
    ax2.set_title('Descubrimiento de Rutas Óptimas')
    ax2.grid(True, alpha=0.3)
    
    # Gráfica 3: Convergencia del algoritmo
    if len(convergence_data) > 0:
        convergence_episodes = [i+1 for i, converged in enumerate(convergence_data) if converged]
        convergence_y = [1] * len(convergence_episodes)
        
        ax3.scatter(convergence_episodes, convergence_y, alpha=0.6, color='green', s=10)
        ax3.set_xlabel('Episodio')
        ax3.set_ylabel('Convergencia')
        ax3.set_title('Convergencia del Algoritmo Q-Learning')
        ax3.set_ylim(0, 2)
        ax3.grid(True, alpha=0.3)
    
    # Gráfica 4: Comparación de métricas
    if len(episodes) > 500:
        window = 200
        success_rates = []
        optimal_rates = []
        
        for i in range(window, len(scores) + 1):
            window_scores = scores[i-window:i]
            window_optimal = optimal_counts[i-1] - optimal_counts[max(0, i-window-1)]
            
            success_rate = sum(1 for s in window_scores if s > 0) / window
            success_rates.append(success_rate)
            
            total_successes = sum(1 for s in window_scores if s > 0)
            optimal_rate = window_optimal / max(total_successes, 1)
            optimal_rates.append(optimal_rate)
        
        episodes_windowed = range(window, len(scores) + 1)
        ax4.plot(episodes_windowed, success_rates, color='blue', label='Tasa de Éxito')
        ax4.plot(episodes_windowed, optimal_rates, color='orange', label='Tasa de Rutas Óptimas')
        ax4.set_xlabel('Episodio')
        ax4.set_ylabel('Tasa')
        ax4.set_title('Evolución de Métricas de Rendimiento')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(TrainingConfig.RESULTS_PATH, dpi=300, bbox_inches='tight')
    print(f"{LogConfig.EMOJIS['SAVE']} Gráficas guardadas en {TrainingConfig.RESULTS_PATH}")
    plt.show()

def save_detailed_results(scores, moving_averages, optimal_counts, agent_stats, eval_results=None):
    """
    Guardar resultados detallados en archivos
    """
    results = {
        'training': {
            'scores': scores,
            'moving_averages': moving_averages,
            'optimal_counts': optimal_counts,
            'final_stats': agent_stats
        }
    }
    
    if eval_results:
        results['evaluation'] = eval_results
    
    # Guardar en formato numpy
    results_file = "qtable_detailed_results.npy"
    np.save(results_file, results)
    print(f"{LogConfig.EMOJIS['SAVE']} Resultados detallados guardados en {results_file}")
    
    # Guardar resumen en texto
    summary_file = "qtable_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=== RESUMEN DE ENTRENAMIENTO Q-LEARNING ===\n\n")
        f.write(f"Episodios totales: {len(scores)}\n")
        f.write(f"Tasa de éxito final: {agent_stats['success_rate']:.1%}\n")
        f.write(f"Rutas óptimas encontradas: {agent_stats['optimal_solutions']}\n")
        f.write(f"Tasa de rutas óptimas: {agent_stats['optimal_rate']:.1%}\n")
        f.write(f"Rutas teóricas descubiertas: {agent_stats['theoretical_paths_found']}/3\n")
        f.write(f"Pasos promedio: {agent_stats['avg_steps']:.1f}\n")
        f.write(f"Epsilon final: {agent_stats['epsilon']:.4f}\n")
        f.write(f"Learning rate final: {agent_stats['learning_rate']:.4f}\n")
        
        if eval_results:
            f.write(f"\n=== RESULTADOS DE EVALUACIÓN ===\n")
            f.write(f"Episodios evaluados: {EvalConfig.EVAL_EPISODES}\n")
            f.write(f"Tasa de éxito: {eval_results['success_rate']:.1%}\n")
            f.write(f"Tasa de rutas óptimas: {eval_results['optimal_rate']:.1%}\n")
            f.write(f"Rutas óptimas únicas: {eval_results['unique_optimal_paths']}\n")
            f.write(f"Rutas teóricas encontradas: {eval_results['theoretical_paths_found']}/3\n")
    
    print(f"{LogConfig.EMOJIS['SAVE']} Resumen guardado en {summary_file}")

def main():
    """Función principal con argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description='Q-Learning para Frozen Lake')
    parser.add_argument('--mode', choices=['train', 'eval', 'demo', 'all'], 
                       default='all', help='Modo de ejecución')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Número de episodios (override config)')
    parser.add_argument('--load', type=str, default=None,
                       help='Cargar modelo desde archivo')
    parser.add_argument('--save', type=str, default=None,
                       help='Guardar modelo en archivo específico')
    parser.add_argument('--no-plots', action='store_true',
                       help='No mostrar gráficas')
    parser.add_argument('--verbose', action='store_true',
                       help='Mostrar información detallada')
    
    args = parser.parse_args()
    
    # Validar configuración
    validate_config()
    
    # Crear entorno y agente
    print(f"{LogConfig.EMOJIS['INFO']} Inicializando Q-Learning Frozen Lake...")
    env = create_environment()
    agent = create_agent()
    
    # Cargar modelo si se especifica
    if args.load:
        agent.load(args.load)
    elif os.path.exists(TrainingConfig.MODEL_PATH) and args.mode in ['eval', 'demo']:
        print(f"{LogConfig.EMOJIS['INFO']} Cargando modelo existente para {args.mode}...")
        agent.load()
    
    # Ejecutar según el modo
    if args.mode == 'train' or args.mode == 'all':
        print(f"\n{LogConfig.EMOJIS['TRAIN']} === FASE DE ENTRENAMIENTO ===")
        
        # Determinar número de episodios
        num_episodes = args.episodes if args.episodes else TrainingConfig.NUM_EPISODES
        
        # Entrenar agente
        scores, moving_averages, optimal_counts, convergence_data = train_q_learning_agent(
            agent, env, num_episodes
        )
        
        # Guardar modelo
        if args.save:
            agent.save(args.save)
        else:
            agent.save()
        
        # Mostrar tabla Q final si es verbose
        if args.verbose:
            agent.print_q_table()
        
        # Crear gráficas si no está deshabilitado
        if not args.no_plots:
            plot_q_learning_results(scores, moving_averages, optimal_counts, convergence_data)
        
        # Guardar resultados detallados
        final_stats = agent.get_statistics()
        save_detailed_results(scores, moving_averages, optimal_counts, final_stats)
        
        print(f"\n{LogConfig.EMOJIS['SUCCESS']} Entrenamiento completado exitosamente!")
    
    if args.mode == 'eval' or args.mode == 'all':
        print(f"\n{LogConfig.EMOJIS['EVAL']} === FASE DE EVALUACIÓN ===")
        
        # Verificar que el agente esté entrenado
        if agent.total_episodes == 0:
            print(f"{LogConfig.EMOJIS['WARNING']} El agente no está entrenado. Ejecuta primero el entrenamiento.")
            return
        
        # Evaluar agente
        eval_results = evaluate_q_learning_agent(agent, env)
        
        # Guardar resultados de evaluación si hay datos de entrenamiento
        if args.mode == 'all':
            save_detailed_results([], [], [], agent.get_statistics(), eval_results)
        
        print(f"\n{LogConfig.EMOJIS['SUCCESS']} Evaluación completada exitosamente!")
    
    if args.mode == 'demo' or args.mode == 'all':
        print(f"\n{LogConfig.EMOJIS['INFO']} === FASE DE DEMOSTRACIÓN ===")
        
        # Verificar que el agente esté entrenado
        if agent.total_episodes == 0:
            print(f"{LogConfig.EMOJIS['WARNING']} El agente no está entrenado. Ejecuta primero el entrenamiento.")
            return
        
        # Demostrar agente
        demonstrate_q_learning_agent(agent, env)
        
        print(f"\n{LogConfig.EMOJIS['SUCCESS']} Demostración completada exitosamente!")
    
    # Cerrar entorno
    env.close()
    
    print(f"\n{LogConfig.EMOJIS['SUCCESS']} ¡Ejecución completa de Q-Learning Frozen Lake!")
    print("Archivos generados:")
    if os.path.exists(TrainingConfig.MODEL_PATH):
        print(f"  📊 Modelo: {TrainingConfig.MODEL_PATH}")
    if os.path.exists(TrainingConfig.RESULTS_PATH):
        print(f"  📈 Gráficas: {TrainingConfig.RESULTS_PATH}")
    if os.path.exists("qtable_summary.txt"):
        print(f"  📝 Resumen: qtable_summary.txt")

def quick_test():
    """Función para pruebas rápidas del sistema"""
    print(f"{LogConfig.EMOJIS['INFO']} === PRUEBA RÁPIDA DEL SISTEMA ===")
    
    # Validar configuración
    validate_config()
    
    # Crear entorno y agente
    env = create_environment()
    agent = create_agent()
    
    print(f"Entorno creado: {env.spec.id}")
    print(f"Espacio de estados: {env.observation_space}")
    print(f"Espacio de acciones: {env.action_space}")
    
    # Ejecutar algunos episodios de prueba
    print(f"\nEjecutando 10 episodios de prueba...")
    
    for episode in range(10):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(20):
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            agent.train_step(state, action, reward, next_state, 
                           terminated or truncated, step + 1)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        agent.end_episode(total_reward, steps)
        
        status = "✅" if total_reward > 0 else "❌"
        print(f"  Episodio {episode + 1}: {status} {steps} pasos, recompensa: {total_reward}")
    
    # Mostrar estadísticas
    stats = agent.get_statistics()
    print(f"\nEstadísticas de prueba:")
    print(f"  Tasa de éxito: {stats['success_rate']:.1%}")
    print(f"  Rutas óptimas: {stats['optimal_solutions']}")
    print(f"  Pasos promedio: {stats['avg_steps']:.1f}")
    print(f"  Epsilon actual: {stats['epsilon']:.3f}")
    
    env.close()
    print(f"\n{LogConfig.EMOJIS['SUCCESS']} Prueba completada exitosamente!")

if __name__ == "__main__":
    try:
        # Si no hay argumentos, ejecutar prueba rápida
        import sys
        if len(sys.argv) == 1:
            quick_test()
        else:
            main()
    except KeyboardInterrupt:
        print(f"\n{LogConfig.EMOJIS['WARNING']} Ejecución interrumpida por el usuario")
    except Exception as e:
        print(f"\n{LogConfig.EMOJIS['ERROR']} Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()