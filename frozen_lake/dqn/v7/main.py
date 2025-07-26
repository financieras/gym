"""
main_enhanced.py - Entrenamiento con Agente DQN Mejorado

Este archivo utiliza el agente mejorado con recompensas moldeadas
para encontrar rutas óptimas en Frozen Lake.
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque

from config import *
from agent import create_enhanced_agent

def create_environment():
    """Crear entorno Frozen Lake"""
    return gym.make(
        'FrozenLake-v1',
        desc=None,
        map_name=EnvConfig.MAP_SIZE,
        is_slippery=EnvConfig.IS_SLIPPERY,
        render_mode=EnvConfig.RENDER_MODE
    )

def train_enhanced_agent(agent, env, num_episodes=TrainingConfig.NUM_EPISODES):
    """
    Entrenar agente DQN mejorado con recompensas moldeadas
    """
    print(f"{LogConfig.EMOJIS['TRAIN']} Entrenando agente mejorado - {num_episodes} episodios")
    print(f"🎯 Objetivo: Encontrar las 3 rutas óptimas de 6 pasos")
    print_config()
    
    # Métricas
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
        
        for step in range(EnvConfig.MAX_STEPS):
            action = agent.act(state, training=True)
            next_state, env_reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            
            # Entrenar con recompensa moldeada
            agent.step(state, action, env_reward, next_state, done, steps)
            
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
        
        # Promedio móvil
        if len(scores_window) >= TrainingConfig.SOLVE_WINDOW:
            moving_avg = np.mean(scores_window)
            moving_averages.append(moving_avg)
            
            # Verificar si resuelto
            if moving_avg >= TrainingConfig.SOLVE_SCORE:
                print(f"\n{LogConfig.EMOJIS['SUCCESS']} ¡Entorno resuelto en {episode} episodios!")
                break
        else:
            moving_averages.append(np.mean(scores))
        
        # Progreso cada 200 episodios
        if episode % 200 == 0:
            elapsed = time.time() - start_time
            stats = agent.get_statistics()
            
            print(f"\nEpisodio {episode:4d}/{num_episodes}")
            print(f"  Promedio últimos {len(scores_window):3d}: {np.mean(scores_window):.3f}")
            print(f"  Tasa de éxito: {stats['success_rate']:.1%}")
            print(f"  Rutas óptimas: {agent.optimal_solutions} ({stats['optimal_rate']:.1%} de éxitos)")
            print(f"  Pasos promedio: {stats['avg_steps']:.1f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Tiempo: {elapsed:.1f}s")
        
        # Guardar modelo periódicamente
        if episode % TrainingConfig.SAVE_EVERY == 0:
            agent.save(f"enhanced_checkpoint_episode_{episode}.pth")
    
    # Guardar modelo final
    agent.save("enhanced_frozen_lake_dqn.pth")
    
    # Estadísticas finales
    total_time = time.time() - start_time
    final_stats = agent.get_statistics()
    
    print(f"\n{LogConfig.EMOJIS['SUCCESS']} ¡Entrenamiento completado!")
    print(f"Tiempo total: {total_time:.1f}s ({total_time/60:.1f} minutos)")
    print(f"Episodios exitosos: {agent.successful_episodes}/{episode} ({final_stats['success_rate']:.1%})")
    print(f"Rutas óptimas encontradas: {agent.optimal_solutions}")
    print(f"Tasa de rutas óptimas: {final_stats['optimal_rate']:.1%} de éxitos")
    print(f"Pasos promedio: {final_stats['avg_steps']:.1f}")
    
    # Análisis de rutas óptimas por fase
    if agent.optimal_episodes:
        print(f"\nEpisodios con rutas óptimas: {agent.optimal_episodes}")
        early_phase = sum(1 for ep in agent.optimal_episodes if ep <= episode // 3)
        mid_phase = sum(1 for ep in agent.optimal_episodes if episode // 3 < ep <= 2 * episode // 3)
        late_phase = sum(1 for ep in agent.optimal_episodes if ep > 2 * episode // 3)
        
        print(f"Distribución temporal:")
        print(f"  Fase inicial (1-{episode//3}): {early_phase} rutas óptimas")
        print(f"  Fase media ({episode//3+1}-{2*episode//3}): {mid_phase} rutas óptimas")
        print(f"  Fase final ({2*episode//3+1}-{episode}): {late_phase} rutas óptimas")
    
    return scores, moving_averages, optimal_counts, steps_per_episode

def evaluate_enhanced_agent(agent, env, num_episodes=EvalConfig.EVAL_EPISODES):
    """
    Evaluar agente mejorado
    """
    print(f"\n{LogConfig.EMOJIS['EVAL']} Evaluando agente mejorado - {num_episodes} episodios...")
    
    successful_episodes = 0
    optimal_episodes = 0
    total_steps = []
    optimal_paths_found = []
    unique_optimal_paths = set()
    
    # Verificar rutas teóricas específicas
    theoretical_paths_found = {i: 0 for i in range(len(OptimalPaths.PATHS_4X4))}
    
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
        
        # Analizar resultados
        if total_reward > 0:  # Éxito
            successful_episodes += 1
            total_steps.append(steps)
            
            # Verificar si es óptima
            if steps <= EvalConfig.OPTIMAL_STEPS:
                optimal_episodes += 1
                path_str = " → ".join(map(str, path))
                unique_optimal_paths.add(path_str)
                optimal_paths_found.append(path)
                
                # Verificar si coincide con rutas teóricas
                for i, theoretical_path in enumerate(OptimalPaths.PATHS_4X4):
                    if path == theoretical_path:
                        theoretical_paths_found[i] += 1
                        print(f"🎯 Ruta teórica {i+1} encontrada en episodio {episode+1}!")
                        break
        
        # Progreso cada 250 episodios
        if (episode + 1) % 250 == 0:
            current_success = successful_episodes / (episode + 1)
            current_optimal = optimal_episodes / (episode + 1)
            print(f"  Progreso: {episode + 1}/{num_episodes} - "
                  f"Éxito: {current_success:.1%} - "
                  f"Óptimas: {current_optimal:.1%} - "
                  f"Rutas únicas: {len(unique_optimal_paths)}")
    
    # Calcular métricas finales
    success_rate = successful_episodes / num_episodes
    optimal_rate = optimal_episodes / num_episodes
    optimal_rate_among_successful = optimal_episodes / max(successful_episodes, 1)
    avg_steps = np.mean(total_steps) if total_steps else 0
    
    # Mostrar resultados
    print(f"\n{LogConfig.EMOJIS['SUCCESS']} === RESULTADOS DE EVALUACIÓN MEJORADA ===")
    print(f"Episodios evaluados: {num_episodes}")
    print(f"Tasa de éxito: {success_rate:.1%} ({successful_episodes}/{num_episodes})")
    print(f"Tasa de rutas óptimas: {optimal_rate:.1%} ({optimal_episodes}/{num_episodes})")
    print(f"Rutas óptimas entre éxitos: {optimal_rate_among_successful:.1%}")
    
    if total_steps:
        print(f"Pasos promedio a la meta: {avg_steps:.1f}")
        print(f"Rango de pasos: {min(total_steps)} - {max(total_steps)}")
    
    print(f"Rutas óptimas únicas encontradas: {len(unique_optimal_paths)}")
    
    # Mostrar rutas únicas encontradas
    if unique_optimal_paths:
        print(f"\n{LogConfig.EMOJIS['OPTIMAL']} RUTAS ÓPTIMAS ÚNICAS DESCUBIERTAS:")
        for i, path in enumerate(sorted(unique_optimal_paths), 1):
            count = sum(1 for p in optimal_paths_found if " → ".join(map(str, p)) == path)
            frequency = count / optimal_episodes if optimal_episodes > 0 else 0
            print(f"  {i}. {path}")
            print(f"     Encontrada {count} veces ({frequency:.1%} de rutas óptimas)")
    
    # Análisis de rutas teóricas
    theoretical_found = sum(1 for count in theoretical_paths_found.values() if count > 0)
    print(f"\n{LogConfig.EMOJIS['INFO']} VERIFICACIÓN DE RUTAS TEÓRICAS:")
    for i, count in theoretical_paths_found.items():
        path_str = " → ".join(map(str, OptimalPaths.PATHS_4X4[i]))
        if count > 0:
            print(f"  ✅ Ruta teórica {i+1}: {path_str} (encontrada {count} veces)")
        else:
            print(f"  ❌ Ruta teórica {i+1}: {path_str} (no encontrada)")
    
    print(f"\n🎯 COMPLETITUD: {theoretical_found}/3 rutas teóricas encontradas ({theoretical_found/3:.1%})")
    
    # Distribución de pasos
    if total_steps:
        steps_dist = {}
        for steps in total_steps:
            steps_dist[steps] = steps_dist.get(steps, 0) + 1
        
        print(f"\n📊 DISTRIBUCIÓN DE EFICIENCIA:")
        for steps in sorted(steps_dist.keys())[:15]:  # Mostrar solo primeros 15
            count = steps_dist[steps]
            pct = count / successful_episodes * 100
            if steps <= 6:
                icon = "🎯"
            elif steps <= 10:
                icon = "✅"
            else:
                icon = "⚠️"
            print(f"  {icon} {steps:2d} pasos: {count:3d} veces ({pct:5.1f}%)")
        
        if len(steps_dist) > 15:
            remaining = sum(count for steps, count in steps_dist.items() if steps > sorted(steps_dist.keys())[14])
            print(f"  ⚠️  >15 pasos: {remaining:3d} veces")
    
    return {
        'success_rate': success_rate,
        'optimal_rate': optimal_rate,
        'optimal_rate_among_successful': optimal_rate_among_successful,
        'avg_steps': avg_steps,
        'successful_episodes': successful_episodes,
        'optimal_episodes': optimal_episodes,
        'unique_optimal_paths': len(unique_optimal_paths),
        'theoretical_paths_found': theoretical_found,
        'total_steps': total_steps
    }

def demonstrate_enhanced_agent(agent, env, num_demos=EvalConfig.DEMO_EPISODES):
    """
    Demostrar agente mejorado con análisis detallado
    """
    print(f"\n{LogConfig.EMOJIS['INFO']} === DEMOSTRACIÓN DEL AGENTE MEJORADO ===")
    print("Buscando las mejores ejecuciones...")
    
    action_names = ["←", "↓", "→", "↑"]
    demos_found = []
    
    # Recolectar muchas ejecuciones para encontrar las mejores
    for attempt in range(num_demos * 5):  # Intentar más para encontrar buenas demos
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
        
        if total_reward > 0:  # Solo guardar demos exitosas
            demos_found.append({
                'steps': steps,
                'path': path,
                'actions': actions_taken,
                'reward': total_reward,
                'is_optimal': steps <= EvalConfig.OPTIMAL_STEPS
            })
    
    # Ordenar demos: primero óptimas, luego por menor número de pasos
    demos_found.sort(key=lambda x: (not x['is_optimal'], x['steps']))
    
    # Mostrar las mejores demos
    demos_to_show = min(num_demos, len(demos_found))
    optimal_demos = sum(1 for demo in demos_found[:demos_to_show] if demo['is_optimal'])
    
    print(f"Se encontraron {len(demos_found)} ejecuciones exitosas")
    print(f"Mostrando las {demos_to_show} mejores:")
    
    for i, demo in enumerate(demos_found[:demos_to_show]):
        print(f"\n--- Demo {i+1} ---")
        
        status = "🎯 ÓPTIMA" if demo['is_optimal'] else "✅ EXITOSA"
        print(f"Resultado: {status} ({demo['steps']} pasos)")
        
        # Mostrar camino completo
        path_str = " → ".join([f"{state}({state//4},{state%4})" for state in demo['path']])
        print(f"Camino: {path_str}")
        
        # Mostrar secuencia de acciones
        action_sequence = " → ".join([action_names[a] for a in demo['actions']])
        print(f"Acciones: {action_sequence}")
        
        # Verificar si es ruta teórica
        for j, theoretical_path in enumerate(OptimalPaths.PATHS_4X4):
            if demo['path'] == theoretical_path:
                print(f"🏆 ¡Esta es la ruta teórica {j+1}!")
                break
        
        # Mostrar algunos pasos detallados para rutas óptimas
        if demo['is_optimal'] and i < 2:  # Solo para las 2 primeras óptimas
            print("Pasos detallados:")
            for step_idx, (action, next_state) in enumerate(zip(demo['actions'], demo['path'][1:])):
                current_state = demo['path'][step_idx]
                current_pos = f"({current_state//4},{current_state%4})"
                next_pos = f"({next_state//4},{next_state%4})"
                print(f"  {step_idx+1}. {current_pos} --{action_names[action]}--> {next_pos}")
        
        if i < demos_to_show - 1:
            input("Presiona Enter para la siguiente demostración...")
    
    print(f"\n{LogConfig.EMOJIS['SUCCESS']} Resumen de demostraciones:")
    print(f"Demos exitosas mostradas: {demos_to_show}")
    print(f"Demos óptimas: {optimal_demos}/{demos_to_show} ({optimal_demos/demos_to_show:.1%})")
    
    if len(demos_found) > 0:
        avg_steps_shown = np.mean([demo['steps'] for demo in demos_found[:demos_to_show]])
        print(f"Pasos promedio en demos mostradas: {avg_steps_shown:.1f}")

def plot_enhanced_results(scores, moving_averages, optimal_counts, steps_per_episode):
    """
    Visualizar resultados del entrenamiento mejorado
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    episodes = range(1, len(scores) + 1)
    
    # Gráfica 1: Recompensas y progreso
    ax1.plot(episodes, scores, alpha=0.6, color='lightblue', linewidth=0.8, label='Recompensas')
    ax1.plot(episodes, moving_averages, color='red', linewidth=2, label=f'Promedio móvil ({TrainingConfig.SOLVE_WINDOW})')
    ax1.axhline(y=TrainingConfig.SOLVE_SCORE, color='green', linestyle='--', label='Objetivo')
    ax1.set_xlabel('Episodio')
    ax1.set_ylabel('Recompensa')
    ax1.set_title('Progreso del Entrenamiento Mejorado')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfica 2: Rutas óptimas acumuladas
    ax2.plot(episodes, optimal_counts, color='orange', linewidth=2, marker='o', markersize=2)
    ax2.set_xlabel('Episodio')
    ax2.set_ylabel('Rutas Óptimas Acumuladas')
    ax2.set_title('Descubrimiento de Rutas Óptimas')
    ax2.grid(True, alpha=0.3)
    
    # Gráfica 3: Distribución de pasos
    successful_steps = [steps for i, steps in enumerate(steps_per_episode) if scores[i] > 0]
    if successful_steps:
        ax3.hist(successful_steps, bins=min(30, max(successful_steps)), alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(x=EvalConfig.OPTIMAL_STEPS, color='red', linestyle='--', label=f'Óptimo: {EvalConfig.OPTIMAL_STEPS} pasos')
        ax3.axvline(x=np.mean(successful_steps), color='orange', linestyle='--', label=f'Promedio: {np.mean(successful_steps):.1f}')
        ax3.set_xlabel('Pasos a la Meta')
        ax3.set_ylabel('Frecuencia')
        ax3.set_title('Distribución de Pasos en Episodios Exitosos')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Gráfica 4: Evolución de eficiencia
    window_size = 100
    efficiency_over_time = []
    for i in range(window_size, len(scores) + 1):
        window_scores = scores[i-window_size:i]
        window_steps = steps_per_episode[i-window_size:i]
        
        # Calcular eficiencia: % de episodios exitosos con <= 10 pasos
        successful_and_efficient = sum(1 for j, score in enumerate(window_scores) 
                                     if score > 0 and window_steps[j] <= 10)
        total_successful = sum(1 for score in window_scores if score > 0)
        
        if total_successful > 0:
            efficiency = successful_and_efficient / total_successful
        else:
            efficiency = 0
        
        efficiency_over_time.append(efficiency)
    
    if efficiency_over_time:
        ax4.plot(range(window_size, len(scores) + 1), efficiency_over_time, color='green', linewidth=2)
        ax4.axhline(y=0.5, color='red', linestyle='--', label='50% eficiencia')
        ax4.set_xlabel('Episodio')
        ax4.set_ylabel('Tasa de Eficiencia')
        ax4.set_title(f'Eficiencia (≤10 pasos) - Ventana {window_size}')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('enhanced_training_results.png', dpi=300, bbox_inches='tight')
    print(f"{LogConfig.EMOJIS['SAVE']} Gráfica mejorada guardada en enhanced_training_results.png")
    plt.show()

def main():
    """Función principal mejorada"""
    print(f"{LogConfig.COLORS['BOLD']}🧊 === FROZEN LAKE DQN MEJORADO ==={LogConfig.COLORS['END']}")
    print("🎯 Versión optimizada con recompensas moldeadas para rutas óptimas")
    
    validate_config()
    
    # Crear entorno y agente mejorado
    env = create_environment()
    agent = create_enhanced_agent()
    
    print(f"\n{LogConfig.EMOJIS['INFO']} Configuración del agente mejorado:")
    print(f"  Epsilon: {DQNConfig.EPSILON_START} → {DQNConfig.EPSILON_END} (decay: {DQNConfig.EPSILON_DECAY})")
    print(f"  Episodios: {TrainingConfig.NUM_EPISODES}")
    print(f"  Recompensas moldeadas: ✅")
    print(f"  Exploración dirigida: ✅")
    print(f"  Bonificaciones por rutas teóricas: ✅")
    
    print(f"\n{LogConfig.EMOJIS['INFO']} Opciones:")
    print("1. Entrenar agente mejorado")
    print("2. Cargar y evaluar agente mejorado")
    print("3. Cargar y demostrar agente mejorado")
    print("4. Flujo completo mejorado")
    
    choice = input("\nSelecciona una opción (1-4): ").strip()
    
    if choice == "1":
        # Solo entrenar
        print(f"\n{LogConfig.EMOJIS['TRAIN']} Iniciando entrenamiento mejorado...")
        scores, moving_averages, optimal_counts, steps_per_episode = train_enhanced_agent(agent, env)
        plot_enhanced_results(scores, moving_averages, optimal_counts, steps_per_episode)
        
    elif choice == "2":
        # Cargar y evaluar
        agent.load("enhanced_frozen_lake_dqn.pth")
        results = evaluate_enhanced_agent(agent, env)
        
    elif choice == "3":
        # Cargar y demostrar
        agent.load("enhanced_frozen_lake_dqn.pth")
        demonstrate_enhanced_agent(agent, env)
        
    elif choice == "4":
        # Flujo completo mejorado
        print(f"\n{LogConfig.EMOJIS['TRAIN']} Ejecutando flujo completo mejorado...")
        
        # Entrenar
        scores, moving_averages, optimal_counts, steps_per_episode = train_enhanced_agent(agent, env)
        plot_enhanced_results(scores, moving_averages, optimal_counts, steps_per_episode)
        
        # Evaluar
        results = evaluate_enhanced_agent(agent, env)
        
        # Demostrar si hay rutas óptimas
        if results['optimal_rate'] > 0:
            print(f"\n{LogConfig.EMOJIS['SUCCESS']} ¡Se encontraron rutas óptimas! Mostrando demostraciones...")
            demonstrate_enhanced_agent(agent, env)
        else:
            print(f"\n{LogConfig.EMOJIS['WARNING']} No se encontraron rutas óptimas en evaluación.")
            
        # Comparación con objetivos
        print(f"\n{LogConfig.EMOJIS['INFO']} === COMPARACIÓN CON OBJETIVOS ===")
        print(f"Tasa de éxito: {results['success_rate']:.1%} (objetivo: {EvalConfig.TARGET_SUCCESS_RATE:.1%})")
        print(f"Rutas óptimas: {results['optimal_rate']:.1%} (objetivo: {EvalConfig.TARGET_OPTIMAL_RATE:.1%})")
        print(f"Rutas teóricas: {results['theoretical_paths_found']}/3")
        
        # Evaluación final
        if results['success_rate'] >= EvalConfig.TARGET_SUCCESS_RATE:
            print(f"{LogConfig.EMOJIS['SUCCESS']} ¡Objetivo de tasa de éxito ALCANZADO!")
        
        if results['optimal_rate'] >= EvalConfig.TARGET_OPTIMAL_RATE:
            print(f"{LogConfig.EMOJIS['OPTIMAL']} ¡Objetivo de rutas óptimas ALCANZADO!")
            
        if results['theoretical_paths_found'] == 3:
            print(f"{LogConfig.EMOJIS['OPTIMAL']} ¡TODAS las rutas teóricas encontradas!")
            
        # Mejora respecto a versión anterior
        previous_optimal_rate = 0.0  # De los resultados anteriores
        if results['optimal_rate'] > previous_optimal_rate:
            improvement = results['optimal_rate'] / max(previous_optimal_rate, 0.001)
            print(f"📈 Mejora: {improvement:.1f}x mejor que la versión anterior")
            
    else:
        print(f"{LogConfig.EMOJIS['ERROR']} Opción no válida.")
    
    env.close()
    print(f"\n{LogConfig.EMOJIS['SUCCESS']} ¡Programa terminado!")

if __name__ == "__main__":
    main()