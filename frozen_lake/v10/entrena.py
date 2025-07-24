import gymnasium as gym
import numpy as np
import time
import argparse
import os
import matplotlib.pyplot as plt
from collections import deque

def parse_arguments():
    parser = argparse.ArgumentParser(description="Entrena y prueba un agente Q-learning para Frozen Lake, optimizando para el mínimo número de pasos (6).")
    parser.add_argument('--episodes', type=int, default=1_000_000, help='Número de episodios de entrenamiento')
    parser.add_argument('--max_steps', type=int, default=100, help='Máximo de pasos por episodio')
    parser.add_argument('--alpha', type=float, default=0.2, help='Tasa de aprendizaje inicial')  # Aumentado para mejor convergencia
    parser.add_argument('--alpha_min', type=float, default=0.01, help='Tasa de aprendizaje mínima')
    parser.add_argument('--gamma', type=float, default=0.95, help='Factor de descuento')  # Reducido para priorizar recompensas inmediatas
    parser.add_argument('--epsilon', type=float, default=1.0, help='Tasa de exploración inicial')  # Comenzar con exploración completa
    parser.add_argument('--epsilon_min', type=float, default=0.01, help='Tasa de exploración mínima')
    parser.add_argument('--epsilon_decay', type=float, default=0.999995, help='Decaimiento de epsilon')  # Decaimiento más lento
    parser.add_argument('--step_penalty', type=float, default=-0.02, help='Penalización por paso')  # Penalización más fuerte
    parser.add_argument('--goal_proximity_reward', type=float, default=0.1, help='Recompensa por cercanía al objetivo')
    parser.add_argument('--test_episodes', type=int, default=100, help='Número de episodios de prueba')
    parser.add_argument('--save_path', type=str, default='q_table_opt_steps.npy', help='Ruta para guardar la Q-table')
    parser.add_argument('--render_mode', type=str, default='none', choices=['human', 'rgb_array', 'none'], help='Modo de renderizado')  # Añadido 'none'
    parser.add_argument('--force_train', action='store_true', help='Forzar nuevo entrenamiento')
    parser.add_argument('--is_slippery', type=bool, default=True, help='Superficie resbaladiza')
    parser.add_argument('--print_interval', type=int, default=50_000, help='Intervalo para imprimir métricas')
    parser.add_argument('--train', action='store_true', help='Ejecutar fase de entrenamiento')
    parser.add_argument('--test', action='store_true', help='Ejecutar fase de prueba')
    return parser.parse_args()

def train_agent(args):
    env = gym.make("FrozenLake-v1", is_slippery=args.is_slippery)
    q_table = np.zeros((env.observation_space.n, env.action_space.n))  # Inicialización mejorada
    
    epsilon = args.epsilon
    alpha = args.alpha
    stats = {
        'successes': 0,
        'steps': deque(maxlen=1000),
        'interval_successes': 0,
        'interval_steps': [],
        'rewards': deque(maxlen=1000)
    }
    
    for episode in range(args.episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0
        visited_states = set()

        for _ in range(args.max_steps):
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Modificación de recompensas
            modified_reward = reward + args.step_penalty
            if next_state in [5, 7, 11, 12]:  # Hoyos
                modified_reward = -1.0
            elif next_state == 15:  # Objetivo
                modified_reward = 1.0
            elif next_state in [10, 13, 14]:  # Cercanía al objetivo
                if next_state not in visited_states:
                    modified_reward += args.goal_proximity_reward
                    visited_states.add(next_state)

            # Actualización Q-table
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            new_value = (1 - alpha) * old_value + alpha * (modified_reward + args.gamma * next_max)
            q_table[state, action] = new_value

            state = next_state
            total_reward += modified_reward
            steps += 1

            if done:
                if reward > 0:  # Éxito solo si se alcanza el objetivo
                    stats['successes'] += 1
                    stats['interval_successes'] += 1
                    stats['steps'].append(steps)
                    stats['interval_steps'].append(steps)
                stats['rewards'].append(total_reward)
                break

        # Decaimiento de parámetros
        epsilon = max(args.epsilon_min, epsilon * args.epsilon_decay)
        alpha = max(args.alpha_min, alpha * 0.99999)

        # Reporte de progreso
        if (episode + 1) % args.print_interval == 0:
            success_rate = stats['interval_successes'] / args.print_interval * 100
            avg_reward = np.mean(stats['rewards']) if stats['rewards'] else 0
            avg_steps = np.mean(stats['interval_steps']) if stats['interval_steps'] else 0
            
            optimal_steps = len([s for s in stats['interval_steps'] if 6 <= s <= 10])
            exact_optimal = len([s for s in stats['interval_steps'] if s == 6])
            long_steps = len([s for s in stats['interval_steps'] if s > 30])
            
            total_successes = stats['interval_successes'] if stats['interval_successes'] > 0 else 1
            optimal_rate = optimal_steps / total_successes * 100
            exact_optimal_rate = exact_optimal / total_successes * 100
            long_steps_rate = long_steps / total_successes * 100
            
            print(f"Episodios {episode + 1 - args.print_interval + 1}-{episode + 1}:")
            print(f"  Tasa de éxito: {success_rate:.2f}% | Recompensa promedio: {avg_reward:.4f}")
            print(f"  Pasos promedio (éxitos): {avg_steps:.2f}")
            print(f"  Trayectorias óptimas (6-10 pasos): {optimal_rate:.2f}%")
            print(f"  Trayectorias exactas (6 pasos): {exact_optimal_rate:.2f}%")
            print(f"  Trayectorias largas (>30 pasos): {long_steps_rate:.2f}%")
            print(f"  Epsilon: {epsilon:.4f} | Alpha: {alpha:.4f}")
            
            # Reinicio de estadísticas del intervalo
            stats['interval_successes'] = 0
            stats['interval_steps'] = []

    env.close()
    np.save(args.save_path, q_table)
    print(f"\nEntrenamiento completado. Q-table guardada en {args.save_path}")
    return q_table

def test_agent(q_table, args):
    env = gym.make("FrozenLake-v1", is_slippery=args.is_slippery)
    if args.render_mode == 'rgb_array':
        plt.ion()
    
    successes = 0
    steps_list = []

    for ep in range(args.test_episodes):
        state, _ = env.reset()
        done = False
        steps = 0
        episode_result = ""

        for _ in range(args.max_steps):
            if args.render_mode == 'human':
                env.render()
            elif args.render_mode == 'rgb_array':
                try:
                    img = env.render()
                    plt.imshow(img)
                    plt.title(f"Episodio {ep + 1}, Paso {steps}")
                    plt.axis('off')
                    plt.pause(0.01)
                    plt.clf()
                except Exception:
                    pass

            action = np.argmax(q_table[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

            if done:
                if reward > 0:
                    successes += 1
                    steps_list.append(steps)
                    episode_result = f"ÉXITO en {steps} pasos"
                else:
                    if state in [5, 7, 11, 12]:
                        episode_result = f"CAÍDA EN HOYO (paso {steps})"
                    else:
                        episode_result = f"TIEMPO AGOTADO ({steps} pasos)"
                break

        print(f"Episodio {ep + 1}: {episode_result}")

    # Reporte final de pruebas
    success_rate = successes / args.test_episodes * 100
    print(f"\nResumen de prueba ({args.test_episodes} episodios):")
    print(f"Tasa de éxito: {success_rate:.2f}%")
    
    if successes > 0:
        avg_steps = np.mean(steps_list)
        min_steps = np.min(steps_list)
        max_steps = np.max(steps_list)
        optimal_rate = len([s for s in steps_list if s <= 10]) / successes * 100
        
        print(f"Pasos promedio: {avg_steps:.2f} (min: {min_steps}, max: {max_steps})")
        print(f"Trayectorias óptimas (<=10 pasos): {optimal_rate:.2f}%")
        
        plt.figure(figsize=(10, 6))
        plt.hist(steps_list, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=6, color='red', linestyle='--', label='Óptimo (6 pasos)')
        plt.title('Distribución de Pasos en Éxitos')
        plt.xlabel('Pasos')
        plt.ylabel('Frecuencia')
        plt.legend()
        plt.savefig('frozen_lake_steps_distribution.png')
        plt.close()

    env.close()
    if args.render_mode == 'rgb_array':
        plt.ioff()

def main():
    args = parse_arguments()
    
    if args.train:
        print("Iniciando entrenamiento...")
        q_table = train_agent(args)
    else:
        if os.path.exists(args.save_path):
            q_table = np.load(args.save_path)
            print(f"Q-table cargada desde {args.save_path}")
        else:
            print("No se encontró Q-table. Ejecute con --train primero.")
            return
    
    if args.test:
        print("\nIniciando pruebas...")
        test_agent(q_table, args)

if __name__ == "__main__":
    main()