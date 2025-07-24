import gymnasium as gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Entrena y prueba un agente Q-learning puro para Frozen Lake, optimizando para el mínimo número de pasos (6).")
    parser.add_argument('--episodes', type=int, default=300_000, help='Número de episodios de entrenamiento')
    parser.add_argument('--max_steps', type=int, default=200, help='Máximo de pasos por episodio')
    parser.add_argument('--alpha', type=float, default=0.1, help='Tasa de aprendizaje inicial')
    parser.add_argument('--alpha_min', type=float, default=0.01, help='Tasa de aprendizaje mínima')
    parser.add_argument('--gamma', type=float, default=0.99, help='Factor de descuento')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Tasa de exploración inicial')
    parser.add_argument('--epsilon_min', type=float, default=0.01, help='Tasa de exploración mínima')
    parser.add_argument('--epsilon_decay', type=float, default=0.99995, help='Decaimiento de epsilon')
    parser.add_argument('--step_penalty', type=float, default=-0.005, help='Penalización por paso')
    parser.add_argument('--test_episodes', type=int, default=1000, help='Número de episodios de prueba')
    parser.add_argument('--save_path', type=str, default='q_table_opt_steps_v15.npy', help='Ruta para guardar/cargar la Q-table')
    parser.add_argument('--render_mode', type=str, default='rgb_array', choices=['human', 'rgb_array'], help='Modo de renderizado: human o rgb_array')
    parser.add_argument('--no_render', action='store_true', help='Desactiva el renderizado y salida por episodio durante la prueba')
    parser.add_argument('--force_train', action='store_true', help='Forzar nuevo entrenamiento incluso si existe Q-table')
    parser.add_argument('--is_slippery', type=bool, default=True, help='Habilitar superficie resbaladiza (True) o determinista (False)')
    parser.add_argument('--print_interval', type=int, default=100000, help='Intervalo de episodios para imprimir métricas')
    return parser.parse_args()

def render_rgb_array(env, state, step, episode, steps_taken):
    try:
        img = env.render()
        plt.imshow(img)
        plt.title(f"Episodio {episode + 1}, Paso {steps_taken}")
        plt.axis('off')
        plt.pause(0.1)
        plt.clf()
    except Exception as e:
        print(f"Error en render_rgb_array: {e}")

def inspect_q_table(q_table):
    print("\nInspección de la Q-table (acciones óptimas):")
    actions = {0: 'Izquierda', 1: 'Abajo', 2: 'Derecha', 3: 'Arriba'}
    key_states = [0, 1, 2, 5, 6, 8, 9, 10, 13, 14, 15]
    optimal_paths = {
        '0→1→2→6→10→14→15': {0: 2, 1: 2, 2: 1, 6: 1, 10: 1, 14: 1},
        '0→4→8→9→10→14→15': {0: 1, 4: 1, 8: 2, 9: 2, 10: 1, 14: 1},
        '0→4→8→9→13→14→15': {0: 1, 4: 1, 8: 2, 9: 1, 13: 2, 14: 1}
    }
    for state in key_states:
        best_action = np.argmax(q_table[state])
        q_values = q_table[state]
        is_optimal = []
        for path_name, path in optimal_paths.items():
            if state in path and best_action == path[state]:
                is_optimal.append(f"Óptima para {path_name}")
        is_optimal = ", ".join(is_optimal) if is_optimal else "Subóptima"
        print(f"Estado {state}: Mejor acción = {actions[best_action]} ({is_optimal}), Q-valores = {q_values}")

def train_agent(args):
    env = gym.make("FrozenLake-v1", is_slippery=args.is_slippery)
    q_table = np.ones((env.observation_space.n, env.action_space.n)) * (0.8 if not args.is_slippery else 0.5)
    q_table[14, :] = 0.9  # Inicialización alta para el estado cercano al objetivo
    q_table[15, :] = 0.0  # Estado objetivo

    epsilon = args.epsilon
    alpha = args.alpha
    successes = 0
    rewards = []
    steps_in_success = []
    optimal_steps = 0
    exact_optimal_steps = 0
    long_steps = 0

    total_episodes = 100_000 if not args.is_slippery else args.episodes
    epsilon_decay = 0.9995 if not args.is_slippery else args.epsilon_decay

    for episode in range(total_episodes):
        state, _ = env.reset()
        state = int(state)
        total_reward = 0
        done = False
        steps = 0

        for _ in range(args.max_steps):
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = int(next_state)
            done = terminated or truncated

            reward += args.step_penalty

            steps += 1

            old_value = q_table[state, action]
            next_max = 0 if done else np.max(q_table[next_state])
            q_table[state, action] = old_value + alpha * (reward + args.gamma * next_max - old_value)

            if episode < 1000 and state in [0, 1, 2, 5, 6, 8, 9, 10, 13, 14]:
                print(f"Episodio {episode + 1}, Estado {state}, Acción {action}, Recompensa {reward:.4f}, Q-valor actualizado {q_table[state, action]:.4f}")

            state = next_state
            total_reward += reward

            if done:
                if reward > 0:
                    successes += 1
                    steps_in_success.append(steps)
                    if 6 <= steps <= 10:
                        optimal_steps += 1
                    if steps == 6:
                        exact_optimal_steps += 1
                    if steps > 30:
                        long_steps += 1
                break

        rewards.append(total_reward)
        epsilon = max(args.epsilon_min, epsilon * epsilon_decay)
        alpha = max(args.alpha_min, args.alpha - (args.alpha - args.alpha_min) * episode / total_episodes)

        if (episode + 1) % args.print_interval == 0:
            success_rate = successes / args.print_interval * 100
            avg_reward = np.mean(rewards[-args.print_interval:])
            success_count = min(len(steps_in_success), args.print_interval)
            avg_steps = np.mean(steps_in_success[-success_count:]) if success_count > 0 else 0
            optimal_rate = (optimal_steps / success_count * 100) if success_count > 0 else 0
            exact_optimal_rate = (exact_optimal_steps / success_count * 100) if success_count > 0 else 0
            long_steps_rate = (long_steps / success_count * 100) if success_count > 0 else 0
            print(f"Episodios {episode + 1 - args.print_interval + 1}-{episode + 1}: "
                  f"Tasa de éxito: {success_rate:.2f}%, Recompensa promedio: {avg_reward:.4f}, "
                  f"Pasos promedio en éxitos: {avg_steps:.2f}, Éxitos en 6-10 pasos: {optimal_rate:.2f}%, "
                  f"Éxitos en 6 pasos: {exact_optimal_rate:.2f}%, Éxitos con >30 pasos: {long_steps_rate:.2f}%")
            successes = 0
            optimal_steps = 0
            exact_optimal_steps = 0
            long_steps = 0

    env.close()
    np.save(args.save_path, q_table)
    print(f"Q-table guardada en {args.save_path}")
    inspect_q_table(q_table)
    return q_table

def test_agent(q_table, args):
    env = gym.make("FrozenLake-v1", is_slippery=args.is_slippery) if args.no_render else gym.make("FrozenLake-v1", render_mode=args.render_mode, is_slippery=args.is_slippery)
    if not args.no_render and args.render_mode == 'rgb_array':
        plt.ion()

    success_steps = []
    successes = 0
    optimal_steps = 0
    exact_optimal_steps = 0
    hole_falls = 0
    timeout_fails = 0
    successful_trajectories = []

    for ep in range(args.test_episodes):
        state, _ = env.reset()
        state = int(state)
        done = False
        steps = 0
        trajectory = [state]

        for _ in range(args.max_steps):
            if not args.no_render:
                try:
                    if args.render_mode == 'human':
                        env.render()
                    elif args.render_mode == 'rgb_array':
                        render_rgb_array(env, state, steps, ep, steps + 1)
                except Exception as e:
                    print(f"Error de renderizado: {e}. Continuando sin renderizar.")
                    break

            action = int(np.argmax(q_table[state]))
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = int(next_state)
            done = terminated or truncated
            steps += 1
            trajectory.append(next_state)

            if done:
                if not args.no_render:
                    try:
                        if args.render_mode == 'human':
                            env.render()
                        elif args.render_mode == 'rgb_array':
                            render_rgb_array(env, state, steps, ep, steps + 1)
                    except Exception as e:
                        print(f"Error de renderizado en estado final: {e}")
                if reward > 0:
                    successes += 1
                    success_steps.append(steps)
                    if 6 <= steps <= 10:
                        optimal_steps += 1
                    if steps == 6:
                        exact_optimal_steps += 1
                    if len(successful_trajectories) < 10:
                        successful_trajectories.append(trajectory)
                else:
                    if terminated:
                        hole_falls += 1
                    elif truncated:
                        timeout_fails += 1
                break
            state = next_state

    total_fails = hole_falls + timeout_fails
    if success_steps:
        print(f"\nResumen de la prueba: Tasa de éxito: {successes / args.test_episodes * 100:.2f}%, "
              f"Pasos promedio en éxitos: {np.mean(success_steps):.2f}, "
              f"Éxitos en 6-10 pasos: {optimal_steps / successes * 100:.2f}%, "
              f"Éxitos en 6 pasos: {exact_optimal_steps / successes * 100:.2f}%, "
              f"Fallos por agujero: {hole_falls}, Fallos por tiempo agotado: {timeout_fails}, "
              f"Total episodios contabilizados: {successes + total_fails}")
        if successful_trajectories:
            print("\nPrimeras 10 trayectorias exitosas:")
            for i, traj in enumerate(successful_trajectories):
                print(f"Trayectoria {i + 1}: {traj}")
    else:
        print(f"\nResumen de la prueba: Tasa de éxito: 0.00%, "
              f"Pasos promedio en éxitos: N/A, "
              f"Éxitos en 6-10 pasos: N/A, Éxitos en 6 pasos: N/A, "
              f"Fallos por agujero: {hole_falls}, Fallos por tiempo agotado: {timeout_fails}, "
              f"Total episodios contabilizados: {successes + total_fails}")

    if not args.no_render and success_steps:
        plt.hist(success_steps, bins=range(min(success_steps), max(success_steps) + 2, 1), edgecolor='black')
        plt.title('Distribución de Pasos en Episodios Exitosos (Prueba)')
        plt.xlabel('Número de Pasos')
        plt.ylabel('Frecuencia')
        plt.axvline(x=6, color='r', linestyle='--', label='Óptimo (6 pasos)')
        plt.axvline(x=np.mean(success_steps), color='b', linestyle='--', label=f'Promedio ({np.mean(success_steps):.2f} pasos)')
        plt.legend()
        plt.show()

    env.close()
    if not args.no_render and args.render_mode == 'rgb_array':
        plt.ioff()
        plt.close()

def main():
    args = parse_arguments()
    print("Entrenando agente...")
    if not args.force_train and os.path.exists(args.save_path):
        print(f"Cargando Q-table desde {args.save_path}")
        q_table = np.load(args.save_path)
    else:
        q_table = train_agent(args)
    print("Entrenamiento finalizado.")
    print("Probando agente...")
    test_agent(q_table, args)

if __name__ == "__main__":
    main()