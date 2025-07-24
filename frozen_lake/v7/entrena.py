import gymnasium as gym
import numpy as np
import time
import argparse
import os
import matplotlib.pyplot as plt

def parse_arguments():
    # Configura los argumentos de línea de comandos para personalizar hiperparámetros
    parser = argparse.ArgumentParser(description="Entrena y prueba un agente Q-learning para Frozen Lake, optimizando para el mínimo número de pasos (6).")
    parser.add_argument('--episodes', type=int, default=500_000, help='Número de episodios de entrenamiento')
    parser.add_argument('--max_steps', type=int, default=100, help='Máximo de pasos por episodio')
    parser.add_argument('--alpha', type=float, default=0.1, help='Tasa de aprendizaje inicial')
    parser.add_argument('--alpha_min', type=float, default=0.01, help='Tasa de aprendizaje mínima')
    parser.add_argument('--gamma', type=float, default=0.99, help='Factor de descuento')
    parser.add_argument('--epsilon', type=float, default=0.7, help='Tasa de exploración inicial')
    parser.add_argument('--epsilon_min', type=float, default=0.001, help='Tasa de exploración mínima')
    parser.add_argument('--epsilon_decay', type=float, default=0.999, help='Decaimiento de epsilon en etapas tardías')
    parser.add_argument('--step_penalty', type=float, default=-0.01, help='Penalización por paso para minimizar trayectorias')
    parser.add_argument('--goal_proximity_reward', type=float, default=0.1, help='Recompensa por alcanzar estados cercanos al objetivo')
    parser.add_argument('--test_episodes', type=int, default=10, help='Número de episodios de prueba')
    parser.add_argument('--save_path', type=str, default='q_table_opt_steps_v3.npy', help='Ruta para guardar/cargar la Q-table')
    parser.add_argument('--render_mode', type=str, default='human', choices=['human', 'rgb_array'], help='Modo de renderizado: human o rgb_array')
    parser.add_argument('--force_train', action='store_true', help='Forzar nuevo entrenamiento incluso si existe Q-table')
    return parser.parse_args()

def render_rgb_array(env, state, step, episode, steps_taken):
    # Renderiza el entorno como una imagen RGB y la muestra con Matplotlib
    img = env.render()
    plt.imshow(img)
    plt.title(f"Episodio {episode + 1}, Paso {steps_taken}")
    plt.axis('off')
    plt.pause(0.01)  # Pausa breve para visualización
    plt.clf()  # Limpia la figura para el siguiente fotograma

def train_agent(args):
    # Crea el entorno FrozenLake-v1 con superficie resbaladiza
    env = gym.make("FrozenLake-v1", is_slippery=False)
    # Inicializa la tabla Q con valores optimistas para fomentar exploración
    q_table = np.ones((env.observation_space.n, env.action_space.n)) * 0.5
    # Establece valores Q en el estado objetivo (15) a 0 para evitar acciones innecesarias
    q_table[15, :] = 0.0
    
    epsilon = args.epsilon  # Tasa de exploración inicial
    alpha = args.alpha  # Tasa de aprendizaje inicial
    successes = 0  # Contador de episodios exitosos
    rewards = []  # Lista para almacenar recompensas
    steps_in_success = []  # Lista para almacenar pasos en episodios exitosos
    optimal_steps = 0  # Contador de episodios exitosos con 6-10 pasos
    exact_optimal_steps = 0  # Contador de episodios exitosos con exactamente 6 pasos

    # Entrena durante el número especificado de episodios
    for episode in range(args.episodes):
        state, _ = env.reset()  # Reinicia el entorno
        total_reward = 0  # Recompensa total del episodio
        done = False  # Indicador de fin de episodio
        steps = 0  # Contador de pasos en el episodio

        # Ejecuta hasta el máximo de pasos o hasta que termine el episodio
        for _ in range(args.max_steps):
            # Decide si explorar (aleatorio) o explotar (mejor acción)
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Acción aleatoria
            else:
                action = np.argmax(q_table[state])  # Mejor acción según Q-table

            # Realiza la acción y obtiene el nuevo estado y recompensa
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # Termina si se cae o llega al objetivo

            # Aplica penalización por paso
            reward += args.step_penalty
            # Añade recompensa adicional por estados cercanos al objetivo (p. ej., estado 14)
            if next_state == 14:
                reward += args.goal_proximity_reward

            steps += 1

            # Actualiza la Q-table usando la fórmula de Q-learning
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            q_table[state, action] = old_value + alpha * (reward + args.gamma * next_max - old_value)

            state = next_state  # Actualiza el estado
            total_reward += reward  # Acumula la recompensa

            if done:
                if reward > 0:  # Incrementa éxitos si se llega al objetivo
                    successes += 1
                    steps_in_success.append(steps)
                    if 6 <= steps <= 10:
                        optimal_steps += 1
                    if steps == 6:
                        exact_optimal_steps += 1
                break

        rewards.append(total_reward)  # Guarda la recompensa del episodio
        # Reduce epsilon con un decaimiento no lineal
        if episode < 150_000:
            epsilon = max(args.epsilon_min, epsilon * 0.995)
        else:
            epsilon = max(args.epsilon_min, epsilon * args.epsilon_decay)
        # Reduce alpha linealmente
        alpha = max(args.alpha_min, args.alpha - (args.alpha - args.alpha_min) * episode / args.episodes)

        # Imprime métricas cada 10,000 episodios
        if (episode + 1) % 10000 == 0:
            success_rate = successes / 10000 * 100
            avg_reward = np.mean(rewards[-10000:])
            avg_steps = np.mean(steps_in_success[-1000:]) if steps_in_success else 0
            optimal_rate = optimal_steps / max(1, len([s for s in steps_in_success[-1000:] if s > 0])) * 100
            exact_optimal_rate = exact_optimal_steps / max(1, len([s for s in steps_in_success[-1000:] if s > 0])) * 100
            print(f"Episodios {episode + 1 - 9999}-{episode + 1}: Tasa de éxito: {success_rate:.2f}%, "
                  f"Recompensa promedio: {avg_reward:.4f}, Pasos promedio en éxitos: {avg_steps:.2f}, "
                  f"Éxitos en 6-10 pasos: {optimal_rate:.2f}%, Éxitos en 6 pasos: {exact_optimal_rate:.2f}%")
            successes = 0  # Reinicia el contador de éxitos
            optimal_steps = 0  # Reinicia el contador de éxitos óptimos
            exact_optimal_steps = 0  # Reinicia el contador de éxitos en 6 pasos

    env.close()  # Cierra el entorno
    np.save(args.save_path, q_table)  # Guarda la Q-table
    print(f"Q-table guardada en {args.save_path}")
    return q_table

def test_agent(q_table, args):
    # Crea el entorno con el modo de renderizado especificado
    env = gym.make("FrozenLake-v1", render_mode=args.render_mode, is_slippery=True)
    if args.render_mode == 'rgb_array':
        plt.ion()  # Activa el modo interactivo de Matplotlib
    
    # Lista para almacenar pasos en episodios exitosos durante la prueba
    success_steps = []

    # Prueba el agente durante el número especificado de episodios
    for ep in range(args.test_episodes):
        state, _ = env.reset()  # Reinicia el entorno
        done = False  # Indicador de fin de episodio
        total_reward = 0  # Recompensa total del episodio
        steps = 0  # Contador de pasos en el episodio
        print(f"\nEpisodio {ep + 1}")

        # Ejecuta hasta el máximo de pasos o hasta que termine el episodio
        for _ in range(args.max_steps):
            try:
                if args.render_mode == 'human':
                    env.render()  # Renderiza en modo humano (SDL2)
                elif args.render_mode == 'rgb_array':
                    render_rgb_array(env, state, steps, ep)  # Renderiza con Matplotlib
            except Exception as e:
                print(f"Error de renderizado: {e}. Cambia a --render_mode=rgb_array si el problema persiste.")
                break

            action = np.argmax(q_table[state])  # Selecciona la mejor acción
            state, reward, terminated, truncated, _ = env.step(action)  # Realiza la acción
            done = terminated or truncated
            total_reward += reward
            steps += 1

            time.sleep(0.01)  # Pausa breve para visualización fluida
            if done:
                if args.render_mode == 'human':
                    env.render()  # Renderiza el estado final
                elif args.render_mode == 'rgb_array':
                    render_rgb_array(env, state, steps, ep)
                print(f"Episodio {ep + 1} terminado. Recompensa total: {total_reward:.2f}, Pasos: {steps}")
                if reward > 0:
                    print(f"¡Éxito: Llegaste al objetivo en {steps} pasos!")
                    success_steps.append(steps)
                else:
                    print("Fallo: Caíste en un agujero o se agotó el tiempo.")
                time.sleep(0.05)
                break

    # Muestra un histograma de pasos en episodios exitosos
    if success_steps:
        plt.hist(success_steps, bins=range(min(success_steps), max(success_steps) + 2), edgecolor='black')
        plt.title('Distribución de Pasos en Episodios Exitosos (Prueba)')
        plt.xlabel('Número de Pasos')
        plt.ylabel('Frecuencia')
        plt.axvline(x=6, color='r', linestyle='--', label='Óptimo (6 pasos)')
        plt.legend()
        plt.show()

    env.close()  # Cierra el entorno
    if args.render_mode == 'rgb_array':
        plt.ioff()  # Desactiva el modo interactivo
        plt.close()

def main():
    args = parse_arguments()  # Obtiene los argumentos
    
    # Forzar entrenamiento por defecto
    print("Entrenando agente...")
    q_table = train_agent(args)
    print("Entrenamiento finalizado.")

    print("Probando agente con renderizado...")
    test_agent(q_table, args)

if __name__ == "__main__":
    main()