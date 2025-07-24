import gymnasium as gym
import numpy as np
import time
import argparse
import os
import matplotlib.pyplot as plt

def parse_arguments():
    # Configura los argumentos de línea de comandos para personalizar hiperparámetros
    parser = argparse.ArgumentParser(description="Entrena y prueba un agente Q-learning para Frozen Lake, optimizando para el mínimo número de pasos.")
    parser.add_argument('--episodes', type=int, default=200_000, help='Número de episodios de entrenamiento')
    parser.add_argument('--max_steps', type=int, default=100, help='Máximo de pasos por episodio')
    parser.add_argument('--alpha', type=float, default=0.05, help='Tasa de aprendizaje')
    parser.add_argument('--gamma', type=float, default=0.99, help='Factor de descuento')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Tasa de exploración inicial')
    parser.add_argument('--epsilon_min', type=float, default=0.05, help='Tasa de exploración mínima')
    parser.add_argument('--epsilon_decay', type=float, default=0.9995, help='Decaimiento de epsilon')
    parser.add_argument('--step_penalty', type=float, default=-0.04, help='Penalización por paso para minimizar trayectorias')
    parser.add_argument('--test_episodes', type=int, default=10, help='Número de episodios de prueba')
    parser.add_argument('--save_path', type=str, default='q_table_opt.npy', help='Ruta para guardar/cargar la Q-table')
    parser.add_argument('--render_mode', type=str, default='human', choices=['human', 'rgb_array'], help='Modo de renderizado: human o rgb_array')
    return parser.parse_args()

def render_rgb_array(env, state, step, episode, steps_taken):
    # Renderiza el entorno como una imagen RGB y la muestra con Matplotlib
    img = env.render()
    plt.imshow(img)
    plt.title(f"Episodio {episode + 1}, Paso {steps_taken}")
    plt.axis('off')
    plt.pause(0.1)  # Pausa breve para visualización
    plt.clf()  # Limpia la figura para el siguiente fotograma

def train_agent(args):
    # Crea el entorno FrozenLake-v1 con superficie resbaladiza
    env = gym.make("FrozenLake-v1", is_slippery=True)
    # Inicializa la tabla Q con ceros (estados x acciones)
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    
    epsilon = args.epsilon  # Tasa de exploración inicial
    successes = 0  # Contador de episodios exitosos
    rewards = []  # Lista para almacenar recompensas
    steps_in_success = []  # Lista para almacenar pasos en episodios exitosos

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
            steps += 1

            # Actualiza la Q-table usando la fórmula de Q-learning
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            q_table[state, action] = old_value + args.alpha * (reward + args.gamma * next_max - old_value)

            state = next_state  # Actualiza el estado
            total_reward += reward  # Acumula la recompensa

            if done:
                if reward > 0:  # Incrementa éxitos si se llega al objetivo
                    successes += 1
                    steps_in_success.append(steps)
                break

        rewards.append(total_reward)  # Guarda la recompensa del episodio
        # Reduce epsilon para favorecer explotación con el tiempo
        epsilon = max(args.epsilon_min, epsilon * args.epsilon_decay)

        # Imprime métricas cada 10,000 episodios
        if (episode + 1) % 10000 == 0:
            success_rate = successes / 10000 * 100
            avg_reward = np.mean(rewards[-10000:])
            avg_steps = np.mean(steps_in_success[-1000:]) if steps_in_success else 0
            print(f"Episodios {episode + 1 - 9999}-{episode + 1}: Tasa de éxito: {success_rate:.2f}%, "
                  f"Recompensa promedio: {avg_reward:.4f}, Pasos promedio en éxitos: {avg_steps:.2f}")
            successes = 0  # Reinicia el contador de éxitos

    env.close()  # Cierra el entorno
    np.save(args.save_path, q_table)  # Guarda la Q-table
    print(f"Q-table guardada en {args.save_path}")
    return q_table

def test_agent(q_table, args):
    # Crea el entorno con el modo de renderizado especificado
    env = gym.make("FrozenLake-v1", render_mode=args.render_mode, is_slippery=True)
    if args.render_mode == 'rgb_array':
        plt.ion()  # Activa el modo interactivo de Matplotlib
    
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
                else:
                    print("Fallo: Caíste en un agujero o se agotó el tiempo.")
                time.sleep(0.05)
                break
    env.close()  # Cierra el entorno
    if args.render_mode == 'rgb_array':
        plt.ioff()  # Desactiva el modo interactivo
        plt.close()

def main():
    args = parse_arguments()  # Obtiene los argumentos
    
    # Carga la Q-table si existe, o entrena una nueva
    if os.path.exists(args.save_path):
        print(f"Cargando Q-table existente desde {args.save_path}")
        q_table = np.load(args.save_path)
    else:
        print("Entrenando agente...")
        q_table = train_agent(args)
        print("Entrenamiento finalizado.")

    print("Probando agente con renderizado...")
    test_agent(q_table, args)

if __name__ == "__main__":
    main()