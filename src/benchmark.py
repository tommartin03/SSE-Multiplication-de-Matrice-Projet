import subprocess
import time
import matplotlib.pyplot as plt

# Fonction pour exécuter le benchmark avec différentes méthodes de multiplication
def run_benchmark(dimensions, repetitions):
    results_naive = []
    results_moins_naive = []
    results_parallel = []

    for dim in dimensions:
        for _ in range(repetitions):
            # Naive multiplication
            start_time = time.time()
            process = subprocess.Popen(
                ["./benchmark_program", str(dim), "naive"],  # Passer "naive" comme argument
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            elapsed_time = (time.time() - start_time) * 1000  # Temps en ms
            results_naive.append({"dimension": dim, "time": elapsed_time})

            # Moins naive multiplication
            start_time = time.time()
            process = subprocess.Popen(
                ["./benchmark", str(dim), "moins_naive"],  # Passer "moins_naive"
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            elapsed_time = (time.time() - start_time) * 1000
            results_moins_naive.append({"dimension": dim, "time": elapsed_time})

            # Parallel multiplication
            start_time = time.time()
            process = subprocess.Popen(
                ["./benchmark_program", str(dim), "parallel"],  # Passer "parallel"
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            elapsed_time = (time.time() - start_time) * 1000
            results_parallel.append({"dimension": dim, "time": elapsed_time})

    return results_naive, results_moins_naive, results_parallel

# Fonction pour tracer les courbes des trois méthodes de multiplication
def plot_results(results_naive, results_moins_naive, results_parallel):
    plt.figure(figsize=(10, 6))

    # Extraire les dimensions uniques
    dimensions = sorted(set(r["dimension"] for r in results_naive))

    # Calculer le temps moyen pour chaque méthode et chaque dimension
    avg_times_naive = []
    avg_times_moins_naive = []
    avg_times_parallel = []

    for dim in dimensions:
        # Calculer la moyenne des temps pour la méthode Naive
        times_naive = [r["time"] for r in results_naive if r["dimension"] == dim]
        avg_times_naive.append(sum(times_naive) / len(times_naive))

        # Calculer la moyenne des temps pour la méthode Moins Naive
        times_moins_naive = [r["time"] for r in results_moins_naive if r["dimension"] == dim]
        avg_times_moins_naive.append(sum(times_moins_naive) / len(times_moins_naive))

        # Calculer la moyenne des temps pour la méthode Parallel
        times_parallel = [r["time"] for r in results_parallel if r["dimension"] == dim]
        avg_times_parallel.append(sum(times_parallel) / len(times_parallel))

    # Tracer les courbes
    plt.plot(dimensions, avg_times_naive, marker='o', label="Naive Multiplication", color='blue')
    plt.plot(dimensions, avg_times_moins_naive, marker='x', label="Moins Naive Multiplication", color='green')
    plt.plot(dimensions, avg_times_parallel, marker='s', label="Parallel Multiplication", color='red')

    # Ajouter des labels et un titre
    plt.title("Benchmark des Multiplications de Matrices avec différentes méthodes")
    plt.xlabel("Dimension de la Matrice")
    plt.ylabel("Temps d'Exécution Moyen (ms)")
    plt.legend()
    plt.grid(True)
    plt.savefig("benchmark_sse_three_methods.png")
    plt.show()

# Liste des dimensions de matrices à tester
dimensions = [10, 20, 30, 40, 50]
repetitions = 10  # Nombre de répétitions pour chaque dimension

# Exécuter le benchmark et récupérer les résultats pour chaque méthode
results_naive, results_moins_naive, results_parallel = run_benchmark(dimensions, repetitions)

# Tracer les résultats
plot_results(results_naive, results_moins_naive, results_parallel)
