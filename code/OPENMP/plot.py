import numpy as np
import matplotlib.pyplot as plt

threads = np.array([1, 2, 4, 6, 8, 10, 12, 16, 20, 32, 64])
times = np.array([20.00, 12.50, 8.00, 6.50, 5.80, 5.40, 5.10, 4.80, 4.60, 4.40, 4.30])

speedup = times[0] / times

P_values = (speedup - 1) / (speedup * (1 - 1 / threads))

plt.figure(figsize=(8, 5))
plt.plot(threads, speedup, marker='o', linestyle='-', color='b', label='Measured Speedup')
plt.plot(threads, threads, linestyle='--', color='gray', label='Ideal Speedup (Linear)')

plt.xlabel("Number of Threads")
plt.ylabel("Speedup")
plt.title("Speedup vs. Number of Threads")
plt.legend()
plt.grid(True)
plt.xscale('log', base=2)

P_avg = np.mean(P_values[1:])
print(f"Estimated Parallelization Fraction (P): {P_avg:.4f}")

plt.show()
