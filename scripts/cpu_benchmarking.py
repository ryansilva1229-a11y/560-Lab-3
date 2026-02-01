import subprocess
import re
import matplotlib.pyplot as plt

n_sizes = [512, 1024, 2048]
times = []

for n in n_sizes:
    result = subprocess.run(
        ["./matrix_cpu", str(n)],
        capture_output=True,
        text=True,
        check=True
    )

    output = result.stdout.strip()
    match = re.search(r"([\d.]+) seconds", output)

    times.append(float(match.group(1)))

plt.plot(n_sizes, times, marker='o')
plt.xlabel('Matrix Size (N x N)')
plt.ylabel('Time (seconds)')
plt.title('CPU Matrix Multiplication Benchmark')
plt.grid(True)
plt.show()