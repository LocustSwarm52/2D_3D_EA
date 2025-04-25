import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def f(x, y):
    return (
        np.exp(-((x + 3)**2 + (y - 3)**2)) +
        np.exp(-((x - 3)**2 + (y + 3)**2))
    )

pop_size        = 300
num_generations = 500
mutation_prob   = 0.2
mutation_scale  = 0.4
crossover_rate  = 0
accept_worse_prob = 0.05  # 5% chance to accept worse solutions
random_restart_prob = 0.01  # small chance to introduce diversity

population  = np.random.uniform(-4, 4, size=(pop_size, 2))
pop_history = [population.copy()]

for gen in range(num_generations):
    idxs = np.random.permutation(pop_size)
    new_pop = population.copy()
    for j in range(0, pop_size, 2):
        i1, i2 = idxs[j], idxs[j+1]
        p1, p2 = population[i1].copy(), population[i2].copy()

        # crossover
        if np.random.rand() < crossover_rate:
            α = np.random.rand()
            c1 = α * p1 + (1-α) * p2
            c2 = (1-α) * p1 + α * p2
        else:
            c1, c2 = p1.copy(), p2.copy()

        # mutation
        if np.random.rand() < mutation_prob:
            c1 += np.random.normal(scale=mutation_scale, size=2)
        if np.random.rand() < mutation_prob:
            c2 += np.random.normal(scale=mutation_scale, size=2)

        # random reinitialization
        if np.random.rand() < random_restart_prob:
            c1 = np.random.uniform(-4, 4, size=2)
        if np.random.rand() < random_restart_prob:
            c2 = np.random.uniform(-4, 4, size=2)

        c1 = np.clip(c1, -4, 4)
        c2 = np.clip(c2, -4, 4)

        # deterministic crowding pairing
        d11 = np.linalg.norm(c1 - p1) + np.linalg.norm(c2 - p2)
        d12 = np.linalg.norm(c1 - p2) + np.linalg.norm(c2 - p1)
        if d11 <= d12:
            pairs = [(i1, p1, c1), (i2, p2, c2)]
        else:
            pairs = [(i1, p1, c2), (i2, p2, c1)]

        # replacement with occasional acceptance of worse solutions
        for idx, parent, child in pairs:
            fp, fc = f(parent[0], parent[1]), f(child[0], child[1])
            if fc > fp or np.random.rand() < accept_worse_prob:
                new_pop[idx] = child

    population = new_pop
    pop_history.append(population.copy())

# Plotting 
xv = np.linspace(-4, 4, 200)
yv = np.linspace(-4, 4, 200)
X, Y = np.meshgrid(xv, yv)
Z = f(X, Y)

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection='3d')
surf = ax1.plot_surface(X, Y, Z, rstride=5, cstride=5, linewidth=0.1, alpha=0.8, cmap='plasma')
ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('f(x,y)')
ax1.set_title('3D Surface — EA with Escape Strategy')
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)

ax2 = fig.add_subplot(122)
cont = ax2.contourf(X, Y, Z, levels=50, cmap='plasma')
ax2.set_xlabel('X'); ax2.set_ylabel('Y')
ax2.set_title('2D Contour — EA with Escape Strategy')
fig.colorbar(cont, ax=ax2)

init = pop_history[0]
scat3d = ax1.scatter(init[:,0], init[:,1], f(init[:,0], init[:,1]), c='red', s=20)
scat2d = ax2.scatter(init[:,0], init[:,1], c='red', s=20)

def update(frame):
    pop = pop_history[frame]
    x, y = pop[:,0], pop[:,1]
    z = f(x, y)
    scat3d._offsets3d = (x, y, z)
    scat2d.set_offsets(np.c_[x, y])
    ax1.set_title(f'3D Surface — Generation {frame}')
    ax2.set_title(f'2D Contour — Generation {frame}')
    return scat3d, scat2d

ani = FuncAnimation(fig, update, frames=len(pop_history), interval=50, blit=False)
plt.tight_layout()
plt.show()
