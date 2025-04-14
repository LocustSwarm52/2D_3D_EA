import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Target function with two equal peaks
def f(x, y):
    return (
        np.exp(-((x + 3)**2 + (y - 3)**2))  # Peak at (-3,  3)
        + np.exp(-((x - 3)**2 + (y + 3)**2))  # Peak at ( 3, -3)
    )

# EA parameters
pop_size        = 300
num_generations = 50
mutation_rate   = 0.01
crossover_rate  = 0.7

# Initialize population and history
population  = np.random.uniform(-4, 4, size=(pop_size, 2))
pop_history = [population.copy()]

# Deterministic crowding reproduction + replacement
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
        if np.random.rand() < mutation_rate:
            c1 += np.random.normal(scale=0.5, size=2)
        if np.random.rand() < mutation_rate:
            c2 += np.random.normal(scale=0.5, size=2)
        c1 = np.clip(c1, -4, 4)
        c2 = np.clip(c2, -4, 4)

        # deterministic crowding pairing
        d11 = np.linalg.norm(c1 - p1) + np.linalg.norm(c2 - p2)
        d12 = np.linalg.norm(c1 - p2) + np.linalg.norm(c2 - p1)
        if d11 <= d12:
            pairs = [(i1, p1, c1), (i2, p2, c2)]
        else:
            pairs = [(i1, p1, c2), (i2, p2, c1)]

        # replacement
        for idx, parent, child in pairs:
            if f(child[0], child[1]) > f(parent[0], parent[1]):
                new_pop[idx] = child

    population = new_pop
    pop_history.append(population.copy())

# Prepare mesh for plotting
xv = np.linspace(-4, 4, 200)
yv = np.linspace(-4, 4, 200)
X, Y = np.meshgrid(xv, yv)
Z    = f(X, Y)

fig = plt.figure(figsize=(14, 6))

# 3D surface with plasma colormap
ax1 = fig.add_subplot(121, projection='3d')
surf = ax1.plot_surface(
    X, Y, Z,
    rstride=5, cstride=5,
    linewidth=0.1, alpha=0.8,
    cmap='plasma'     # purple→yellow gradient
)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('f(x,y)')
ax1.set_title('3D Surface — EA with Deterministic Crowding')
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)

# 2D contour with plasma colormap
ax2 = fig.add_subplot(122)
cont = ax2.contourf(
    X, Y, Z,
    levels=50,
    cmap='plasma'     # match the 3D plot
)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('2D Contour — EA with Deterministic Crowding')
fig.colorbar(cont, ax=ax2)

# Initial scatter (all dots in red)
init = pop_history[0]
scat3d = ax1.scatter(
    init[:,0], init[:,1], f(init[:,0], init[:,1]),
    c='red', s=20
)
scat2d = ax2.scatter(
    init[:,0], init[:,1],
    c='red', s=20
)

# Animation update (400 ms/frame)
def update(frame):
    pop = pop_history[frame]
    x, y = pop[:,0], pop[:,1]
    z    = f(x, y)
    scat3d._offsets3d = (x, y, z)
    scat2d.set_offsets(np.c_[x, y])
    ax1.set_title(f'3D Surface — Generation {frame}')
    ax2.set_title(f'2D Contour — Generation {frame}')
    return scat3d, scat2d

ani = FuncAnimation(
    fig, update,
    frames=len(pop_history),
    interval=400,   # slowed by half
    blit=False
)

plt.tight_layout()
plt.show()
