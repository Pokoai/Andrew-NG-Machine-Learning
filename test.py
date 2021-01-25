
x1, x2 = find_decision_boundary(svc, 0, 5, 1.5, 5, 2 * 10**-3)

fig. ax = plt.subplots(figsize=(10, 8))
ax.scatter(x1, x2, s=10, c='r', label='Boundary')

plot_init_pic(data, fig, ax)

ax.set_title('SVM(C=1) Decition Boundary')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.legend()

plt.show()