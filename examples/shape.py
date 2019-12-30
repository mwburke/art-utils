import matplotlib.pyplot as plt

from art_utils.shape import chiakins_curve


points = [[0.05, 20], [0.25 ,360], [0.75, 460], [0.95, 20]]

iterations = 1
updated_points = chiakins_curve(points, iterations)
plt.plot(updated_points[:, 0], updated_points[:, 1])
plt.title('1 iterations')
plt.show()

iterations = 3
updated_points = chiakins_curve(points, iterations)
plt.plot(updated_points[:, 0], updated_points[:, 1])
plt.title('3 iterations')
plt.show()


iterations = 5
updated_points = chiakins_curve(points, iterations)
plt.plot(updated_points[:, 0], updated_points[:, 1])
plt.title('5 iterations')
plt.show()
