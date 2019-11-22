import tensorflow as tf


def kmeans(data, k, n_iter=10, verbose=0):
    dims = data.shape[1]

    maxs = tf.math.reduce_max(data, axis=0)
    mins = tf.math.reduce_min(data, axis=0)

    data = (data - mins) / (maxs - mins)

    centroids = tf.random.uniform(shape=[k, dims], minval=0, maxval=1)

    for i in range(n_iter):
        distances = tf.stack([
            tf.norm(data - centroids[j, :], axis=1) for j in range(k)
        ], 1)

        assignments = tf.argmin(distances, axis=1)

        new_centroids = []

        for j in range(k):
            if tf.reduce_sum(data[assignments == j]) != 0:
                new_centroids.append(tf.reduce_mean(data[assignments == j], axis=0))
            else:
                new_centroids.append(centroids[j, :])

        centroids = tf.stack(new_centroids)

        if verbose != 0:
            if ((i + 1) % 10) == 0:
                print(f'Iteration: {i + 1}')

    centroids = centroids * (maxs - mins) + mins

    return assignments, centroids
