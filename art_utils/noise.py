import tensorflow as tf


def map_gradients(gradient_map, gis, length):
    index_tensor = tf.reshape(tf.concat([
        tf.reshape(tf.tile(tf.expand_dims(gis, 1), [1, 3]), [length * 3, 1]),
        tf.expand_dims(tf.tile(tf.range(0, limit=3), [length]), 1)
    ], 1), [length, 3, 2])
    return tf.gather_nd(gradient_map, index_tensor)


def tf_repeat(x, num_repeats):
    x = tf.reshape(x, [-1, 1])
    x = tf.tile(x, [1, num_repeats])
    return tf.reshape(x, [-1])


def get_input_vectors(shape, phases, scaling, offset):
    x = tf.reshape(tf_repeat(offset[0] + tf.linspace(0.0, float(shape[0]) - 1., shape[0]) / float(scaling),
                             shape[1] * phases),
                   [shape[0], shape[1], phases]) * tf.pow(2.0, tf.linspace(0.0, float(phases) - 1., phases))
    y = tf.reshape(tf_repeat(tf.tile(
        offset[1] + tf.linspace(0.0, float(shape[1]) - 1., shape[1]) / scaling,
        [shape[0]]
    ), phases), [shape[0], shape[1], phases]) * tf.pow(2.0, tf.linspace(0.0, float(phases) - 1., phases))
    z = tf.reshape(
        tf.tile(offset[2] + 10. * tf.linspace(0.0, float(phases) - 1., phases), [shape[0] * shape[1]]),
        [shape[0], shape[1], phases, 1])
    x = tf.reshape(x, [shape[0], shape[1], phases, 1])
    y = tf.reshape(y, [shape[0], shape[1], phases, 1])

    return tf.reshape(tf.concat([x, y, z], 3), [shape[0] * shape[1] * phases, 3])


def get_simplex_vertices(offsets, vertex_table, length):
    vertex_table_x_index = tf.cast((offsets[:, 0] >= offsets[:, 1]), tf.int32)
    vertex_table_y_index = tf.cast((offsets[:, 1] >= offsets[:, 2]), tf.int32)
    vertex_table_z_index = tf.cast((offsets[:, 0] >= offsets[:, 2]), tf.int32)

    index_list = tf.concat([
        tf.reshape(tf.tile(tf.concat([
            tf.expand_dims(vertex_table_x_index, 1),
            tf.expand_dims(vertex_table_y_index, 1),
            tf.expand_dims(vertex_table_z_index, 1),
        ], 1), [1, 6]), [6 * length, 3]),
        tf.expand_dims(tf.tile(tf.range(0, limit=6), [length]), 1)], 1)
    vertices = tf.reshape(tf.gather_nd(vertex_table, index_list), [-1, 2, 3])
    return vertices


def calculate_gradient_contribution(offsets, gis, gradient_map, length):
    t = 0.5 - offsets[:, 0] ** 2. - offsets[:, 1] ** 2. - offsets[:, 2] ** 2.
    mapped_gis = map_gradients(gradient_map, gis, length)
    dot_products = tf.reduce_sum(mapped_gis * offsets, 1)
    return tf.cast(tf.math.greater_equal(t, 0.), tf.float32) * t ** 4. * dot_products


def noise3d(input_vectors, perm, grad3, vertex_table, length):
    skew_factors = (input_vectors[:, 0] + input_vectors[:, 1] + input_vectors[:, 2]) * 1.0 / 3.0
    skewed_vectors = tf.floor(input_vectors + tf.expand_dims(skew_factors, 1))
    unskew_factors = (skewed_vectors[:, 0] + skewed_vectors[:, 1] + skewed_vectors[:, 2]) * 1.0 / 6.0
    offsets_0 = input_vectors - (skewed_vectors - tf.expand_dims(unskew_factors, 1))
    simplex_vertices = get_simplex_vertices(offsets_0, vertex_table, length)  # divided it by 2, doesn't error now
    offsets_1 = offsets_0 - simplex_vertices[:, 0, :] + 1.0 / 6.0
    offsets_2 = offsets_0 - simplex_vertices[:, 1, :] + 1.0 / 3.0
    offsets_3 = offsets_0 - 0.5
    masked_skewed_vectors = tf.cast(skewed_vectors, tf.int32) % 256
    gi0s = tf.gather_nd(
        perm,
        tf.expand_dims(masked_skewed_vectors[:, 0], 1) +
        tf.expand_dims(tf.gather_nd(
            perm,
            tf.expand_dims(masked_skewed_vectors[:, 1], 1) +
            tf.expand_dims(tf.gather_nd(
                perm,
                tf.expand_dims(masked_skewed_vectors[:, 2], 1)), 1)), 1)
    ) % 12
    gi1s = tf.gather_nd(
        perm,
        tf.expand_dims(masked_skewed_vectors[:, 0], 1) +
        tf.expand_dims(tf.cast(simplex_vertices[:, 0, 0], tf.int32), 1) +
        tf.expand_dims(tf.gather_nd(
            perm,
            tf.expand_dims(masked_skewed_vectors[:, 1], 1) +
            tf.expand_dims(tf.cast(simplex_vertices[:, 0, 1], tf.int32), 1) +
            tf.expand_dims(tf.gather_nd(
                perm,
                tf.expand_dims(masked_skewed_vectors[:, 2], 1) +
                tf.expand_dims(tf.cast(simplex_vertices[:, 0, 2], tf.int32), 1)), 1)), 1)
    ) % 12
    gi2s = tf.gather_nd(
        perm,
        tf.expand_dims(masked_skewed_vectors[:, 0], 1) +
        tf.expand_dims(tf.cast(simplex_vertices[:, 1, 0], tf.int32), 1) +
        tf.expand_dims(tf.gather_nd(
            perm,
            tf.expand_dims(masked_skewed_vectors[:, 1], 1) +
            tf.expand_dims(tf.cast(simplex_vertices[:, 1, 1], tf.int32), 1) +
            tf.expand_dims(tf.gather_nd(
                perm,
                tf.expand_dims(masked_skewed_vectors[:, 2], 1) +
                tf.expand_dims(tf.cast(simplex_vertices[:, 1, 2], tf.int32), 1)), 1)), 1)
    ) % 12
    gi3s = tf.gather_nd(
        perm,
        tf.expand_dims(masked_skewed_vectors[:, 0], 1) +
        1 +
        tf.expand_dims(tf.gather_nd(
            perm,
            tf.expand_dims(masked_skewed_vectors[:, 1], 1) +
            1 +
            tf.expand_dims(tf.gather_nd(
                perm,
                tf.expand_dims(masked_skewed_vectors[:, 2], 1) +
                1), 1)), 1)
    ) % 12
    n0s = calculate_gradient_contribution(offsets_0, gi0s, grad3, length)
    n1s = calculate_gradient_contribution(offsets_1, gi1s, grad3, length)
    n2s = calculate_gradient_contribution(offsets_2, gi2s, grad3, length)
    n3s = calculate_gradient_contribution(offsets_3, gi3s, grad3, length)
    return 23.0 * tf.squeeze(
        tf.expand_dims(n0s, 1) + tf.expand_dims(n1s, 1) + tf.expand_dims(n2s, 1) + tf.expand_dims(n3s, 1))


def calculate_image(noise_values, phases, shape):
    val = tf.floor((tf.add_n(tf.split(
        tf.reshape(noise_values, [shape[0], shape[1], phases]) / tf.pow(
            2.0,
            tf.linspace(0.0, float(phases) - 1., phases)), phases, 2)) + 1.0) * 128.)
    return tf.concat([val, val, val], 2)
