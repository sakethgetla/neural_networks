import tensorflow as tf


fashion_mnist = tf.keras.datasets.faction_mnist

(train_images, train_lables), (test_images, test_lables) = faction_mnist.load_data()

model = tf.keras.sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_catago


