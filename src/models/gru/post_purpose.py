import tensorflow as tf

class PostPurposeGRU:
    def __init__(self, vocab_length, embedding_matrix, gru_size=256, hidden_size=128, num_hidden_layers=0):
        model = tf.keras.models.Sequential()
        model.add(
            tf.keras.layers.Embedding(vocab_length, embedding_matrix.shape[1],
                                      input_length=embedding_matrix.shape[0],
                                      embeddings_initializer=tf.keras.initializers.Constant(
                                          embedding_matrix),
                                      trainable=True, mask_zero=True))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_size)))

        for _ in range(num_hidden_layers):
            model.add(tf.keras.layers.Dense(hidden_size, activation='relu'))

        model.add(tf.keras.layers.Dense(3, activation='softmax'))  # 3 classes for Analysis, Evaluation, Synthesis

        self.model = model

    def get_model(self):
        return self.model
