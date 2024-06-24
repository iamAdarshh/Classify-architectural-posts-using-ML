import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dense
from tensorflow.keras.initializers import Constant
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable(package='Custom', name='PostPurposeGRU')
class PostPurposeGRU:
    def __init__(self, vocab_size, embedding_matrix, gru_size=256, hidden_size=128, num_hidden_layers=0):
        model = tf.keras.models.Sequential()
        model.add(
            Embedding(vocab_size, embedding_matrix.shape[1],
                      embeddings_initializer=Constant(embedding_matrix),
                      trainable=True, mask_zero=True)
        )
        model.add(Bidirectional(GRU(gru_size)))

        for _ in range(num_hidden_layers):
            model.add(Dense(hidden_size, activation='relu'))

        # 3 classes: Analysis, Evaluation, Synthesis
        model.add(Dense(3, activation='softmax'))

        self.model = model

    def get_model(self):
        return self.model
