import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load data
data = pd.read_csv('data.csv')

# Prepare English and Tagalog sentences
eng_sentences = data['english'].values
tag_sentences = data['tagalog'].values

# Tokenize English sentences
eng_tokenizer = Tokenizer(filters='')
eng_tokenizer.fit_on_texts(eng_sentences)
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_max_length = max([len(sentence.split()) for sentence in eng_sentences])

# Tokenize Tagalog sentences
tag_tokenizer = Tokenizer(filters='')
tag_tokenizer.fit_on_texts(tag_sentences)
tag_vocab_size = len(tag_tokenizer.word_index) + 1
tag_max_length = max([len(sentence.split()) for sentence in tag_sentences])

# Convert sentences to sequences
eng_sequences = eng_tokenizer.texts_to_sequences(eng_sentences)
tag_sequences = tag_tokenizer.texts_to_sequences(tag_sentences)

# Pad sequences
eng_input_data = pad_sequences(eng_sequences, maxlen=eng_max_length, padding='post')
tag_input_data = pad_sequences(tag_sequences, maxlen=tag_max_length, padding='post')

# Prepare Tagalog output (for decoder)
tag_output_data = np.zeros((tag_input_data.shape[0], tag_max_length, tag_vocab_size))
for i, seq in enumerate(tag_sequences):
    for t, word_id in enumerate(seq):
        tag_output_data[i, t, word_id] = 1



# Build Seq2Seq model
def build_seq2seq_model(eng_vocab_size, tag_vocab_size, eng_max_length, tag_max_length):
    # Encoder
    encoder_inputs = tf.keras.layers.Input(shape=(eng_max_length,))
    encoder_embedding = tf.keras.layers.Embedding(eng_vocab_size, 256)(encoder_inputs)
    encoder_gru = tf.keras.layers.GRU(256, return_state=True)
    encoder_output, encoder_state = encoder_gru(encoder_embedding)
    
    # Decoder
    decoder_inputs = tf.keras.layers.Input(shape=(tag_max_length,))
    decoder_embedding = tf.keras.layers.Embedding(tag_vocab_size, 256)(decoder_inputs)
    decoder_gru = tf.keras.layers.GRU(256, return_sequences=True)(decoder_embedding, initial_state=encoder_state)
    decoder_dense = tf.keras.layers.Dense(tag_vocab_size, activation='softmax')(decoder_gru)
    
    # Model
    model = tf.keras.models.Model([encoder_inputs, decoder_inputs], decoder_dense)
    return model

# Create model
model = build_seq2seq_model(eng_vocab_size, tag_vocab_size, eng_max_length, tag_max_length)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# Train the model
model.fit([eng_input_data, tag_input_data], tag_output_data, epochs=100, batch_size=32)


def translate_sentence(input_sentence):
    # Tokenize and pad the input sentence
    input_sequence = eng_tokenizer.texts_to_sequences([input_sentence])
    input_padded = pad_sequences(input_sequence, maxlen=eng_max_length, padding='post')
    
    # Predict the translation
    prediction = model.predict([input_padded, np.zeros((1, tag_max_length))])
    
    # Convert prediction to words
    predicted_sequence = np.argmax(prediction, axis=-1)
    predicted_sentence = ' '.join([tag_tokenizer.index_word[i] for i in predicted_sequence[0] if i != 0])
    
    return predicted_sentence

# Example translation
english_sentence = input("ENGLISH: ")
tagalog_translation = translate_sentence(english_sentence)
print(f"English: {english_sentence}\nTagalog: {tagalog_translation}")

