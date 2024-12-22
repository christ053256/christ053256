import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# Parameters
batch_size = 32
epochs = 50
latent_dim = 128
embedding_dim = 100
max_words = 10000
temperature = 0.1  # Control randomness

# Storage for the dataset
input_texts = []
target_texts = []

# Read the dataset
dataset_path = "./data/cleaned_tgl.txt"
with open(dataset_path, "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split("|||")
        if len(parts) != 2:
            continue
        input_text, target_text = parts
        input_text = input_text.strip()
        target_text = target_text.strip()
        target_text = "START " + target_text + " END"  # Add start and end tokens
        input_texts.append(input_text)
        target_texts.append(target_text)

print(f"Loaded {len(input_texts)} valid translation pairs.")

# Create and fit tokenizers
input_tokenizer = Tokenizer(num_words=max_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
input_tokenizer.fit_on_texts(input_texts)
target_tokenizer = Tokenizer(num_words=max_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
target_tokenizer.fit_on_texts(target_texts)

# Convert texts to sequences
encoder_input_sequences = input_tokenizer.texts_to_sequences(input_texts)
decoder_input_sequences = target_tokenizer.texts_to_sequences(target_texts)

# Get sequence lengths
max_encoder_seq_length = max(len(seq) for seq in encoder_input_sequences)
max_decoder_seq_length = max(len(seq) for seq in decoder_input_sequences)

# Pad sequences
encoder_input_data = pad_sequences(encoder_input_sequences, maxlen=max_encoder_seq_length, padding='post')
decoder_input_data = pad_sequences(decoder_input_sequences, maxlen=max_decoder_seq_length, padding='post')

# Create target data
decoder_target_data = []
for seq in decoder_input_sequences:
    decoder_target_data.append(seq[1:] + [0])  # Shift sequence by 1 and add padding
decoder_target_data = pad_sequences(decoder_target_data, maxlen=max_decoder_seq_length, padding='post')

# Vocabulary sizes
num_encoder_tokens = min(len(input_tokenizer.word_index) + 1, max_words)
num_decoder_tokens = min(len(target_tokenizer.word_index) + 1, max_words)

print(f"Number of samples: {len(input_texts)}")
print(f"Number of unique input tokens: {num_encoder_tokens}")
print(f"Number of unique output tokens: {num_decoder_tokens}")
print(f"Max sequence length for inputs: {max_encoder_seq_length}")
print(f"Max sequence length for outputs: {max_decoder_seq_length}")

# Build the model
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(num_encoder_tokens, embedding_dim)
encoder_embedded = encoder_embedding(encoder_inputs)
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_embedded)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(num_decoder_tokens, embedding_dim)
decoder_embedded = decoder_embedding(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedded, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Custom loss function to handle padding
def masked_loss(y_true, y_pred):
    mask = tf.math.not_equal(y_true, 0)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    return tf.reduce_mean(loss)

# Compile and train the model
model.compile(optimizer="rmsprop", loss=masked_loss)
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
)

# Build inference models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_embedded = decoder_embedding(decoder_inputs)
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_embedded, initial_state=decoder_states_inputs
)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)

# Translation function with temperature
def decode_sequence(input_seq, temperature=0.5):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.array([[target_tokenizer.word_index.get('START', 1)]])  # Start token index
    decoded_sentence = []
    
    while True:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        
        output_tokens = output_tokens[0, -1, :]
        output_tokens = np.log(output_tokens + 1e-7) / temperature  # Apply temperature scaling
        output_tokens = np.exp(output_tokens) / np.sum(np.exp(output_tokens))  # Normalize
        sampled_token_index = np.random.choice(len(output_tokens), p=output_tokens)
        
        sampled_word = None
        for word, index in target_tokenizer.word_index.items():
            if index == sampled_token_index:
                sampled_word = word
                break
        
        if sampled_word is None or sampled_word == 'END' or len(decoded_sentence) > max_decoder_seq_length:
            break
            
        decoded_sentence.append(sampled_word)
        target_seq = np.array([[sampled_token_index]])
        states_value = [h, c]
    
    return ' '.join(decoded_sentence)

# Real-time translation
while True:
    user_input = input("Enter English text to translate (or 'exit' to quit): ").strip()
    if user_input.lower() == 'exit':
        break
        
    input_seq = input_tokenizer.texts_to_sequences([user_input])
    input_seq = pad_sequences(input_seq, maxlen=max_encoder_seq_length, padding='post')
    
    translated_text = decode_sequence(input_seq, temperature=temperature)
    print(f"Translated to Filipino: {translated_text}")
