import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from utils import decode_seq

def encryption(source_code, iterations_to_crack = 2000, randomnessIndex = 10, lossThreshold = 0.3):

    fullcharacterset = r"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890.,;:/\?!"
    source_sentences = []
    target_sentences = []
    source_chars = set()
    target_chars = set()
    nb_samples = 1

    source_line = str(source_code).split('\t')[0]
    target_line = '\t' + str(fullcharacterset) + '\n'
    source_sentences.append(source_line)
    target_sentences.append(target_line)
    for ch in source_line:
        if (ch not in source_chars):
            source_chars.add(ch)
    for ch in target_line:
        if (ch not in target_chars):
            target_chars.add(ch)
    target_chars = sorted(list(target_chars))
    source_chars = sorted(list(source_chars))
    source_index_to_char_dict = {}
    source_char_to_index_dict = {}
    for k, v in enumerate(source_chars):
        source_index_to_char_dict[k] = v
        source_char_to_index_dict[v] = k
    target_index_to_char_dict = {}
    target_char_to_index_dict = {}
    for k, v in enumerate(target_chars):
        target_index_to_char_dict[k] = v
        target_char_to_index_dict[v] = k
    source_sent = source_sentences
    target_sent = target_sentences
    max_len_source_sent = max([len(line) for line in source_sent])
    max_len_target_sent = max([len(line) for line in target_sent])

    tokenized_source_sentences = np.zeros(shape = (nb_samples,max_len_source_sent,len(source_chars)), dtype='float32')
    tokenized_target_sentences = np.zeros(shape = (nb_samples,max_len_target_sent,len(target_chars)), dtype='float32')
    target_data = np.zeros((nb_samples, max_len_target_sent, len(target_chars)),dtype='float32')
    for i in range(nb_samples):
        for k,ch in enumerate(source_sent[i]):
            tokenized_source_sentences[i,k,source_char_to_index_dict[ch]] = 1
        for k,ch in enumerate(target_sent[i]):
            tokenized_target_sentences[i,k,target_char_to_index_dict[ch]] = 1
            # decoder_target_data will be ahead by one timestep and will not include the start character.
            if k > 0:
                target_data[i,k-1,target_char_to_index_dict[ch]] = 1

    # Encoder model
    encoder_input = Input(shape=(None,len(source_chars)))
    encoder_LSTM = LSTM(256,return_state = True)
    encoder_outputs, encoder_h, encoder_c = encoder_LSTM (encoder_input)
    encoder_states = [encoder_h, encoder_c]
    # Decoder model
    decoder_input = Input(shape=(None,len(target_chars)))
    decoder_LSTM = LSTM(256,return_sequences=True, return_state = True)
    decoder_out, _ , _ = decoder_LSTM(decoder_input, initial_state=encoder_states)
    decoder_dense = Dense(len(target_chars),activation='softmax')
    decoder_out = decoder_dense (decoder_out)
    model = Model(inputs=[encoder_input, decoder_input],outputs=[decoder_out])

    # create weights with the right shape, sample:
    # nested randomness creation
    weights = [ w * np.random.rand(*w.shape) for w in model.get_weights()]
    for i in range(randomnessIndex):
        weights = [ w * np.random.rand(*w.shape) for w in weights]
    # update
    model.set_weights(weights)

    # Inference models for testing
    # Encoder inference model
    encoder_model_inf = Model(encoder_input, encoder_states)
    # Decoder inference model
    decoder_state_input_h = Input(shape=(256,))
    decoder_state_input_c = Input(shape=(256,))
    decoder_input_states = [decoder_state_input_h, decoder_state_input_c]
    decoder_out, decoder_h, decoder_c = decoder_LSTM(decoder_input,
                                                     initial_state=decoder_input_states)
    decoder_states = [decoder_h , decoder_c]
    decoder_out = decoder_dense(decoder_out)
    decoder_model_inf = Model(inputs=[decoder_input] + decoder_input_states,
                              outputs=[decoder_out] + decoder_states )

    for seq_index in range(1):
        inp_seq = tokenized_source_sentences[seq_index:seq_index+1]
        obfuscated_code = decode_seq(inp_seq, encoder_model_inf, decoder_model_inf, target_chars, target_index_to_char_dict, target_char_to_index_dict, max_len_target_sent)
        print('-')
        print('Input sentence:', source_sent[seq_index])
        print('Decoded sentence:', obfuscated_code)

    return obfuscated_code
