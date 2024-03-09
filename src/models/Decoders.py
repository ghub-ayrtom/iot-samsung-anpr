import numpy as np


# Интерфейс для декодирования последовательности
class Decoder:
    def decode(self, predicted_sequence, chars_list):
        raise NotImplementedError


class BeamDecoder(Decoder):
    def decode(self, predicted_sequence, chars_list):
        labels = []
        final_probabilities = []  # final_prob (old name)
        final_labels = []

        k = 1

        for i in range(predicted_sequence.shape[0]):
            sequences = [[list(), 0.0]]
            all_sequences = []

            single_prediction = predicted_sequence[i, :, :]

            for j in range(single_prediction.shape[1]):
                single_sequence = []

                for char in single_prediction[:, j]:
                    single_sequence.append(char)
                all_sequences.append(single_sequence)

            for row in all_sequences:
                all_candidates = []

                for i in range(len(sequences)):
                    sequence, score = sequences[i]

                    for j in range(len(row)):
                        candidate = [sequence + [j], score - row[j]]
                        all_candidates.append(candidate)

                ordered = sorted(all_candidates, key=lambda tup: tup[1])
                sequences = ordered[:k]  # Выбор k лучших

            full_predicted_labels = []
            probabilities = []  # probs (old name)

            for i in sequences:
                predicted_labels = i[0]
                without_repeating = []
                current_char = predicted_labels[0]

                if current_char != len(chars_list) - 1:
                    without_repeating.append(current_char)

                for c in predicted_labels:
                    if (current_char == c) or (c == len(chars_list) - 1):
                        if c == len(chars_list) - 1:
                            current_char = c

                        continue

                    without_repeating.append(c)
                    current_char = c

                full_predicted_labels.append(without_repeating)
                probabilities.append(i[1])

            for i, label in enumerate(full_predicted_labels):
                decoded_label = ''

                for j in label:
                    decoded_label += chars_list[j]
                labels.append(decoded_label)

                final_probabilities.append(probabilities[i])
                final_labels.append(full_predicted_labels[i])

        return labels, final_probabilities, final_labels


class GreedyDecoder(Decoder):
    def decode(self, predicted_sequence, chars_list):
        labels = []
        full_predicted_labels = []

        # predicted_sequence.shape = [batch, len(chars_list), len_seq]
        for i in range(predicted_sequence.shape[0]):
            single_prediction = predicted_sequence[i, :, :]
            predicted_labels = []

            for j in range(single_prediction.shape[1]):
                predicted_labels.append(np.argmax(single_prediction[:, j], axis=0))

            without_repeating = []
            current_char = predicted_labels[0]

            if current_char != len(chars_list) - 1:
                without_repeating.append(current_char)

            for c in predicted_labels:
                if (current_char == c) or (c == len(chars_list) - 1):
                    if c == len(chars_list) - 1:
                        current_char = c

                    continue

                without_repeating.append(c)
                current_char = c

            full_predicted_labels.append(without_repeating)

        for i, label in enumerate(full_predicted_labels):
            decoded_label = ''

            for j in label:
                decoded_label += chars_list[j]
            labels.append(decoded_label)

        return labels, full_predicted_labels


def decode_function(predicted_sequence, chars_list, decoder=GreedyDecoder):
    return decoder().decode(predicted_sequence, chars_list)
