import numpy as np
from signjoey.vocabulary import BOS_ID, EOS_ID, PAD_ID, EOS_TOKEN

def array_to_sentence(vocab_dict, array: np.array, cut_at_eos=True):
    """
    Converts an array of IDs to a sentence, optionally cutting the result
    off at the end-of-sequence token.

    :param array: 1D array containing indices
    :param cut_at_eos: cut the decoded sentences at the first <eos>
    :return: list of strings (tokens)
    """
    sentence = []
    for i in array:
        s = vocab_dict[i]
        if cut_at_eos and s == EOS_TOKEN:
            break
        sentence.append(s)
    return sentence

def arrays_to_sentences(vocab_dict, arrays: np.array, cut_at_eos=True):
    """
    Convert multiple arrays containing sequences of token IDs to their
    sentences, optionally cutting them off at the end-of-sequence token.

    :param arrays: 2D array containing indices
    :param cut_at_eos: cut the decoded sentences at the first <eos>
    :return: list of list of strings (tokens)
    """
    sentences = []
    for array in arrays:
        sentences.append(array_to_sentence(vocab_dict=vocab_dict, array=array, cut_at_eos=cut_at_eos))
    return sentences