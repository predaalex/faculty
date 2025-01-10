import nltk
from spellchecker import SpellChecker
import re
from Levenshtein import distance as levenshtein_distance
from nltk.tokenize import word_tokenize

spell = SpellChecker()
vocabulary = ['example', 'list', 'of', 'words', 'in', 'your', 'vocabulary']
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

def text_preparation_with_spell_correction(sentence):
    # 1. Lowercase everything
    sentence = sentence.lower()

    # 2. Remove all symbols other than a-z.
    pattern = re.compile(r"[^a-z ]")
    sentence = re.sub(pattern, " ", sentence)

    # 3. Correct spelling
    words = sentence.split()
    corrected_words = []
    for word in words:
        corrected_word = spell.correction(word)
        if corrected_word is None:
            corrected_word = '<unk>'  # Use '<unk>' if no correction found
        corrected_words.append(corrected_word)
    corrected_sentence = ' '.join(corrected_words)

    return corrected_sentence


def find_closest_word(word, vocabulary):
    return min(vocabulary, key=lambda v: levenshtein_distance(word, v))


def text_preparation_with_levenshtein(sentence):
    # 1. Lowercase everything
    sentence = sentence.lower()

    # 2. Remove all symbols other than a-z.
    pattern = re.compile(r"[^a-z ]")
    sentence = re.sub(pattern, " ", sentence)

    # 3. Replace words with the closest in vocabulary
    words = sentence.split()
    closest_words = [find_closest_word(word, vocabulary) for word in words]
    corrected_sentence = ' '.join(closest_words)

    return corrected_sentence


# Pattern for characters
# pattern = re.compile(r"[^a-z ]")

# Pattern for punctuatin
pattern = re.compile(r"[^\w\s]")


def text_preparetion_simple(sentence):
    # 1. Lowercase everything
    sentence = sentence.lower()

    # 2. Remove all symbols other than a-z.
    sentence = re.sub(pattern, "", sentence)

    # # Tokenize the cleaned sentence
    # words = word_tokenize(sentence)
    #
    # # Generate n-grams for each n in ngram_ranges
    # all_grams = words[:]
    # ngram_ranges = [2, 3, 4]
    # for n in ngram_ranges:
    #     n_grams = ['_'.join(gram) for gram in nltk.ngrams(words, n)]
    #     all_grams.extend(n_grams)
    #
    # # Combine all components back to single string
    # sentence = ' '.join(all_grams)

    return sentence



sentence1 = "A World War II-era bomber flying out of formation"
print(text_preparetion_simple(sentence1))

sentence_example = '"A domesticated carnivvorous mzammal that typicbally hfaas a lons sfnout, an acxujte sense off osmell, noneetractaaln crlaws, anid xbarkring,y howlingu, or whining rvoiche."'
# print(text_preparation_with_spell_correction(sentence_example))