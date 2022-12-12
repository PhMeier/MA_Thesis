"""
In this script new instances for the Yanaka Naturalistic Dataset are created.

Plan:
- Need the verbs according to their signature
- Positive and negative environments are realized in a function
- Stem the verb, construct the new sentence and run a spellchecker, e.g spacy over the new instances?
- Update: 11/12: Kein Spellchecker!
- Idee 12/12: Include Tagger? We've always seen"Someone thinks that a person ..."
              However, there are instances like "Someone thinks that a man and a kid ..."
              Should handle conjunctions, so that they are easier to replace
- 12/12: Find the instances which should be processed for the new dataset.
- 12/12: Input: Premise, Hypothesis, Verbs of the signatures!
         Output: Premise, New Hypothesis, Label!
"""
from nltk.stem import *
import spacy
#import contextualSpellCheck


def pos_environment_sick(instance, verb, stemmer):
    words = instance.split()
    words[0] = "The"
    words[2] = verb
    print(stemmer.stem(words[3]))
    words[3] = stemmer.stem(words[3])
    print(words)
    result = " ".join(words)
    print(result)
    # Neuer Satz: The x forgets to mow the lawn


def neg_environment_sick(instance, verb, stemmer):
    words = instance.split()
    words[0] = "The"
    words[2] = verb
    print(stemmer.stem(words[3]))
    words[3] = stemmer.stem(words[3])
    print(words)
    result = " ".join(words)
    print(result)


def pos_environment_someone_sentences(instance, verb, aux, stemmer):
    words = instance.split()
    print(words)
    new = words[0]
    words[1] = verb
    words[2] = aux
    words[6] = stemmer.stem(words[6])
    words.pop(4)
    words.pop(3)
    words.pop(3)
    print(words)
    result = " ".join(words)
    print(result)


def pos_environment_someone_sentences_parses(instance, verb, aux, stemmer, nlp):
    """
    Function to create new premise and hypothesis including verbs with 'to' conjunction. Made for 'simple' sentences
    like: 'Someone doubts that the woman is putting makeup on the man' and
    'Someone remembers that a child in light colored clothing is standing with his arms extended'
    The matrix verb and its conjuction should be replaced by parameter verb and aux.
    Returns the new premise and hypothesis and its label.
    :param instance: Premise which should be modified
    :param verb: verb which replaces the matrix verb of the premise
    :param aux: auxiliary of the verb
    :param stemmer: nltk stemmer for the complement verb
    :param nlp: spacy pipeline, used to POS-tag the instance
    :return:
    """
    # von sconj bis aux
    result = ""
    doc = nlp(instance) # parse the sentence
    pos_tags = [token.pos_ for token in doc] # get the pos tags
    text = [token.text for token in doc] # get the text
    #print(pos_tags)
    #print(text)
    x = list(zip(pos_tags, text)) # zip both
    #print(x)
    words = []
    stop = False # a specific part of the original sentence should be left out
    for i in range(len(x)):
        #print(x[i])
        if x[i][0] == "VERB" and stop: # appearance of a verb marks the begin of a the part which is wanted (enf of
            # sentence)
            words.append(stemmer.stem(x[i][1])) # stem the verb, eg putting --> put
            stop = False
            continue
        if x[i][0] != "SCONJ" and not stop: # Append this part of the sentence
            words.append(x[i][1])
        if x[i][0] == "SCONJ" and not stop:# marks the begin of the left out area of the sentence
            words.append(x[i][1]) # add the word with sconj but then set stop to True
            i+=1
            stop = True
        if stop: # simply left out this part of the sentence by increasing i
            i+=1
    print(words)
    words[1] = verb
    words[2] = aux
    print(words)
    print(" ".join(words))
    hypothesis = build_hypothesis(words)
    print("hypo: ", hypothesis)


def build_hypothesis(words):
    """
    Builds then ew hypothesis given the words of the premise
    :param words: words of the premise
    :return: New hypothesis
    """
    if words[3].endswith("s"):
        hypothesis = words[0] + " " + words[3] + "es " + " ".join(words[4:])
    else:
        hypothesis = words[0] + " " + words[3] + "s " + " ".join(words[4:])
    return hypothesis


if __name__ == "__main__":
    stemmer = PorterStemmer() # intialize stemmer here, not necessary to initialize it in every function call

    nlp = spacy.load("en_core_web_lg")
    doc = nlp("Someone remembers that a child in light colored clothing is standing with his arms extended") #"Someone remembers that a child in light colored clothing is standing with his arms extended outward.")
    print(f"{'text':{8}} {'POS':{6}} {'TAG':{6}} {'Dep':{6}} {'POS explained':{20}} {'tag explained'} ")

    for idx, token in enumerate(doc):
        print(f'{idx} {token.text:{8}} {token.pos_:{6}} {token.tag_:{6}} {token.dep_:{6}} {spacy.explain(token.pos_):{20}} {spacy.explain(token.tag_)}')

    pos_environment_someone_sentences_parses("Someone doubts that the woman is putting makeup on the man", "forgets",
                                             "to", stemmer, nlp)
    pos_environment_someone_sentences_parses("Someone observes that kids are being dressed in costumes and playing a game", "forgets",
                                             "to", stemmer, nlp)
    pos_environment_someone_sentences_parses("Someone doubts that some food is being prepared by a chef", "forgets",
                                             "to", stemmer, nlp)

    pos_environment_someone_sentences_parses("Someone remembers that a child in light colored clothing is standing with his arms extended", "forgets",
                                             "to", stemmer, nlp)
    #pos_environment_sick("A boy is mowing the lawn", "forgets to", stemmer)
    #pos_environment_sick("A boy is mowing the lawn", "forget to")

    #pos_environment_someone_sentences("Someone doubts that the woman is putting makeup on the man", "forgets", "to", stemmer)

    """
    nlp = spacy.load('en_core_web_sm') # spellcheck
    contextualSpellCheck.add_to_pipe(nlp)
    print(nlp.pipe_names)
    doc = nlp("The boy forgest to mow lawn.")
    print(doc._.performed_spellCheck)  # Should be True
    print("laberlachs: ", doc._.outcome_spellCheck)
    """
