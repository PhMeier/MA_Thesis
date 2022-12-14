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
- Verben reinstecken Done
- Unterscheidung nach Auxiliar: to and that!

"""
import spacy
import pandas as pd
from nltk.stem import *
from spacy.matcher import Matcher
from nltk.stem.snowball import SnowballStemmer


def read_in_verbs(filename):
    """
    Redas in the veridical verbs.
    :param filename:
    :return:
    """
    signature_and_verbs = {}
    key = ""
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("\n")[0]
            if "/" in line:
                key = line
                signature_and_verbs[key] = []
            else:
                signature_and_verbs[key].append(line)
    return signature_and_verbs


def read_sick_instances(filename):
    df = pd.read_csv(filename)
    prem = df["sentence_B_original"]
    hypo = df["sentence_A_dataset"]
    return prem, hypo


def neg_environment_someone_sentences_parses(instance, verb, aux, nlp, label):
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
    lemmas = [token.lemma_ for token in doc]# get the lemmas
    #print(pos_tags)
    #print(text)
    x = list(zip(pos_tags, text)) # zip both
    #print(x)
    words = []
    stop = False # a specific part of the original sentence should be left out
    if aux == "to":
        for i in range(len(x)):
            print(x[i])
            if x[i][0] == "VERB" and stop: # appearance of a verb marks the begin of a the part which is wanted (enf of
                # sentence)
                # take the lemma instead!
                words.append(lemmas[text.index(x[i][1])])
                stop = False
                continue
            if x[i][0] != "SCONJ" and not stop: # Append this part of the sentence
                if x[i][0] == "VERB" and i > 4: # eg when later in the sentence appears "playing"
                    words.append(lemmas[text.index(x[i][1])])
                else:
                    words.append(x[i][1])
            if x[i][0] == "SCONJ" and not stop:# marks the begin of the left out area of the sentence
                words.append(x[i][1]) # add the word with sconj but then set stop to True
                i+=1
                stop = True
            if stop: # simply left out this part of the sentence by increasing i
                i+=1
    else:
        words[1] = verb
    print(words)
    verb = "does not " + verb
    words[1] = verb
    words[2] = aux
    print(words)
    print(" ".join(words))
    hypothesis = build_hypothesis(words, nlp)
    label = label_dictionary[label.split("/")[1]]
    print("hypo: ", hypothesis)
    return result, hypothesis, label


def pos_environment_someone_sentences_parses(instance, verb, aux, nlp, label):
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
    lemmas = [token.lemma_ for token in doc]# get the lemmas
    #print(pos_tags)
    #print(text)
    x = list(zip(pos_tags, text)) # zip both
    #print(x)
    words = []
    if aux == "to":
        stop = False # a specific part of the original sentence should be left out
        for i in range(len(x)):
            print(x[i])
            if x[i][0] == "VERB" and stop: # appearance of a verb marks the begin of a the part which is wanted (enf of
                # sentence)
                # take the lemma instead!
                words.append(lemmas[text.index(x[i][1])])
                stop = False
                continue
            if x[i][0] != "SCONJ" and not stop: # Append this part of the sentence
                if x[i][0] == "VERB" and i > 4: # eg when later in the sentence appears "playing"
                    words.append(lemmas[text.index(x[i][1])])
                else:
                    words.append(x[i][1])
            if x[i][0] == "SCONJ" and not stop:# marks the begin of the left out area of the sentence
                words.append(x[i][1]) # add the word with sconj but then set stop to True
                i+=1
                stop = True
            if stop: # simply left out this part of the sentence by increasing i
                i+=1
    else:
        words[1] = verb
    print(words)
    words[1] = verb
    words[2] = aux
    print(words)
    print(" ".join(words))
    hypothesis = build_hypothesis(words, nlp)
    label = label_dictionary[label.split("/")[0]]
    print("hypo: ", hypothesis)
    result = " ".join(words)
    return result, hypothesis, label


def build_hypothesis(words, nlp):
    """
    Builds then ew hypothesis given the words of the premise
    :param words: words of the premise
    :return: New hypothesis
    """
    sentence = " ".join(words)
    doc = nlp(sentence)
    pos_tags = [token.pos_ for token in doc] # get the pos tags
    text = [token.text for token in doc] # get the text
    sliced_sentece = pos_tags[6:]
    if "VERB" in sliced_sentece: # if we later see a verb in the sentence
        w = text[6:][pos_tags[6:].index("VERB")]
        words[words.index(w)] = w if w.endswith("s") else w + "s"
    if words[3].endswith("s"):
        hypothesis = words[0] + " " + words[3] + "es " + " ".join(words[4:])
    else:
        hypothesis = words[0] + " " + words[3] + "s " + " ".join(words[4:])
    return hypothesis


def pos_environment_sick(instance, verb, aux, nlp, label):
    result = ""
    verb = verb+"s"
    verb_and_aux = verb + " " + aux + " "
    doc = nlp(instance)  # parse the sentence
    pos_tags = [token.pos_ for token in doc]  # get the pos tags
    text = [token.text for token in doc]
    lemmas = [token.lemma_ for token in doc]
    #print(lemmas)
    if aux == "to":
        if pos_tags[:4] == ['DET', 'NOUN', 'AUX', 'VERB']:
            stemmed_verb = lemmas[3]
            hypo_verb = adapt_verb(stemmed_verb)
            result = " ".join(text[:2]) + " " + verb_and_aux + stemmed_verb + " " + " ".join(text[4:])
            hypo = " ".join(text[:2]) + " " + hypo_verb + " " + " ".join(text[4:])
            label = label_dictionary[label.split("/")[0]]
            print("{} | {} | {}".format(result, hypo, label))
            return result, hypo, label
        else:
            return "", "", ""
    else:
        return "", "", ""


def neg_environment_sick(instance, verb, aux, nlp, label):
    verb_and_aux = "does not " + verb + " " + aux + " "
    doc = nlp(instance)  # parse the sentence
    pos_tags = [token.pos_ for token in doc]  # get the pos tags
    text = [token.text for token in doc]
    lemmas = [token.lemma_ for token in doc]
    #print(lemmas)
    if aux == "to":
        if pos_tags[:4] == ['DET', 'NOUN', 'AUX', 'VERB']:
            stemmed_verb = lemmas[3]
            hypo_verb = adapt_verb(stemmed_verb)
            result = " ".join(text[:2]) + " " + verb_and_aux + stemmed_verb + " " + " ".join(text[4:])
            hypo = " ".join(text[:2]) + " " + hypo_verb + " " + " ".join(text[4:])
            label = label_dictionary[label.split("/")[1]]
            print("{} | {} | {}".format(result, hypo, label))
            return result, hypo, label
        else:
            return "", "", ""
    else:
        return "", "", ""




def active_passive_checker(instance, nlp):
    """
    Check if a sentence is active or passive. Returns True when passive, else False.
    :param instance:
    :param spacy:
    :return:
    """
    doc = nlp(instance)
    token_dep = [token.dep_ for token in doc]
    # if one of these three is included, the sentence is passive
    if "nsubjpass" in token_dep or "auxpass" in token_dep or "agent" in token_dep:
        return True
    else:
        return False


def adapt_verb(verb):
    """
    Adapt the given verb to the third person according to its ending.
    :param verb:
    :return verb:
    """
    if verb.endswith("y"):
        if verb.endswith("ay"):
            verb = verb + "s"
            return verb
        else:
            verb = verb[:len(verb)-1] + "ies"
            return verb
    elif verb.endswith("s"):
        return verb
    else:
        verb = verb + "s"
        return verb


if __name__ == "__main__":
    label_dictionary = {"Plus":0, "Neutral":1, "Minus":2}
    nlp = spacy.load("en_core_web_lg")
    pos, neg = [], []
    #pos_environment_sick("The woman is dicing garlic", "forget", "to", nlp, "Plus/Plus")
    #neg_environment_sick("The woman is dicing garlic", "forget", "to", nlp, "Plus/Plus")
    sick_premise, sick_hypo = read_sick_instances("../utils/extracted_sick_instances.csv")
    signatures_and_verbs = read_in_verbs("all_veridical_verbs.txt")
    for sentence in sick_premise:
        if not active_passive_checker(sentence, nlp):
            for signature, verbs in signatures_and_verbs.items():
                for verb in verbs:
                    verb, aux = verb.split()
                    p_res, p_hypo, p_label = pos_environment_sick(sentence, verb, aux, nlp, signature)
                    n_res, n_hypo, n_label = neg_environment_sick(sentence, verb, aux, nlp, signature)
                    if p_res != "":
                        pos.append([p_res, p_hypo, p_label])
                    if n_res != "":
                        neg.append([n_res, n_hypo, n_label])

    pos_df = pd.DataFrame(pos, columns=["Premise", "Hypothesis", "Label"])
    neg_df = pd.DataFrame(neg, columns=["Premise", "Hypothesis", "Label"])
    pos_df.to_csv("pos_env_sick.txt", index=False, header=True)
    neg_df.to_csv("neg_env_sick.txt", index=False, header=True)






    #pos_environment_sick("The woman is dicing garlic", "forget", "to", nlp, "Plus/Plus")
    #neg_environment_sick("A boy is studying a calendar", "forget", "to", nlp, "Plus/Plus")

    #signatures_and_verbs = read_in_verbs("all_veridical_verbs.txt")
    #active_passive_checker("A boy is being served by its mom.", nlp)
    #active_passive_checker("A boy serves its mom.", nlp)

    """
    #pos_environment_sick("A fish is being sliced by a man", "forgets", "to", nlp)

    doc = nlp("Someone remembers that a child in light colored clothing is standing with his arms extended") #"Someone remembers that a child in light colored clothing is standing with his arms extended outward.")
    print(f"{'text':{8}} {'POS':{6}} {'TAG':{6}} {'Dep':{6}} {'POS explained':{20}} {'tag explained'} ")

    for idx, token in enumerate(doc):
        print(f'{idx} {token.text:{8}} {token.pos_:{6}} {token.tag_:{6}} {token.dep_:{6}} {spacy.explain(token.pos_):{20}} {spacy.explain(token.tag_)}')

    pos_environment_someone_sentences_parses("Someone doubts that the woman is putting makeup on the man", "forgets",
                                             "to", nlp)
    neg_environment_someone_sentences_parses("Someone observes that kids are being dressed in costumes and playing a game", "forget",
                                             "to", nlp)
   # pos_environment_someone_sentences_parses("Someone doubts that some food is being prepared by a chef", "forgets",
   #                                          "to", nlp)
    """
    """
    pos_environment_someone_sentences_parses("Someone remembers that a child in light colored clothing is standing with his arms extended", "forgets",
                                             "to", nlp)
    #pos_environment_sick("A boy is mowing the lawn", "forgets to", stemmer)
    #pos_environment_sick("A boy is mowing the lawn", "forget to")

    #pos_environment_someone_sentences("Someone doubts that the woman is putting makeup on the man", "forgets", "to", stemmer)
    """
    """
    nlp = spacy.load('en_core_web_sm') # spellcheck
    contextualSpellCheck.add_to_pipe(nlp)
    print(nlp.pipe_names)
    doc = nlp("The boy forgest to mow lawn.")
    print(doc._.performed_spellCheck)  # Should be True
    print("laberlachs: ", doc._.outcome_spellCheck)
    """
