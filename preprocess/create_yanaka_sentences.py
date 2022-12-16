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
    prem = df["sentence_B_original"].to_list()
    hypo = df["sentence_A_dataset"].to_list()
    return prem, hypo


def neg_environment_someone_sentences_parses(instance, orig_hypo, verb, aux, nlp, label_verid):
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
    doc = nlp(instance)  # parse the sentence
    pos_tags = [token.pos_ for token in doc]  # get the pos tags
    text = [token.text for token in doc]  # get the text
    lemmas = [token.lemma_ for token in doc]  # get the lemmas
    # print(pos_tags)
    # print(text)
    x = list(zip(pos_tags, text))  # zip both
    # print(x)
    words = []
    stop = False  # a specific part of the original sentence should be left out
    if aux == "to":
        for i in range(len(x)):
            #print(x[i])
            if x[i][0] == "VERB" and stop:  # appearance of a verb marks the begin of a the part which is wanted (enf of
                # sentence)
                # take the lemma instead!
                words.append(lemmas[text.index(x[i][1])])
                stop = False
                continue
            if x[i][0] != "SCONJ" and not stop:  # Append this part of the sentence
                if x[i][0] == "VERB" and i > 4:  # eg when later in the sentence appears "playing"
                    words.append(lemmas[text.index(x[i][1])])
                else:
                    words.append(x[i][1])
            if x[i][0] == "SCONJ" and not stop:  # marks the begin of the left out area of the sentence
                words.append(x[i][1])  # add the word with sconj but then set stop to True
                i += 1
                stop = True
            if stop:  # simply left out this part of the sentence by increasing i
                i += 1
        hypothesis = build_hypothesis(words, nlp)
    else:
        words = text
        hypothesis = orig_hypo

    verb = "does not " + verb
    words[1] = verb
    words[2] = aux
    #print(words)
    result = " ".join(words)
    #hypothesis = build_hypothesis(words, nlp)
    label = label_dictionary[label_verid.split("/")[1]]
    label_compos = label_composite_someone(label_verid, False)
    #print("hypo: ", hypothesis)
    #print("NEG: ", result, hypothesis, label, label_compos)
    return result, hypothesis, label, label_compos


def pos_environment_someone_sentences_parses(instance, orig_hypo, verb, aux, nlp, label_verid):
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
    verb = verb + "s"
    doc = nlp(instance)  # parse the sentence
    pos_tags = [token.pos_ for token in doc]  # get the pos tags
    text = [token.text for token in doc]  # get the text
    lemmas = [token.lemma_ for token in doc]  # get the lemmas
    # print(pos_tags)
    # print(text)
    x = list(zip(pos_tags, text))  # zip both
    # print(x)
    words = []
    if aux == "to":
        stop = False  # a specific part of the original sentence should be left out
        for i in range(len(x)):
            #print(x[i])
            if x[i][0] == "VERB" and stop:  # appearance of a verb marks the begin of a the part which is wanted (enf of
                # sentence)
                # take the lemma instead!
                words.append(lemmas[text.index(x[i][1])])
                stop = False
                continue
            if x[i][0] != "SCONJ" and not stop:  # Append this part of the sentence
                if x[i][0] == "VERB" and i > 4:  # eg when later in the sentence appears "playing"
                    words.append(lemmas[text.index(x[i][1])])
                else:
                    words.append(x[i][1])
            if x[i][0] == "SCONJ" and not stop:  # marks the begin of the left out area of the sentence
                words.append(x[i][1])  # add the word with sconj but then set stop to True
                i += 1
                stop = True
            if stop:  # simply left out this part of the sentence by increasing i
                i += 1
        words[1] = verb
        words[2] = aux
        #print(words)
        #print(" ".join(words))
        hypothesis = build_hypothesis(words, nlp)
    else:
        text[1] = verb
        words = text
        hypothesis = orig_hypo
    label = label_dictionary[label_verid.split("/")[0]]
    label_compos = label_composite_someone(label_verid, True)
    #print("hypo: ", hypothesis)
    result = " ".join(words)
    #print(result, hypothesis, label, label_compos)
    return result, hypothesis, label, label_compos


def label_composite_someone(label, pos_env):
    label_list = ["Plus/Minus", "Minus/Plus", "Neutral/Minus", "Minus/Neutral", "Neutral/Neutral", "Neutral/Plus"]
    if label == "Plus/Neutral" and not pos_env:
        return 1
    if label in label_list:
        return 1
    else:
        return 0


def build_hypothesis(words, nlp):
    """
    Builds then ew hypothesis given the words of the premise
    :param words: words of the premise
    :return: New hypothesis
    """
    sentence = " ".join(words)
    doc = nlp(sentence)
    pos_tags = [token.pos_ for token in doc]  # get the pos tags
    text = [token.text for token in doc]  # get the text
    sliced_sentece = pos_tags[6:]
    if "VERB" in sliced_sentece:  # if we later see a verb in the sentence
        w = text[6:][pos_tags[6:].index("VERB")]
        words[words.index(w)] = w if w.endswith("s") else w + "s"
    if words[3].endswith("s"):
        hypothesis = words[0] + " " + words[3] + "es " + " ".join(words[4:])
    else:
        hypothesis = words[0] + " " + words[3] + "s " + " ".join(words[4:])
    return hypothesis


def pos_environment_sick(instance, verb, aux, nlp, label_verid):
    """
    Creates new veridical instances with 'to' auxiliar.
    :param instance:
    :param verb:
    :param aux:
    :param nlp:
    :param label:
    :return:
    """
    if aux == "to":
        result = ""
        verb = verb + "s"
        verb_and_aux = verb + " " + aux + " "
        doc = nlp(instance)  # parse the sentence
        pos_tags = [token.pos_ for token in doc]  # get the pos tags
        text = [token.text for token in doc]
        lemmas = [token.lemma_ for token in doc]
        # print(lemmas)
        if pos_tags[:4] == ['DET', 'NOUN', 'AUX', 'VERB']:
            stemmed_verb = lemmas[3]
            hypo_verb = adapt_verb(stemmed_verb)
            result = " ".join(text[:2]) + " " + verb_and_aux + stemmed_verb + " " + " ".join(text[4:])
            hypo = " ".join(text[:2]) + " " + hypo_verb + " " + " ".join(text[4:])
            label = label_dictionary[label_verid.split("/")[0]]
            label_s2 = calculate_composite_label(label, label_verid)
            print("{} | {} | {}".format(result, hypo, label))
            return result, hypo, label, label_s2
        else:
            return "", "", "", ""
    else:
        return "", "", "", ""


def neg_environment_sick(instance, verb, aux, nlp, label_verid):
    """
    Similar function like pos_environment_sick, but creates a negated sentence, see verb_and_aux definition.
    :param instance:
    :param verb:
    :param aux:
    :param nlp:
    :param label:
    :return:
    """
    if aux == "to":
        verb_and_aux = "does not " + verb + " " + aux + " "
        doc = nlp(instance)  # parse the sentence
        pos_tags = [token.pos_ for token in doc]  # get the pos tags
        text = [token.text for token in doc]
        lemmas = [token.lemma_ for token in doc]
        if pos_tags[:4] == ['DET', 'NOUN', 'AUX', 'VERB']:  # first four words of the sick instance need this tags
            stemmed_verb = lemmas[3]
            hypo_verb = adapt_verb(stemmed_verb)
            result = " ".join(text[:2]) + " " + verb_and_aux + stemmed_verb + " " + " ".join(text[4:])
            hypo = " ".join(text[:2]) + " " + hypo_verb + " " + " ".join(text[4:])
            label = label_dictionary[label_verid.split("/")[1]]
            label_s2 = calculate_composite_label(label, label_verid)
            print("{} | {} | {}".format(result, hypo, label))
            return result, hypo, label, label_s2
        else:
            return "", "", "", ""
    else:
        return "", "", "", ""


def calculate_composite_label(label_1, label_verid):
    """
    The label of the inference f(s1) --> s2 can differ from f(s1) --> s1.
    :param label:
    :return:
    """
    label_list = ["Minus/Neutral", "Neutral/Minus"]  # these labels cause an unknwon for the composite
    # for these signatures the label f(s1) --> s2 is equal to f(s1) --> s1
    set_equal = ["Minus/Plus", "Plus/Minus", "Plus/Neutral", "Plus/Plus", "Neutral/Neutral"]
    if label_verid in set_equal:
        return label_1
    if label_verid in label_list:
        return 1


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
            verb = verb[:len(verb) - 1] + "ies"
            return verb
    elif verb.endswith("s"):
        return verb
    else:
        verb = verb + "s"
        return verb


def someone_routine(signatures_and_verbs, nlp):
    df = pd.read_csv("../utils/premise_dev_to_composite.csv")
    prem = df["f(s1)"].to_list()
    hypo = df["s1"].to_list()
    transit = df["s2"].to_list()
    label_fs1_to_s1 = df["label_f(s1)_to_s1"].to_list()
    label_fs1_to_s2 = df["label_f(s1)_to_s2"].to_list()
    pos, neg = [], []
    for i in range(len(prem[:1])):
        for signature, verbs in signatures_and_verbs.items():
            for verb in verbs:
                verb, aux = verb.split()
                p_result, p_hypothesis, p_label, p_label_compos = pos_environment_someone_sentences_parses(prem[i], hypo[i], verb, aux, nlp, signature)
                n_result, n_hypothesis, n_label, n_label_compos = neg_environment_someone_sentences_parses(prem[i], hypo[i], verb, aux, nlp, signature)
                pos.append([p_result, p_hypothesis, transit[i], p_label, p_label_compos, label_fs1_to_s1[i], label_fs1_to_s2[i]])
                neg.append([n_result, n_hypothesis, transit[i], n_label, n_label_compos, label_fs1_to_s1[i], label_fs1_to_s2[i]])
    pos_df = pd.DataFrame(pos, columns=["Premise", "Hypothesis", "Transitive", "Label", "Label Trans", "orig_label_f(s1)_to_s1", "orig_label_f(s1)_to_s2"])
    neg_df = pd.DataFrame(neg, columns=["Premise", "Hypothesis", "Transitive", "Label", "Label Trans", "orig_label_f(s1)_to_s1", "orig_label_f(s1)_to_s2"])
    pos_df.to_csv("pos_env_someone.csv", index=False, header=True)
    neg_df.to_csv("neg_env_someone.csv", index=False, header=True)


def extracted_sick_intances_routine(signatures_and_verbs, nlp):
    pos, neg = [], []
    sick_premise, sick_hypo = read_sick_instances("../utils/extracted_sick_instances.csv")

    for sentence, sick_hypo in zip(sick_premise[:2], sick_hypo[:2]):# noch sick_hypo mit reinnehmen
        if not active_passive_checker(sentence, nlp):
            for signature, verbs in signatures_and_verbs.items():
                for verb in verbs:
                    verb, aux = verb.split()
                    p_res, p_hypo, p_label, p_label_s2 = pos_environment_sick(sentence, verb, aux, nlp, signature)
                    n_res, n_hypo, n_label, n_label_s2 = neg_environment_sick(sentence, verb, aux, nlp, signature)
                    # auch die gegebene hypothese speichern, label entsprechend auch mitgeben
                    if p_res != "":
                        pos.append([p_res, p_hypo, sick_hypo, p_label, p_label_s2])
                    if n_res != "":
                        neg.append([n_res, n_hypo, sick_hypo, n_label, n_label_s2])
    #print(pos)
    pos_df = pd.DataFrame(pos, columns=["f(s1)", "s1", "s2", "Label", "Label_s2"])
    neg_df = pd.DataFrame(neg, columns=["f(s1)", "s1", "s2", "Label", "Label_s2"])
    pos_df.to_csv("pos_env_sick.txt", index=False, header=True)
    neg_df.to_csv("neg_env_sick.txt", index=False, header=True)


if __name__ == "__main__":
    label_dictionary = {"Plus": 0, "Neutral": 1, "Minus": 2}
    nlp = spacy.load("en_core_web_lg")
    signatures_and_verbs = read_in_verbs("all_veridical_verbs.txt")
    pos, neg = [], []
    s_pos, s_neg = [], []
    # pos_environment_sick("The woman is dicing garlic", "forget", "to", nlp, "Plus/Plus")
    # neg_environment_sick("The woman is dicing garlic", "forget", "to", nlp, "Plus/Plus")

    someone_routine(signatures_and_verbs, nlp)
    extracted_sick_intances_routine(signatures_and_verbs, nlp)

    # pos_environment_sick("The woman is dicing garlic", "forget", "to", nlp, "Plus/Plus")
    # neg_environment_sick("A boy is studying a calendar", "forget", "to", nlp, "Plus/Plus")

    # signatures_and_verbs = read_in_verbs("all_veridical_verbs.txt")
    # active_passive_checker("A boy is being served by its mom.", nlp)
    # active_passive_checker("A boy serves its mom.", nlp)

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
