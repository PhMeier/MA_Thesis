
import spacy
import pandas as pd

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
            print("{} | {} | {} | {}".format(result, hypo, label, label_s2))
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
            print("{} | {} | {} | {}".format(result, hypo, label, label_s2))
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

if __name__ == "__main__":
    label_dictionary = {"Plus": 0, "Neutral": 1, "Minus": 2}
    nlp = spacy.load("en_core_web_lg")
    signatures_and_verbs = read_in_verbs("all_veridical_verbs.txt")
    pos_environment_sick("A man is riding a motorbike", "forget", "to", nlp, "Minus/Plus")
    path_to_sick = "../data/SICK/SICK.txt"
    df = pd.read_csv(path_to_sick, sep="\t")
    sick_premise = df["sentence_A"].to_list()
    sick_hypo = df["sentence_B"].to_list()
    label = df["entailment_label"].to_list()
    pos, neg = [], []
    for sentence, sick_hypo in zip(sick_premise, sick_hypo):# noch sick_hypo mit reinnehmen
        if sentence.startswith("A"):
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
    pos_df.to_csv("pos_env_complete_sick.csv", index=False, header=True)
    neg_df.to_csv("neg_env_complete_sick.csv", index=False, header=True)