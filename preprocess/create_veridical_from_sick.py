import csv

import spacy
import pandas as pd

def pos_environment_sick(instance, verb, aux, nlp, label_verid, sick_label):
    """
    Creates new veridical instances with 'to' auxiliar.
    :param instance:
    :param verb:
    :param aux:
    :param nlp:
    :param label:
    :return:
    """
    doc = nlp(instance)  # parse the sentence
    pos_tags = [token.pos_ for token in doc]  # get the pos tags
    text = [token.text for token in doc]
    lemmas = [token.lemma_ for token in doc]
    if aux == "to":
        pronoun, plural = find_pronoun(text[1])
        if plural:
            verb_and_aux = verb + " " + aux + " "
        else:
            verb_and_aux = verb + "s " + aux + " "
        # print(lemmas)
        if pos_tags[:4] == ['DET', 'NOUN', 'AUX', 'VERB']:
            stemmed_verb = lemmas[3]
            hypo_verb = adapt_verb(stemmed_verb, plural)
            result = " ".join(text[:2]) + " " + verb_and_aux + stemmed_verb + " " + " ".join(text[4:])
            hypo = " ".join(text[:2]) + " " + hypo_verb + " " + " ".join(text[4:])
            label = label_dictionary[label_verid.split("/")[0]]
            label_s2 = calculate_composite_label(label, label_verid, sick_label, True)
            print("{} | {} | {}".format(result, hypo, label))
            return result, hypo, label, label_s2, label_verid
        else:
            return "", "", "", "", ""
    else:
        # "The woman is dicing garlic" The woman does not forget that she is dicing garlic
        pronoun, plural = find_pronoun(text[1])
        if plural:
            verb_and_aux = verb + " " + aux + " "
        else:
            verb_and_aux = verb + "s " + aux + " "
        if pos_tags[:4] == ['DET', 'NOUN', 'AUX', 'VERB']:
            complement_aux_and_verb = text[2] + " " + text[3] + " "
            stemmed_verb = lemmas[3]
            hypo_verb = adapt_verb(stemmed_verb, plural)
            result = " ".join(text[:2]) + " " + verb_and_aux + pronoun + " " + complement_aux_and_verb + " ".join(
                text[4:])
            hypo = " ".join(text[:2]) + " " + hypo_verb + " " + " ".join(text[4:])
            label = label_dictionary[label_verid.split("/")[0]]
            label_s2 = calculate_composite_label(label, label_verid, sick_label, True)
            print("{} | {} | {}".format(result, hypo, label))
            return result, hypo, label, label_s2, label_verid
            # print(result)
            # print(hypo)
        else:
            return "", "", "", "", ""



def neg_environment_sick(instance, verb, aux, nlp, label_verid, sick_label):
    """
    Similar function like pos_environment_sick, but creates a negated sentence, see verb_and_aux definition.
    :param instance:
    :param verb:
    :param aux:
    :param nlp:
    :param label:
    :return:
    """
    doc = nlp(instance)  # parse the sentence
    pos_tags = [token.pos_ for token in doc]  # get the pos tags
    text = [token.text for token in doc]
    lemmas = [token.lemma_ for token in doc]
    if aux == "to":
        verb_and_aux = "does not " + verb + " " + aux + " "
        if pos_tags[:4] == ['DET', 'NOUN', 'AUX', 'VERB']:  # first four words of the sick instance need this tags
            stemmed_verb = lemmas[3]
            pronoun, plural = find_pronoun(text[1])
            hypo_verb = adapt_verb(stemmed_verb, plural)
            result = " ".join(text[:2]) + " " + verb_and_aux + stemmed_verb + " " + " ".join(text[4:])
            hypo = " ".join(text[:2]) + " " + hypo_verb + " " + " ".join(text[4:])
            label = label_dictionary[label_verid.split("/")[1]]
            label_s2 = calculate_composite_label(label, label_verid, sick_label, False)
            print("{} | {} | {}".format(result, hypo, label))
            return result, hypo, label, label_s2, label_verid
        else:
            return "", "", "", "", ""
    else:
        # "The woman is dicing garlic" The woman does not forget that she is dicing garlic
        verb_and_aux = "does not " + verb + " " + aux + " "
        if pos_tags[:4] == ['DET', 'NOUN', 'AUX', 'VERB']:
            complement_aux_and_verb = text[2] + " " + text[3] + " "
            pronoun, plural = find_pronoun(text[1])
            stemmed_verb = lemmas[3]
            hypo_verb = adapt_verb(stemmed_verb, plural)
            result = " ".join(text[:2]) + " " + verb_and_aux + pronoun + " " + complement_aux_and_verb + " ".join(text[4:])
            hypo = " ".join(text[:2]) + " " + hypo_verb + " " + " ".join(text[4:])
            label = label_dictionary[label_verid.split("/")[1]]
            label_s2 = calculate_composite_label(label, label_verid, sick_label, False)
            print("{} | {} | {}".format(result, hypo, label))
            return result, hypo, label, label_s2, label_verid
            #print(result)
            #print(hypo)
        else:
            return "", "", "", "", ""
    return "", "", "", "", ""


def find_pronoun(text):
    it = ["monkey", "dog","cat", "puppy", "person"]
    female = ["girl", "woman", "girl", "lady"]
    plural = ["animals", "men", "ferrets", "people", "dogs"]
    if text in female:
        return "she", False
    if text in it:
        return "it", False
    if text in plural:
        return "they", True
    else:
        return "he", False


def calculate_composite_label(label_1, label_verid, sick_label, pos_env):
    """
    The label of the inference f(s1) --> s2 can differ from f(s1) --> s1.
    :param label:
    :return:
    """
    if sick_label == "Neutral": # 09/01
        return 1 # Contradiction.Neutral.Contradiction(neg), Entailment.Neutral.Entailment(pos)
    label_list = ["Minus/Neutral", "Neutral/Minus", "Neutral/Plus"]  # these labels cause an unknwon for the composite
    # for these signatures the label f(s1) --> s2 is equal to f(s1) --> s1
    set_equal = ["Minus/Plus", "Plus/Minus", "Plus/Neutral", "Plus/Plus", "Neutral/Neutral"]
    if pos_env:
        if label_verid in set_equal:
            return label_1
        if label_verid in label_list:
            return 1
    else:
        label_list = ["Minus/Neutral", "Neutral/Minus"]
        if label_verid == "Neutral/Plus":
            return 0
        if label_verid in label_list:
            return 1
        if label_verid in set_equal:
            return label_1




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


def adapt_verb(verb, plural):
    """
    Adapt the given verb to the third person according to its ending.
    :param verb:
    :return verb:
    """
    if plural:
        return verb
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

def filter_instances(original, results_step1):
    """
    Based on the results of step 1: s1 --> s2 the original SICK instances are filtered.
    These instances are used to create the veridical instances.
    :param original:
    :param results_step1:
    :return:
    """
    indices = results_step1["index"].tolist()
    #print(indices)
    #print(original.columns)
    #original["index"] = range(0, len(original)) # introduce an index column
    original.insert(1, "Index", range(0, len(original)), True)
    #print(original.columns)
    result = original.loc[indices] # get the instances, which are correctly classified by index
    #print(result)
    result.to_csv("step_2_extracted_instances.csv", index=False)
    #print(result.columns)
    return result




if __name__ == "__main__":
    label_dictionary = {"Plus": 0, "Neutral": 1, "Minus": 2}
    num_to_label = {0:"Entailment", 1:"Neutral", 2:"Contradiction"}
    label_dictionary_sick = {"ENTAILMENT": 0, "NEUTRAL": 1, "CONTRADICTION": 2}
    nlp = spacy.load("en_core_web_lg")
    signatures_and_verbs = read_in_verbs("all_veridical_verbs.txt")
    pos_environment_sick("A man is riding a motorbike", "forget", "to", nlp, "Minus/Plus")
    path_to_sick = "../utils/extracted_sick_instances.csv" # "../data/SICK/SICK.txt"
    df = pd.read_csv(path_to_sick, sep=",")
    df_res_step1 = pd.read_csv("./sick/commonalities_all.csv") # step 2
    df = filter_instances(df, df_res_step1)
    #"""
    #sick_premise = df["sentence_A"].to_list()
    #sick_hypo = df["sentence_B"].to_list()
    #label = df["entailment_label"].to_list()
    sick_premise = df["sentence_B_original"].to_list()
    sick_hypo = df["sentence_A_dataset"].to_list()
    label = df["SemEval_set"].to_list()
    pos, neg = [], []
    for sentence, sick_hypo, lab in zip(sick_premise, sick_hypo, label):# noch sick_hypo mit reinnehmen
        if sentence.startswith("A"):
            if not active_passive_checker(sentence, nlp):
                for signature, verbs in signatures_and_verbs.items():
                    for verb in verbs:
                        verb, aux = verb.split()
                        p_res, p_hypo, p_label, p_label_s2, sig = pos_environment_sick(sentence, verb, aux, nlp, signature, lab)
                        n_res, n_hypo, n_label, n_label_s2, sig = neg_environment_sick(sentence, verb, aux, nlp, signature, lab)
                        # auch die gegebene hypothese speichern, label entsprechend auch mitgeben
                        # create a signature similar to Yanaka 'entailment.neutral.neutral'.
                        if p_res != "":
                            pos_sig = num_to_label[p_label] + "." + lab.title() + "." + num_to_label[p_label_s2]
                            print(pos_sig)
                            pos.append([pos_sig, p_res, p_hypo, sick_hypo, p_label, p_label_s2, sig])
                        if n_res != "":
                            neg_sig = num_to_label[n_label] + "." + lab.title() + "." + num_to_label[n_label_s2]
                            neg.append([neg_sig, n_res, n_hypo, sick_hypo, n_label, n_label_s2, sig])
    #print(pos)
    pos_df = pd.DataFrame(pos, columns=["complete_signature", "f(s1)", "s1", "s2", "Label", "Label_s2", "Signature"])
    neg_df = pd.DataFrame(neg, columns=["complete_signature", "f(s1)", "s1", "s2", "Label", "Label_s2", "Signature"])
    pos_df.to_csv("pos_env_complete_sick_new.csv", index=False, header=True)
    neg_df.to_csv("neg_env_complete_sick_new.csv", index=False, header=True)
    #"""
