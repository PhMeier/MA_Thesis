"""
Identify the original instances from the SICK dataset given the IDs in the naturalistic dataset.
"""
from collections import Counter
import pandas as pd
from transformers import BartTokenizer


def read_data(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data.append(line.strip().split("\t"))
    return data

def get_instances_by_id(sick_data, nat_data):
    # first, get the ids
    idx_nat = []
    extracted_rows = []
    for row in nat_data:
        idx_nat.append(row[0])
    for idx in idx_nat:
        for row in sick_data:
            if idx == row[0]:
                extracted_rows.append(row)
    return extracted_rows

def count_labels(extracted_data):
    labels = []
    for row in extracted_data:
        labels.append(row[3])
    c = Counter(labels)
    print(c)


def get_sick_instances(data):
    result = []
    for row in data[1:]:
        if row[3] == "sick":
            result.append(row)
    return result


if __name__ == "__main__":
    """
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    print(len(tokenizer))
    tokenizer.add_tokens(['<g>'], special_tokens=True)
    tokenizer.add_tokens(['</' + "g" + '>'], special_tokens=True)
    print(tokenizer.all_special_tokens)  # --> ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
    print(tokenizer.all_special_ids)
    print(len(tokenizer))
    y = "<g> ( <pointer:0> and :op1 ( <pointer:1> fear-01 :ARG0 ( <pointer:2> authority ) :ARG1 ( <pointer:3> leave-12 :ARG0 ( <pointer:4> and :op1 ( <pointer:5> overdevelopment ) :op2 ( <pointer:6> and :op1 ( <pointer:7> trend ) :op2 ( <pointer:8> taste ) :topic ( <pointer:9> tourism :mod ( <pointer:10> international ) ) :ARG1-of ( <pointer:11> new-01 ) ) ) :ARG1 ( <pointer:12> island :wiki <lit> Balearic Islands </lit> :name ( <pointer:13> name :op1 <lit> Balearic </lit> ) ) :ARG2 ( <pointer:14> behind ) ) ) :op2 ( <pointer:15> move-02 :ARG0 <pointer:2> :ARG1 ( <pointer:16> protect-01 :ARG0 <pointer:2> :ARG1 ( <pointer:17> area :ARG1-of ( <pointer:18> develop-02 :polarity - ) :ARG1-of ( <pointer:19> remain-01 ) ) :ARG3 ( <pointer:20> preserve-01 :ARG0 ( <pointer:21> nature ) :ARG1 <pointer:17> ) ) ) :op3 ( <pointer:22> proclaim-01 :ARG0 <pointer:2> :ARG1 ( <pointer:23> limit-01 :ARG0 <pointer:2> :ARG1 <pointer:17> :ARG2 ( <pointer:24> construct-01 :mod ( <pointer:25> all ) ) ) ) :op4 ( <pointer:26> blow-up-06 :ARG0 <pointer:2> :ARG1 ( <pointer:27> blemish-01 :ARG1 ( <pointer:28> hotel ) :ARG1-of ( <pointer:29> include-91 :ARG2 ( <pointer:30> blemish-01 :ARG1 ( <pointer:31> hotel ) :ARG1-of ( <pointer:32> have-degree-91 :ARG2 ( <pointer:33> look-02 :polarity - :ARG0 <pointer:30> ) :ARG3 ( <pointer:34> more ) ) ) :ARG3 ( <pointer:35> some ) ) :location ( <pointer:36> coast ) ) :mod ( <pointer:37> even ) ) ) </g> " #,', 'hypothesis': ' ( <pointer:0> and :op1 ( <pointer:1> protect-01 :ARG0 ( <pointer:2> authority ) :ARG1 ( <pointer:3> area :ARG1-of ( <pointer:4> develop-02 :polarity - ) :ARG1-of ( <pointer:5> remain-01 ) ) :ARG3 ( <pointer:6> preserve-01 :ARG0 ( <pointer:7> nature ) :ARG1 <pointer:3> ) ) :op2 ( <pointer:8> proclaim-01 :ARG0 <pointer:2> :ARG1 ( <pointer:9> off-limits : prep-to ( <pointer:10> construct-01 :mod ( <pointer:11> all ) ) :domain <pointer:3> ) ) :op3 ( <pointer:12> blow-up-06 :ARG0 <pointer:2> :ARG1 ( <pointer:13> blemish-01 :ARG1 ( <pointer:14> hotel ) :ARG1-of ( <pointer:15> include-91 :ARG2 ( <pointer:16> blemish-01 :ARG1 ( <pointer:17> hotel ) :ARG1-of ( <pointer:18> have-degree-91 :ARG2 ( <pointer:19> look-02 :polarity - :ARG0 <pointer:16> ) :ARG3 ( <pointer:20> more ) ) ) :ARG3 ( <pointer:21> some ) ) :location ( <pointer:22> coast ) ) :mod ( <pointer:23> even ) ) :ARG1-of ( <pointer:24> cause-01 :ARG0 ( <pointer:25> fear-01 :ARG0 <pointer:2> :ARG1 ( <pointer:26> leave-12 :ARG0 ( <pointer:27> and :op1 ( <pointer:28> overdevelopment ) :op2 ( <pointer:29> and :op1 ( <pointer:30> trend ) :op2 ( <pointer:31> taste ) :ARG1-of ( <pointer:32> new-01 ) :topic ( <pointer:33> tourism :mod ( <pointer:34> international ) ) ) ) :ARG1 <pointer:3> :ARG2 ( <pointer:35> behind ) ) ) ) ) </g>
    text_to_encode = '<t> I want to ask a question. </t>'
    x = tokenizer(y, truncation=True, padding='max_length', add_special_tokens=True)["input_ids"]
    print("bla:", tokenizer.convert_tokens_to_ids('</g>'))
    enc = tokenizer.encode_plus(text_to_encode,max_length=128,add_special_tokens=False,return_token_type_ids=False,return_attention_mask=False)['input_ids']
    print(tokenizer.convert_ids_to_tokens(enc))
    print(tokenizer.convert_ids_to_tokens(enc))
    print(x)
    print(x)
    """
    sick_f = "../data/SICK/SICK.txt"
    naturalistic_f = "C:/Users/phMei/Projekte/transitivity/naturalistic/train.tsv"
    sick_data = read_data(sick_f)
    nat_data = read_data(naturalistic_f)
    #print(nat_data)
    extracted = get_sick_instances(nat_data)
    #extracted = get_instances_by_id(sick_data, nat_data)
    #print(extracted)
    count_labels(extracted)
    df = pd.DataFrame(extracted, columns = ["pair_ID","sentence_A","sentence_B","entailment_label","relatedness_score",
                                            "entailment_AB","entailment_BA","sentence_A_original","sentence_B_original",
                                            "sentence_A_dataset","sentence_B_dataset","SemEval_set"])
    print(extracted)
    print(len(extracted))
    df.to_csv("extracted_sick_instances.csv", index=False)
    #"""