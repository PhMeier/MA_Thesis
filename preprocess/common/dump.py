import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv("../sick/step2_data/complete_text_step2.csv")
    x = set(df["complete_signature"].to_list())
    print(x)
    """
    pos = pd.read_csv("sick/step2_data/complete_text_step2.csv") #"sick/step2_data/pos_env_complete_sick_new.csv")
    label = pos["label"].to_list()
    df_dict = {"label":label}

    res = pd.DataFrame(df_dict, columns=["label"])
    res.to_csv("sick/step2_data/complete_step2_only_label.csv", index=True)
    """
    #sig = set(pos["complete_signature"].to_list())
    #print(sig)

    # join frames
    """
    pos = pd.read_csv("joint_step2_pos.csv")
    neg = pd.read_csv("joint_step2_neg.csv")
    pos = pos.append(neg, ignore_index=True)
    print(pos)
    pos.to_csv("sick/step2_data/complete_joint_step2.csv", index=True)
    """
    """
    pos = pd.read_csv("sick/step2_data/pos_env_complete_sick_new_tags.csv")
    neg = pd.read_csv("sick/step2_data/neg_env_complete_sick_new_tags.csv")
    pos = pos.append(neg, ignore_index=True)
    print(pos)
    pos.to_csv("sick/step2_data/complete_text_step2.csv", index=True)
    """

    """
    pos = pd.read_csv("joint_step3_pos.csv")
    neg = pd.read_csv("joint_step3_neg.csv")
    pos = pos.append(neg, ignore_index=True)
    print(pos)
    pos.to_csv("complete_joint_step3.csv", index=True)

    pos = pd.read_csv("sick/text_step3_pos.csv")
    neg = pd.read_csv("sick/text_step3_neg.csv")
    pos = pos.append(neg, ignore_index=True)
    print(pos)
    pos.to_csv("complete_text_step3.csv", index=True)
    """

    #res = pd.DataFrame(df_dict, columns=["label"])
    #res.to_csv("sick/step2_data/pos_step2_only_label.csv", index=True)
    #res.to_csv("neg_step3_only_label.csv", index=True)
