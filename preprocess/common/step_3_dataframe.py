import pandas as pd





if __name__ == "__main__":
    pos = pd.read_csv("../sick/commonalities_step2_common.csv")
    #neg = pd.read_csv("sick/commonalities_step2_neg.csv")

    pos_idx = pos["index"].to_list()
    #neg_idx = neg["index"].to_list()

    joint_pos = pd.read_csv("complete_joint_step3.csv")
    #joint_neg =pd.read_csv("joint_step3_neg.csv")
    #joint_pos.append(joint_neg, ignore_index=True)

    text_pos = pd.read_csv("complete_text_step3.csv")
    #text_neg =pd.read_csv("sick/text_step3_neg.csv")


    joint_pos_ex = joint_pos.loc[pos_idx]
    #oint_neg_ex = joint_neg.loc[neg_idx]

    text_pos_ex = text_pos.loc[pos_idx]
    #ext_neg_ex = text_neg.loc[neg_idx]


    #print(joint_pos_ex["label"].to_list())
    #print(joint_neg_ex["label"].to_list())

    joint_pos_ex.to_csv("step_3_joint_extractions_common.csv", index=False)
    #joint_neg_ex.to_csv("step_3_joint_extractions_neg.csv", index=False)

    text_pos_ex.to_csv("step_3_text_extractions_common.csv", index=False)
    #text_neg_ex.to_csv("step_3_text_extractions_neg.csv", index=False)




