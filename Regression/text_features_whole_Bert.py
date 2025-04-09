import os

import numpy as np
import jieba
# from elmoformanylangs import Embedder
from transformers import BertTokenizer, BertModel
import torch

prefix = os.path.abspath(os.path.join(os.getcwd(), "."))
# elmo = Embedder("zhs.model")

topics = ["positive", "neutral", "negative"]
answers = {}
text_features = []
text_targets = []

# 加载中文BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
bert = BertModel.from_pretrained("bert-base-chinese")
bert.eval()  # 不训练的话，记得设置为 eval 模式


def extract_features(text_features, text_targets, path):
    for index in range(114):
        if os.path.isdir(os.path.join(prefix, f"{path}_{str(index+1)}")):
            answers[index + 1] = []
            for topic in topics:

                with open(
                    os.path.join(prefix, f"{path}_{str(index+1)}", "%s.txt" % (topic)),
                    "r",
                ) as f:
                    lines = f.readlines()[0]
                    inputs = tokenizer(lines, return_tensors="pt", truncation=True, max_length=256)
                    with torch.no_grad():
                        outputs = bert(**inputs)
                    # 取 [CLS] token 的输出作为句向量
                    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).numpy()
                    answers[index + 1].append(cls_embedding)
            with open(os.path.join(prefix, '{1}_{0}/new_label.txt'.format(index+1, path))) as fli:
                target = float(fli.readline())
            text_targets.append(target)

            temp_feats =[elem for elem in answers[index + 1]]
            text_features.append(np.vstack((temp_feats[0], temp_feats[1], temp_feats[2])))


extract_features(text_features, text_targets, "EATD-Corpus/t")
extract_features(text_features, text_targets, "EATD-Corpus/v")

print("Saving npz file locally...")
os.makedirs("Features/TextWhole", exist_ok=True)
np.savez("Features/TextWhole/whole_samples_reg_avg_bert.npz", text_features)
np.savez("Features/TextWhole/whole_labels_reg_avg_bert.npz", text_targets)
print()


"""
v_17  only have the  label.txt; negative.txt; negative.wav
"""
