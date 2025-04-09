import os

import numpy as np
import jieba
from elmoformanylangs import Embedder

prefix = os.path.abspath(os.path.join(os.getcwd(), "."))
elmo = Embedder('zhs.model')

topics = ['positive', 'neutral', 'negative']
answers = {}
text_features = []
text_targets = []

def extract_features(text_features, text_targets, path):
    for index in range(114):
        if os.path.isdir(os.path.join(prefix, f"{path}_{str(index+1)}")):
            answers[index+1] = []
            for topic in topics:
                
                with open(os.path.join(prefix, f"{path}_{str(index+1)}", '%s.txt'%(topic)) ,'r') as f:
                    lines = f.readlines()[0]
                    # seg_text = seg.cut(lines) 
                    # seg_text = thu1.cut(lines)
                    # seg_text_iter = HanLP.segment(lines) 
                    seg_text_iter = jieba.cut(lines, cut_all=False) 
                    answers[index+1].append([item for item in seg_text_iter])
                    # answers[dir].append(seg_text)
            with open(os.path.join(prefix, '{1}_{0}/new_label.txt'.format(index+1, path))) as fli:
                target = float(fli.readline())
            # text_targets.append(1 if target >= 53 else 0)
            text_targets.append(target)
            a = elmo.sents2elmo(answers[index+1])
            # for item in elmo.sents2elmo(answers[index+1]):
            #     text_feature = np.array(item).mean(axis=0)
            #     text_features.append(text_feature)
            text_features.append([np.array(item).mean(axis=0) for item in elmo.sents2elmo(answers[index+1])])

extract_features(text_features, text_targets, 'EATD-Corpus/t')
extract_features(text_features, text_targets, 'EATD-Corpus/v')

print("Saving npz file locally...")
os.makedirs("Features/TextWhole", exist_ok=True)
np.savez('Features/TextWhole/whole_samples_reg_avg.npz', text_features)
np.savez('Features/TextWhole/whole_labels_reg_avg.npz', text_targets)
print()


"""
v_17  only have the  label.txt; negative.txt; negative.wav
"""

