from sklearn.metrics import ndcg_score
import glob
from collections import Counter
import numpy as np

original_files = glob.glob("log/pretrained_model_tf1.2.1/decode_test_400maxenc_1beam_10mindec_120maxdec_ckpt-238410/decoded/*.txt")
grammar_files = glob.glob("log/grammar/decode_test_400maxenc_1beam_10mindec_120maxdec_ckpt-238410/decoded/*.txt")
syntax_files = glob.glob("log/syntax/decode_test_400maxenc_1beam_10mindec_120maxdec_ckpt-238410/decoded/*.txt")
semantic_files = glob.glob("log/semantic/decode_test_400maxenc_1beam_10mindec_120maxdec_ckpt-238410/decoded/*.txt")
lead3_files = glob.glob("log/lead3/decode_test_400maxenc_1beam_10mindec_120maxdec_ckpt-238410/decoded/*.txt")
irrelevant_files = glob.glob("log/irrelevant/decode_test_400maxenc_1beam_10mindec_120maxdec_ckpt-238410/decoded/*.txt")

original_score, grammar_score, syntax_score, semantic_score, lead3_score, irrelevant_score = [], [], [], [], [], []

for f in sorted(original_files):
    original_score.append(float(open(f).readlines()[-1]))
for f in sorted(grammar_files):
    grammar_score.append(float(open(f).readlines()[-1]))
for f in sorted(syntax_files):
    syntax_score.append(float(open(f).readlines()[-1]))
for f in sorted(semantic_files):
    semantic_score.append(float(open(f).readlines()[-1]))
for f in sorted(lead3_files):
    lead3_score.append(float(open(f).readlines()[-1]))
for f in sorted(irrelevant_files):
    score = float(open(f).readlines()[-1])
    if score == float("-inf"):
        irrelevant_score.append(-100)
    else:
        irrelevant_score.append(score)

print("PTGEN")

print("original-irrelevant")
ndcg_total = ndcg_score([[1,0]]*len(original_score),list(zip(original_score, irrelevant_score)), k=1)
print(ndcg_total)

print("original-lead3")
ndcg_total = ndcg_score([[1,0]]*len(original_score),list(zip(original_score, lead3_score)), k=1)
print(ndcg_total)

print("original-lead3-irrelevant")
ndcg_total = ndcg_score([[1,0,0]]*len(original_score),list(zip(original_score,lead3_score, irrelevant_score)), k=1)
print(ndcg_total)

print("original-grammar-syntax-semantic")
ndcg_total = ndcg_score([[1,0,0,0]]*len(original_score),list(zip(original_score,grammar_score,syntax_score, semantic_score)), k =1)
print(ndcg_total)

print("original-grammar-syntax-semantic-irrelevant")
ndcg_total = ndcg_score([[1,0,0,0,0]]*len(original_score),list(zip(original_score,grammar_score,syntax_score, semantic_score, irrelevant_score)), k=1)
print(ndcg_total)





