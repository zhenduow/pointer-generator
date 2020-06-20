from sklearn.metrics import ndcg_score
import glob

original_files = glob.glob("log/pretrained_model_tf1.2.1/decode_test_400maxenc_1beam_10mindec_120maxdec_ckpt-238410/decoded/*.txt")
grammar_files = glob.glob("log/grammar/decode_test_400maxenc_1beam_10mindec_120maxdec_ckpt-238410/decoded/*.txt")
syntax_files = glob.glob("log/syntax/decode_test_400maxenc_1beam_10mindec_120maxdec_ckpt-238410/decoded/*.txt")
semantic_files = glob.glob("log/semantic/decode_test_400maxenc_1beam_10mindec_120maxdec_ckpt-238410/decoded/*.txt")

original_score, grammar_score, syntax_score, semantic_score = [], [], [], []


for f in sorted(original_files):
    original_score.append(float(open(f).readlines()[-1]))
for f in sorted(grammar_files):
    grammar_score.append(float(open(f).readlines()[-1]))
for f in sorted(syntax_files):
    syntax_score.append(float(open(f).readlines()[-1]))
for f in sorted(semantic_files):
    semantic_score.append(float(open(f).readlines()[-1]))
            
ndcg_total = ndcg_score([[5,4,2,1]]*len(original_score),list(zip(original_score,grammar_score,syntax_score, semantic_score)))
print(ndcg_total)



