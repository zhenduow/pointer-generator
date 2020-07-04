from sklearn.metrics import ndcg_score
import glob
from rouge import Rouge

original_files = glob.glob("log/pretrained_model_tf1.2.1/decode_test_400maxenc_1beam_10mindec_120maxdec_ckpt-238410/reference/*.txt")

references = []
for f in sorted(original_files):
    references.append(' '.join(open(f).readlines()))

irrelevant = []
for i in range(len(references)):
    reference = references[i]
    max_score = 0
    competitive = ''
    scores = {}
    for j in range(len(references)):
        rouge = Rouge()
        if i != j:
            hypothesis = references[j]
            scores = rouge.get_scores(hypothesis, reference)
            if scores[0]['rouge-l']['f'] > max_score:
                max_score = scores[0]['rouge-l']['f']
                competitive = hypothesis
        
    irrelevant.append(competitive)
with open('irrelevant.target', 'w') as f:
    for line in irrelevant:
        f.write(line)