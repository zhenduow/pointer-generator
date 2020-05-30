#!/bin/bash
#SBATCH -A zhenduow
#SBATCH -n 4

#SBATCH --output=job.%J.out
#SBATCH --partition=titan-giant
# And finally run the job​

#srun python2 run_summarization.py --mode=decode --data_path=finished_files/test.bin --vocab_path=finished_files/vocab --log_root=log --exp_name=pretrained_model_tf1.2.1 --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1 --beam_size=1
srun python2 run_summarization.py --mode=decode --data_path=finished_files/test.bin --vocab_path=finished_files/vocab --log_root=log --exp_name=grammar_tweek --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1 --beam_size=1
#srun python2 run_summarization.py --mode=decode --data_path=finished_files/test.bin --vocab_path=finished_files/vocab --log_root=log --exp_name=syntax_change --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1 --beam_size=1
#srun python2 run_summarization.py --mode=decode --data_path=finished_files/test.bin --vocab_path=finished_files/vocab --log_root=log --exp_name=semantic_change --max_enc_steps=400 --max_dec_steps=120 --coverage=1 --single_pass=1 --beam_size=1

