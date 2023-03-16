python detect_perturb_score.py --base_model_name gpt2-xl --dataset xsum --scoring_model_name gpt2-xl --mask_filling_model_name t5-large --perturb
python detect_perturb_score.py --base_model_name gpt2-xl --dataset writing --scoring_model_name gpt2-xl --mask_filling_model_name t5-large --perturb
python detect_perturb_score.py --base_model_name gpt2-xl --dataset squad --scoring_model_name gpt2-xl --mask_filling_model_name t5-large --perturb

python detect_perturb_score.py --base_model_name EleutherAI/gpt-j-6B --dataset xsum --scoring_model_name EleutherAI/gpt-j-6B --mask_filling_model_name t5-large --perturb
python detect_perturb_score.py --base_model_name EleutherAI/gpt-j-6B --dataset writing --scoring_model_name EleutherAI/gpt-j-6B --mask_filling_model_name t5-large --perturb
python detect_perturb_score.py --base_model_name EleutherAI/gpt-j-6B --dataset squad --scoring_model_name EleutherAI/gpt-j-6B --mask_filling_model_name t5-large --perturb

python detect_perturb_score.py --base_model_name EleutherAI_gpt-neo-2.7B --dataset xsum --scoring_model_name EleutherAI_gpt-neo-2.7B --mask_filling_model_name t5-large --perturb
python detect_perturb_score.py --base_model_name EleutherAI_gpt-neo-2.7B --dataset writing --scoring_model_name EleutherAI_gpt-neo-2.7B --mask_filling_model_name t5-large --perturb
python detect_perturb_score.py --base_model_name EleutherAI_gpt-neo-2.7B --dataset squad --scoring_model_name EleutherAI_gpt-neo-2.7B --mask_filling_model_name t5-large --perturb

python detect_perturb_score.py --base_model_name facebook_opt-2.7b --dataset xsum --scoring_model_name facebook_opt-2.7b --mask_filling_model_name t5-large --perturb
python detect_perturb_score.py --base_model_name facebook_opt-2.7b --dataset writing --scoring_model_name facebook_opt-2.7b --mask_filling_model_name t5-large --perturb
python detect_perturb_score.py --base_model_name facebook_opt-2.7b --dataset squad --scoring_model_name facebook_opt-2.7b --mask_filling_model_name t5-large --perturb

python detect_perturb_score.py --base_model_name gpt2-large --dataset xsum --scoring_model_name gpt2-large --mask_filling_model_name t5-large --perturb
python detect_perturb_score.py --base_model_name gpt2-large --dataset writing --scoring_model_name gpt2-large --mask_filling_model_name t5-large --perturb
python detect_perturb_score.py --base_model_name gpt2-large --dataset squad --scoring_model_name gpt2-large --mask_filling_model_name t5-large --perturb

python detect_perturb_score.py --base_model_name gpt2-medium --dataset xsum --scoring_model_name gpt2-medium --mask_filling_model_name t5-large --perturb
python detect_perturb_score.py --base_model_name gpt2-medium --dataset writing --scoring_model_name gpt2-medium --mask_filling_model_name t5-large --perturb
python detect_perturb_score.py --base_model_name gpt2-medium --dataset squad --scoring_model_name gpt2-medium --mask_filling_model_name t5-large --perturb

python detect_perturb_score.py --openai_model text-davinci-003 --dataset xsum --scoring_model_name gpt2 --mask_filling_model_name t5-large --perturb
python detect_perturb_score.py --openai_model text-davinci-003 --dataset writing --scoring_model_name gpt2 --mask_filling_model_name t5-large --perturb
python detect_perturb_score.py --openai_model text-davinci-003 --dataset squad --scoring_model_name gpt2 --mask_filling_model_name t5-large --perturb