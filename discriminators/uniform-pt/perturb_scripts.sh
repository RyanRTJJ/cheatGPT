python perturb.py --mask_filling_model t5-3b --folder_to_perturb inputs/human/squad --ll_save_loc results/gpt2-medium-scorer/human/squad --folder_of_perturbs perturbations/human/squad --mode perturb --num_perturbations 50
python perturb.py --mask_filling_model t5-3b --folder_to_perturb inputs/human/xsum --ll_save_loc results/gpt2-medium-scorer/human/xsum --folder_of_perturbs perturbations/human/xsum --mode perturb --num_perturbations 50
python perturb.py --mask_filling_model t5-3b --folder_to_perturb inputs/human/writing --ll_save_loc results/gpt2-medium-scorer/human/writing --folder_of_perturbs perturbations/human/writing --mode perturb --num_perturbatio\
ns 50

python perturb.py --mask_filling_model t5-3b --folder_to_perturb inputs/LLM/squad --ll_save_loc results/gpt2-medium-scorer/LLM/squad --folder_of_perturbs perturbations/LLM/squad --mode perturb --num_perturbations 50
python perturb.py --mask_filling_model t5-3b --folder_to_perturb inputs/LLM/xsum --ll_save_loc results/gpt2-medium-scorer/LLM/xsum --folder_of_perturbs perturbations/LLM/xsum --mode perturb --num_perturbations 50
python perturb.py --mask_filling_model t5-3b --folder_to_perturb inputs/LLM/writing --ll_save_loc results/gpt2-medium-scorer/LLM/writing --folder_of_perturbs perturbations/LLM/writing --mode perturb --num_perturbations 50

