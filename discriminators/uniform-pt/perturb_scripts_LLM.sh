python perturb.py --mask_filling_model t5-large --folder_to_perturb inputs/LLM/squad --ll_save_loc results/gpt2-medium-scorer/LLM/squad --folder_of_perturbs perturbations/LLM/squad --mode perturb --num_perturbations 50
python perturb.py --mask_filling_model t5-large --folder_to_perturb inputs/LLM/xsum --ll_save_loc results/gpt2-medium-scorer/LLM/xsum --folder_of_perturbs perturbations/LLM/xsum --mode perturb --num_perturbations 50
python perturb.py --mask_filling_model t5-large --folder_to_perturb inputs/LLM/writing --ll_save_loc results/gpt2-medium-scorer/LLM/writing --folder_of_perturbs perturbations/LLM/writing --mode perturb --num_perturbations 50