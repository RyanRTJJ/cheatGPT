import numpy as np
import os
import pickle
import argparse
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt

"""
Sample command:
python experiments.py --scoring_model_name gpt2-xl --source_model_name gpt2-xl --human_samples_folder results/gpt2-xl-scorer/human/squad --LLM_samples_folder results/gpt2-xl-scorer/LLM/squad
"""

class Conductor:
    """
    conducts the experiment
    """
    def __init__(self, args):
        # ASSERT >:(
        assert args.scoring_model_name != None, "Please provide a scoring model name. Should be huggingface-compatible, e.g. 'EleutherAI/gpt-j-6B'"
        assert args.source_model_name != None, "Please provide a source model name. Should be huggingface-compatible, e.g. 'EleutherAI/gpt-j-6B'"
        assert args.human_samples_folder != None, "Where are your human texts located?"
        assert args.LLM_samples_folder != None, "Where are your LLM texts located?"

        """
        Glossary of attributes
        ----------------------
        * scoring_model_name: something huggingface-compatible like 'gpt2-xl' or 'EleutherAI/gpt-j-6B'
        * source_model_name: same as above
        * human_samples_folder: something like 'gpt2-xl-scorer/human/squad'. Should contain all the _pgrid.npy and _og.npy files.
        * LLM_samples_folder: the LLM analog of above, e.g. 'gpt2-xl-scorer/LLM/squad'.
        """
        self.SCORING_MODEL_NAME = args.scoring_model_name
        self.SOURCE_MODEL_NAME = args.source_model_name
        self.HUMAN_SAMPLES_FOLDER = args.human_samples_folder
        self.LLM_SAMPLES_FOLDER = args.LLM_samples_folder

        print("")
        print("+------------------------------------+")
        print("| The Conductor has come on stage!!! |")
        print("+------------------------------------+")
        print(f"scoring_model:        {self.SCORING_MODEL_NAME}")
        print(f"source_model:         {self.SOURCE_MODEL_NAME}")
        print(f"human_samples are in: {self.HUMAN_SAMPLES_FOLDER}")
        print(f"LLM_samples are in:   {self.LLM_SAMPLES_FOLDER}")

    def get_roc_metrics(self, human_statistic, LLM_statistic):
        fpr, tpr, _ = roc_curve([0] * len(human_statistic) + [1] * len(LLM_statistic), np.concatenate((human_statistic, LLM_statistic)))
        roc_auc = auc(fpr, tpr)
        return fpr.tolist(), tpr.tolist(), float(roc_auc)
        
    def get_precision_recall_metrics(self, human_statistic, LLM_statistic):
        precision, recall, _ = precision_recall_curve([0] * len(human_statistic) + [1] * len(LLM_statistic), np.concatenate((human_statistic, LLM_statistic)))
        pr_auc = auc(recall, precision)
        return precision.tolist(), recall.tolist(), float(pr_auc)

    def load_and_synthesize(self):
        """
        +---------------+
        | Functionality |
        +---------------+
        0) If the source_model_name has a '/' in it, we replace it with '_', and use this
            to filter for the data files that have the modified source_model_name in them.
        1) Load in all the data and compute segment-wise means and variances
        2) Compute the Z score of each segment's og log-likelihood (ll) w.r.t its segment's 
            perturbed ll distribution; what we call the "first-order statistic"
        3) Compute the variance of Z score variances, what we call the "second-order statistic"
        """
        # Step 0: Filter for files
        src_LLM_name = self.SOURCE_MODEL_NAME.replace('/', '_')
        human_files = os.listdir(self.HUMAN_SAMPLES_FOLDER)
        human_files_og = [f"{self.HUMAN_SAMPLES_FOLDER}/{file}" for file in human_files if file.startswith(src_LLM_name) and file.endswith('og.npy')]
        human_files_pgrid = [f"{self.HUMAN_SAMPLES_FOLDER}/{file}" for file in human_files if file.startswith(src_LLM_name) and file.endswith('pgrid.npy')]
        LLM_files = os.listdir(self.LLM_SAMPLES_FOLDER)
        LLM_files_og = [f"{self.LLM_SAMPLES_FOLDER}/{file}" for file in LLM_files if file.startswith(src_LLM_name) and file.endswith('og.npy')]
        LLM_files_pgrid = [f"{self.LLM_SAMPLES_FOLDER}/{file}" for file in LLM_files if file.startswith(src_LLM_name) and file.endswith('pgrid.npy')]

        human_files_og.sort()
        human_files_pgrid.sort()
        LLM_files_og.sort()
        LLM_files_pgrid.sort()

        print(f"\nHow many files we got:")
        print("-" * 30)
        print(f"human_files_og: {len(human_files_og)}")
        print(f"human_files_pgrid: {len(human_files_pgrid)}")
        print(f"LLM_files_og: {len(LLM_files_og)}")
        print(f"LLM_files_pgrid: {len(LLM_files_pgrid)}")

        # Step 1: Load in all the probability matrices
        human_lls_og = []       # A list of 1D arrays of shape (N_SEGMENTS)
        human_lls_pgrid = []    # A list of 2D matrices of shape (N_PERTURBS, N_SEGMENTS).
        LLM_lls_og = []         # Same
        LLM_lls_pgrid = []      # Same

        for human_og_file, human_pgrid_file in zip(human_files_og, human_files_pgrid):
            ll_og = np.load(human_og_file).squeeze()
            ll_pgrid = np.load(human_pgrid_file)
            human_lls_og.append(ll_og)
            human_lls_pgrid.append(ll_pgrid)
        for LLM_og_file, LLM_pgrid_file in zip(LLM_files_og, LLM_files_pgrid):
            ll_og = np.load(LLM_og_file).squeeze()
            ll_pgrid = np.load(LLM_pgrid_file)
            LLM_lls_og.append(ll_og)
            LLM_lls_pgrid.append(ll_pgrid)
        
        # For each story, calculate inter-perturbation (i.e. segment-wise) mean and variance of perturb likelihoods
        human_means = []
        human_variances = []
        for pgrid in human_lls_pgrid:
            human_means.append(np.mean(pgrid, 0))           # a (N_SEGMENTS,) array
            human_variances.append(np.var(pgrid, 0))        # a (N_SEGMENTS,) array
        
        LLM_means = []
        LLM_variances = []
        for pgrid in LLM_lls_pgrid:
            LLM_means.append(np.mean(pgrid, 0))           # a (N_SEGMENTS,) array
            LLM_variances.append(np.var(pgrid, 0))        # a (N_SEGMENTS,) array
        
        # Step 2: Compute Z_scores
        human_og_Z_scores = []
        for idx, human_ll_og in enumerate(human_lls_og):
            Z = (human_ll_og - human_means[idx]) / np.sqrt(human_variances[idx])
            human_og_Z_scores.append(Z)

        LLM_og_Z_scores = []
        for idx, LLM_ll_og in enumerate(LLM_lls_og):
            Z = (LLM_ll_og - LLM_means[idx]) / np.sqrt(LLM_variances[idx])
            LLM_og_Z_scores.append(Z)
        
        # Step 3: Compute the variance of Z scores, i.e., the:
        # inter-segment variance of Z scores across a story
        # Want to show that inter_seg_var_LLM > inter_seg_var_human
        inter_seg_vars_human = np.array([np.var(Z_score) for Z_score in human_og_Z_scores])
        inter_seg_vars_LLM = np.array([np.var(Z_score) for Z_score in LLM_og_Z_scores])

        # Step 4: Give a summary on the distribution of these variances
        mean_var_human = np.mean(inter_seg_vars_human)
        var_var_human = np.var(inter_seg_vars_human)
        mean_var_LLM = np.mean(inter_seg_vars_LLM)
        var_var_LLM = np.var(inter_seg_vars_LLM)

        print(".________________________________.")
        print("|+------------------------------+|")
        print("||           RESULTS            ||")
        print("|+------------------------------+|")
        print("|                                |")
        print("|      _--_                      |")
        print("|     /    \     __--__          |")
        print("|    /      \ _/        \_       |")
        print("|  _'     _.-^._          '-._   |")
        print("| ------------------------------ |")
        print(f"| LLM mean of variances  : {mean_var_LLM:.3f} |")
        print(f"| LLM var of variances   : {var_var_LLM:.3f} |")
        print(f"| human mean of variances: {mean_var_human:.3f} |")
        print(f"| human var of variances : {var_var_human:.3f} |")
        print("+--------------------------------+")

        # Step 5: Compute metrics
        fpr, tpr, roc_auc = self.get_roc_metrics(inter_seg_vars_human, inter_seg_vars_LLM)
        p, r, pr_auc = self.get_precision_recall_metrics(inter_seg_vars_human, inter_seg_vars_LLM)
        # Report them
        print(f"ROC AUC: {roc_auc:.3f}, PR AUC: {pr_auc:.3f}")

        human_test = [np.mean(Z) for Z in human_og_Z_scores]
        LLM_test = [np.mean(Z) for Z in LLM_og_Z_scores]
        fpr, tpr, roc_auc = self.get_roc_metrics(human_test, LLM_test)
        p, r, pr_auc = self.get_precision_recall_metrics(human_test, LLM_test)
        print(f"ROC AUC: {roc_auc:.3f}, PR AUC: {pr_auc:.3f}")

        human_og_Z_scores = np.concatenate(human_og_Z_scores)
        LLM_og_Z_scores = np.concatenate(LLM_og_Z_scores)


        return human_og_Z_scores, LLM_og_Z_scores, inter_seg_vars_human, inter_seg_vars_LLM
    
    def plot_hist(self, human_stats, LLM_stats):
        plt.figure()
        bins = np.linspace(min(np.min(human_stats), np.min(LLM_stats)), max(np.max(human_stats), np.max(LLM_stats)), 50+1)
        plt.hist(human_stats, bins=bins, color='blue', alpha=0.7, label='human')
        plt.hist(LLM_stats, bins=bins, color='orange', alpha=0.7, label='LLM')
        plt.legend()
        plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scoring_model_name', type=str, default=None)
    parser.add_argument('--source_model_name', type=str, default=None)
    parser.add_argument('--human_samples_folder', type=str, default=None)
    parser.add_argument('--LLM_samples_folder', type=str, default=None)
    args = parser.parse_args()

    conductor = Conductor(args)
    human_Z_scores, LLM_Z_scores, inter_seg_vars_human, inter_seg_vars_LLM = conductor.load_and_synthesize()
    print(f"inter_seg_vas_human.length: {inter_seg_vars_human.shape}")
    conductor.plot_hist(human_Z_scores.reshape(-1), LLM_Z_scores.reshape(-1))
    conductor.plot_hist(inter_seg_vars_human, inter_seg_vars_LLM)
