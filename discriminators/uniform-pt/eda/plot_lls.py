import numpy as np
import matplotlib.pyplot as plt


def read_og(file_path):
    """
    @param file_path: some <IDX>_og.npy file
    """
    return np.load(file_path)

def read_pgrid(file_path):
    """
    @param file_path: some <IDX>_pgrid.npy file
    """
    return np.load(file_path)

def hist_per_segment(storyname, pgrid, ll_og):
    ll_og = ll_og.squeeze()

    n_perturbs = pgrid.shape[0]
    n_segments = pgrid.shape[1]
    N_BINS_DESIRED = 15

    p_min = np.min(pgrid)
    p_max = np.max(pgrid)
    bins = np.linspace(p_min, p_max, N_BINS_DESIRED+1)

    fig, axs = plt.subplots(n_segments)
    
    for i in range(n_segments):
        axs[i].hist(pgrid[:,i], bins=bins, range=(p_min, p_max), label='perturbations')
        axs[i].axvline(x=ll_og[i], linewidth=2, color='black', label='original')
        
    
    fig.suptitle(f"Likelihoods for story {storyname}")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.show()

STORY_NUMBER = 13
TEST_FILE_OG = f"LLM_results/{STORY_NUMBER}_og.npy"
TEST_FILE_PERTURBS = f"LLM_results/{STORY_NUMBER}_pgrid.npy"

ll_og = read_og(TEST_FILE_OG)
pgrid = read_pgrid(TEST_FILE_PERTURBS)
hist_per_segment(f"{STORY_NUMBER}", pgrid, ll_og)