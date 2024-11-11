import olga
import olga.generation_probability as pgen
import olga.load_model as load_model
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import logomaker




####################
# Helper functions #
####################


## Load OLGA Model
params_file_name = f'{olga.__path__[0]}/default_models/human_T_beta/model_params.txt'
marginals_file_name = f'{olga.__path__[0]}/default_models/human_T_beta/model_marginals.txt'
V_anchor_pos_file =f'{olga.__path__[0]}/default_models/human_T_beta/V_gene_CDR3_anchors.csv'
J_anchor_pos_file = f'{olga.__path__[0]}/default_models/human_T_beta/J_gene_CDR3_anchors.csv'
genomic_data = load_model.GenomicDataVDJ()
genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
generative_model = load_model.GenerativeModelVDJ()
generative_model.load_and_process_igor_model(marginals_file_name)

pgen_model = pgen.GenerationProbabilityVDJ(generative_model, genomic_data)


def logo_plot_from_sequences(sequences, title_prefix, top_k=4):
    """
    Creates a Sequence Logo Plot for the given list of strings using logomaker.
    
    Parameters:
    sequences (list): A list of strings of variable sequence length.
    top_k (int): The number of top characters to display in the logo plot.
    """
    # Calculate the length of the longest sequence
    max_length = max(len(seq) for seq in sequences)
    
    # Pad the sequences to have the same length
    padded_sequences = [seq.ljust(max_length, '|') for seq in sequences]

    # Count the occurrences of characters at each position
    position_counters = [Counter() for _ in range(max_length)]
    for seq in padded_sequences:
        for i, char in enumerate(seq):
            position_counters[i][char] += 1
    
    # Normalize the counts and sort by occurrences
    for counter in position_counters:
        total = sum(counter.values())
        for char in counter:
            counter[char] /= total
    
    # Create a dataframe for the logo plot
    logo_df = pd.DataFrame(columns=["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", 'X', "|"], index=range(max_length)).fillna(0)
    for index, counter in enumerate(position_counters):
        sorted_chars = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:top_k]
        for char, freq in sorted_chars:
            if char == '|':
                continue
            else:
                logo_df.at[index, char] = freq
    
    # Create the sequence logo plot using logomaker
    logo = logomaker.Logo(logo_df, color_scheme='weblogo_protein')
    plt.xlabel('Position')
    plt.xlim(-1, 25)
    plt.ylabel('Frequency')
    plt.title(f'{title_prefix} Logo Plot')
    plt.show()
