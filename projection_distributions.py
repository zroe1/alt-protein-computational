import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForMaskedLM
import requests
from io import BytesIO
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data():
    """Load and preprocess the protein data."""
    query_url = "https://rest.uniprot.org/uniprotkb/stream?compressed=true&fields=accession%2Csequence%2Ccc_subcellular_location&format=tsv&query=%28%28organism_id%3A9606%29%20AND%20%28reviewed%3Atrue%29%20AND%20%28length%3A%5B80%20TO%20500%5D%29%29"
    uniprot_request = requests.get(query_url)
    bio = BytesIO(uniprot_request.content)
    df = pd.read_csv(bio, compression='gzip', sep='\t')
    
    cytosolic = df['Subcellular location [CC]'].str.contains("Cytosol") | df['Subcellular location [CC]'].str.contains("Cytoplasm")
    membrane = df['Subcellular location [CC]'].str.contains("Membrane") | df['Subcellular location [CC]'].str.contains("Cell membrane")
    
    membrane_df = df[membrane & ~cytosolic]
    cytosolic_df = df[cytosolic & ~membrane]
    
    return membrane_df["Sequence"].tolist(), cytosolic_df["Sequence"].tolist()

def get_hidden_states(model, tokenizer, sequences, layer_num):
    """Get hidden states for a set of sequences."""
    tokens = [tokenizer(s, return_tensors="pt") for s in sequences[:200]]  # Use 100 samples
    hidden_states = []
    
    for example in tokens:
        inputs = example
        inputs['output_hidden_states'] = True
        outputs = model.forward(**inputs)
        layer = outputs.hidden_states[layer_num].detach().numpy()
        hidden_states.append(layer.mean(axis=1).squeeze(0))
            
    return hidden_states

def calculate_steering_vector(membrane_states, cytosolic_states):
    """Calculate steering vector from hidden states."""
    membrane_avg = np.mean(membrane_states, axis=0)
    cytosolic_avg = np.mean(cytosolic_states, axis=0)
    return membrane_avg - cytosolic_avg

def project_vector(vector, steering_vector):
    """Project a vector onto the steering vector."""
    return np.dot(vector, steering_vector) / np.linalg.norm(steering_vector)

def find_optimal_thresholds(membrane_projections, cytosolic_projections):
    # Prepare data for ROC and other analyses
    X = np.concatenate([membrane_projections, cytosolic_projections])
    y = np.concatenate([np.ones(len(membrane_projections)), 
                       np.zeros(len(cytosolic_projections))])

    
    # Method 2: Grid search for accuracy
    sorted_projections = np.sort(X)
    accuracies = []
    thresholds = []
    
    for threshold in sorted_projections:
        predictions = (X > threshold).astype(int)
        acc = accuracy_score(y, predictions)
        accuracies.append(acc)
        thresholds.append(threshold)
    
    best_idx = np.argmax(accuracies)
    return thresholds[best_idx]

def plot_projection_distributions(membrane_projections, cytosolic_projections, save_path='projection_distributions.png'):
    """Create a detailed visualization of the projection distributions."""
    plt.figure(figsize=(12, 6))
    
    # # Calculate the threshold
    # threshold = (np.mean(membrane_projections) + np.mean(cytosolic_projections)) / 2
    threshold = find_optimal_thresholds(membrane_projections, cytosolic_projections)
    
    # Create the distribution plot
    sns.kdeplot(data=membrane_projections, color='blue', label='Membrane', alpha=0.6)
    sns.kdeplot(data=cytosolic_projections, color='red', label='Cytosolic', alpha=0.6)
    
    # Add histograms with transparency
    plt.hist(membrane_projections, bins=20, color='blue', alpha=0.3, density=True)
    plt.hist(cytosolic_projections, bins=20, color='red', alpha=0.3, density=True)
    
    # Add vertical lines for means and threshold
    plt.axvline(np.mean(membrane_projections), color='blue', linestyle='--', alpha=0.8, label='Membrane Mean')
    plt.axvline(np.mean(cytosolic_projections), color='red', linestyle='--', alpha=0.8, label='Cytosolic Mean')
    plt.axvline(threshold, color='green', linestyle='-', alpha=0.8, label='Decision Threshold')
    
    # Customize the plot
    plt.title('Distribution of Protein Projections onto Steering Vector', fontsize=12, pad=20)
    plt.xlabel('Projection Value', fontsize=10)
    plt.ylabel('Density', fontsize=10)
    plt.legend(title='Protein Type', title_fontsize=10, fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # Add summary statistics as text
    stats_text = (
        f'Membrane Mean: {np.mean(membrane_projections):.3f}\n'
        f'Cytosolic Mean: {np.mean(cytosolic_projections):.3f}\n'
        f'Threshold: {threshold:.3f}\n'
        f'Membrane Std: {np.std(membrane_projections):.3f}\n'
        f'Cytosolic Std: {np.std(cytosolic_projections):.3f}'
    )
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
    model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t12_35M_UR50D")
    
    # Load and split data
    print("Loading and preprocessing data...")
    membrane_sequences, cytosolic_sequences = load_data()
    membrane_train, membrane_test = train_test_split(membrane_sequences, test_size=0.2, random_state=42)
    cytosolic_train, cytosolic_test = train_test_split(cytosolic_sequences, test_size=0.2, random_state=42)
    
    # Use layer 6 as in original code
    LAYER_NUM = 6
    
    # Get hidden states
    print("Computing hidden states...")
    membrane_states = get_hidden_states(model, tokenizer, membrane_train, LAYER_NUM)
    cytosolic_states = get_hidden_states(model, tokenizer, cytosolic_train, LAYER_NUM)
    
    # Calculate steering vector
    print("Calculating steering vector...")
    steering_vector = calculate_steering_vector(membrane_states, cytosolic_states)
    
    # Get test hidden states and calculate projections
    print("Computing projections...")
    membrane_test_states = get_hidden_states(model, tokenizer, membrane_test, LAYER_NUM)
    cytosolic_test_states = get_hidden_states(model, tokenizer, cytosolic_test, LAYER_NUM)
    
    membrane_projections = [project_vector(state, steering_vector) for state in membrane_test_states]
    cytosolic_projections = [project_vector(state, steering_vector) for state in cytosolic_test_states]
    
    # Create visualization
    print("Generating plot...")
    plot_projection_distributions(membrane_projections, cytosolic_projections)

if __name__ == "__main__":
    main()