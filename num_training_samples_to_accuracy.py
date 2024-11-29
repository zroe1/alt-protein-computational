import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForMaskedLM
import requests
from io import BytesIO
import pandas as pd
import numpy as np
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

def get_hidden_states(model, tokenizer, sequences, num_samples):
    """Get hidden states for a set of sequences."""
    tokens = [tokenizer(s, return_tensors="pt") for s in sequences[:num_samples]]
    hidden_states_dict = {i: [] for i in range(13)}  # 13 hidden layers
    
    for example in tokens:
        inputs = example
        inputs['output_hidden_states'] = True
        outputs = model.forward(**inputs)
        
        for i, layer in enumerate(outputs.hidden_states):
            layer = layer.detach().numpy()
            layer = layer[:, -1, :].squeeze(0)  # Use last token based on previous analysis
            hidden_states_dict[i].append(layer)
            
    return hidden_states_dict

def calculate_steering_vectors(membrane_states, cytosolic_states):
    """Calculate steering vectors for each layer."""
    steering_vectors = {}
    for i in range(13):
        membrane_avg = np.mean(np.array(membrane_states[i]), axis=0)
        cytosolic_avg = np.mean(np.array(cytosolic_states[i]), axis=0)
        steering_vectors[i] = membrane_avg - cytosolic_avg
    return steering_vectors

def find_optimal_threshold(membrane_projections, cytosolic_projections):
    """Find the optimal threshold using grid search for accuracy."""
    X = np.concatenate([membrane_projections, cytosolic_projections])
    y = np.concatenate([np.ones(len(membrane_projections)), 
                       np.zeros(len(cytosolic_projections))])
    
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

def evaluate_accuracy(model, tokenizer, test_sequences, steering_vector, layer_num, threshold, is_membrane=True):
    """Get projections for test sequences."""
    test_tokens = [tokenizer(s, return_tensors="pt") for s in test_sequences[:300]]  # Use 300 test samples
    hidden_states = []
    
    for example in test_tokens:
        inputs = example
        inputs['output_hidden_states'] = True
        outputs = model.forward(**inputs)
        layer = outputs.hidden_states[layer_num].detach().numpy()
        hidden_states.append(layer[:, -1, :].squeeze(0))  # Use last token consistently
    
    projections = [np.dot(state, steering_vector) / np.linalg.norm(steering_vector) for state in hidden_states]
    return projections

def analyze_sample_sizes(model, tokenizer, membrane_train, cytosolic_train, 
                        membrane_test, cytosolic_test, sample_sizes):
    """Analyze accuracy for different sample sizes."""
    results = {
        'membrane_accuracy': [],
        'cytosolic_accuracy': [],
        'overall_accuracy': []
    }
    
    for size in sample_sizes:
        print(f"\nProcessing sample size: {size}")
        
        # Get hidden states for training data
        membrane_states = get_hidden_states(model, tokenizer, membrane_train, size)
        cytosolic_states = get_hidden_states(model, tokenizer, cytosolic_train, size)
        
        # Calculate steering vectors
        steering_vectors = calculate_steering_vectors(membrane_states, cytosolic_states)
        
        # Use layer 3 based on previous analysis
        layer_num = 3
        steering_vector = steering_vectors[layer_num]
        
        # Get projections for test data
        membrane_projections = evaluate_accuracy(model, tokenizer, membrane_test, 
                                              steering_vector, layer_num, 0, True)
        cytosolic_projections = evaluate_accuracy(model, tokenizer, cytosolic_test, 
                                                steering_vector, layer_num, 0, False)
        
        # Calculate optimal threshold using test set projections
        threshold = find_optimal_threshold(membrane_projections, cytosolic_projections)

        print(f"Optimal threshold: {threshold:.3f}")
        print(f"Average membrane projection: {np.mean(membrane_projections):.3f}")
        print(f"Average cytosolic projection: {np.mean(cytosolic_projections):.3f}")
        
        # Calculate accuracies using the optimal threshold
        membrane_acc = sum(1 for p in membrane_projections if p > threshold) / len(membrane_projections)
        cytosolic_acc = sum(1 for p in cytosolic_projections if p < threshold) / len(cytosolic_projections)
        overall_acc = (membrane_acc + cytosolic_acc) / 2
        
        results['membrane_accuracy'].append(membrane_acc)
        results['cytosolic_accuracy'].append(cytosolic_acc)
        results['overall_accuracy'].append(overall_acc)
        
        print(f"Optimal threshold: {threshold:.3f}")
        print(f"Membrane accuracy: {membrane_acc:.3f}")
        print(f"Cytosolic accuracy: {cytosolic_acc:.3f}")
        print(f"Overall accuracy: {overall_acc:.3f}")
    
    return results

def plot_results(sample_sizes, results):
    """Create plots for the results."""
    plt.figure(figsize=(12, 6))
    
    # Plot all accuracies on one graph
    plt.plot(sample_sizes, results['overall_accuracy'], 'b-', label='Overall Accuracy', linewidth=2)
    plt.plot(sample_sizes, results['membrane_accuracy'], 'g--', label='Membrane Accuracy')
    plt.plot(sample_sizes, results['cytosolic_accuracy'], 'r--', label='Cytosolic Accuracy')
    
    plt.xlabel('Number of Training Samples per Class')
    plt.ylabel('Accuracy')
    plt.title('Protein Classification Accuracy vs Training Sample Size')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add points at each measurement
    plt.scatter(sample_sizes, results['overall_accuracy'], color='blue', s=50)
    plt.scatter(sample_sizes, results['membrane_accuracy'], color='green', s=50)
    plt.scatter(sample_sizes, results['cytosolic_accuracy'], color='red', s=50)
    
    plt.tight_layout()
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
    
    # Define sample sizes to test
    # test every value between 1 and 5
    # test every 5 values betwen 5 and 100
    # test every 10 values between 100 and 300
    sample_sizes = list(range(1, 6)) + list(range(5, 101, 5)) + list(range(100, 301, 10))
    
    # Run analysis
    print("Starting analysis...")
    results = analyze_sample_sizes(model, tokenizer, membrane_train, cytosolic_train, 
                                 membrane_test, cytosolic_test, sample_sizes)
    
    # Plot results
    print("Generating plot...")
    plot_results(sample_sizes, results)

if __name__ == "__main__":
    main()