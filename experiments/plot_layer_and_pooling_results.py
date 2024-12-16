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

def calculate_steering_vector(membrane_states, cytosolic_states):
    """Calculate steering vector from hidden states."""
    membrane_avg = np.mean(membrane_states, axis=0)
    cytosolic_avg = np.mean(cytosolic_states, axis=0)
    return membrane_avg - cytosolic_avg

def project_vector(vector, steering_vector):
    """Project a vector onto the steering vector."""
    return np.dot(vector, steering_vector) / np.linalg.norm(steering_vector)

def find_optimal_thresholds(membrane_projections, cytosolic_projections):
    """Find optimal threshold using grid search."""
    # Prepare data for analysis
    X = np.concatenate([membrane_projections, cytosolic_projections])
    y = np.concatenate([np.ones(len(membrane_projections)), 
                       np.zeros(len(cytosolic_projections))])
    
    # Grid search for accuracy
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

def get_hidden_states_with_pooling(model, tokenizer, sequences, layer_num, pooling_strategy):
    """Get hidden states for sequences with different pooling strategies."""
    tokens = [tokenizer(s, return_tensors="pt") for s in sequences[:200]]
    hidden_states = []
    
    for example in tokens:
        inputs = example
        inputs['output_hidden_states'] = True
        outputs = model.forward(**inputs)
        layer = outputs.hidden_states[layer_num].detach().numpy()
        
        if pooling_strategy == 'mean':
            pooled = layer.mean(axis=1).squeeze(0)
        elif pooling_strategy == 'max':
            pooled = layer.max(axis=1).squeeze(0)
        elif pooling_strategy == 'min':
            pooled = layer.min(axis=1).squeeze(0)
        elif pooling_strategy == 'first':
            pooled = layer[:, 0, :].squeeze(0)  # First token
        elif pooling_strategy == 'last':
            pooled = layer[:, -1, :].squeeze(0)  # Last token
        
        hidden_states.append(pooled)
            
    return hidden_states

def evaluate_layer_pooling(model, tokenizer, membrane_train, cytosolic_train, 
                         membrane_test, cytosolic_test, layer_num, pooling_strategy):
    """Evaluate performance for a specific layer and pooling strategy."""
    
    # Get hidden states with specified pooling
    membrane_states = get_hidden_states_with_pooling(model, tokenizer, membrane_train, 
                                                   layer_num, pooling_strategy)
    cytosolic_states = get_hidden_states_with_pooling(model, tokenizer, cytosolic_train, 
                                                     layer_num, pooling_strategy)
    
    # Calculate steering vector
    steering_vector = calculate_steering_vector(membrane_states, cytosolic_states)
    
    # Get test projections
    membrane_test_states = get_hidden_states_with_pooling(model, tokenizer, membrane_test, 
                                                         layer_num, pooling_strategy)
    cytosolic_test_states = get_hidden_states_with_pooling(model, tokenizer, cytosolic_test, 
                                                          layer_num, pooling_strategy)
    
    membrane_projections = [project_vector(state, steering_vector) for state in membrane_test_states]
    cytosolic_projections = [project_vector(state, steering_vector) for state in cytosolic_test_states]
    
    # Find optimal threshold
    threshold = find_optimal_thresholds(membrane_projections, cytosolic_projections)
    
    # Calculate accuracy
    X = np.concatenate([membrane_projections, cytosolic_projections])
    y = np.concatenate([np.ones(len(membrane_projections)), np.zeros(len(cytosolic_projections))])
    predictions = (X > threshold).astype(int)
    accuracy = accuracy_score(y, predictions)
    
    return accuracy

def plot_layer_pooling_comparison(results):
    """Create visualizations for layer-pooling comparison."""
    # Create heatmap
    plt.figure(figsize=(12, 6))
    
    # Convert results to matrix form
    layers = sorted(list(results.keys()))
    pooling_methods = sorted(list(results[layers[0]].keys()))
    accuracy_matrix = np.array([[results[layer][method] for method in pooling_methods] 
                              for layer in layers])
    
    # Plot heatmap
    sns.heatmap(accuracy_matrix, annot=True, fmt='.3f', cmap='viridis',
                xticklabels=pooling_methods, yticklabels=layers,
                cbar_kws={'label': 'Accuracy'})
    
    plt.title('Accuracy by Layer and Pooling Strategy')
    plt.xlabel('Pooling Strategy')
    plt.ylabel('Layer')
    plt.tight_layout()
    plt.savefig('layer_pooling_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create line plot
    plt.figure(figsize=(12, 6))
    for method in pooling_methods:
        accuracies = [results[layer][method] for layer in layers]
        plt.plot(layers, accuracies, marker='o', label=method)
    
    plt.title('Accuracy across Layers by Pooling Strategy')
    plt.xlabel('Layer')
    plt.ylabel('Accuracy')
    plt.legend(title='Pooling Strategy')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('layer_pooling_lines.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Load model and tokenizer
    print("\n" + "="*80)
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
    model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t12_35M_UR50D")
    
    # Load and split data
    print("Loading and preprocessing data...")
    membrane_sequences, cytosolic_sequences = load_data()
    membrane_train, membrane_test = train_test_split(membrane_sequences, test_size=0.2, random_state=42)
    cytosolic_train, cytosolic_test = train_test_split(cytosolic_sequences, test_size=0.2, random_state=42)
    
    # Define pooling strategies and layers to test
    pooling_strategies = ['mean', 'max', 'min', 'first', 'last']
    layers = list(range(13))
    
    # Print analysis parameters
    print("\n" + "="*80)
    print("Analysis Parameters:")
    print(f"Number of layers: {len(layers)}")
    print(f"Pooling strategies: {', '.join(pooling_strategies)}")
    print(f"Training samples per class: 200")
    print("="*80 + "\n")
    
    # Evaluate all combinations
    results = {}
    total_combinations = len(layers) * len(pooling_strategies)
    current = 0
    
    # Create results table header
    print("\nDetailed Results:")
    print("-"*80)
    print(f"{'Layer':<6} {'Pooling':<8} {'Accuracy':<10}")
    print("-"*80)
    
    for layer in layers:
        results[layer] = {}
        layer_best_acc = 0
        layer_best_pooling = ''
        
        for pooling in pooling_strategies:
            current += 1
            
            accuracy = evaluate_layer_pooling(
                model, tokenizer,
                membrane_train, cytosolic_train,
                membrane_test, cytosolic_test,
                layer, pooling
            )
            
            results[layer][pooling] = accuracy
            
            # Update layer's best result
            if accuracy > layer_best_acc:
                layer_best_acc = accuracy
                layer_best_pooling = pooling
            
            # Print result row
            print(f"{layer:<6} {pooling:<8} {accuracy:.4f}")
        
        # Print layer summary
        print(f"Layer {layer} best: {layer_best_pooling} ({layer_best_acc:.4f})")
        print("-"*80)
    
    # Create visualizations
    print("\nGenerating plots...")
    plot_layer_pooling_comparison(results)
    
    # Print overall summary
    print("\n" + "="*80)
    print("Overall Best Combinations:")
    print("-"*40)
    all_combinations = [(layer, pooling, results[layer][pooling]) 
                       for layer in layers 
                       for pooling in pooling_strategies]
    top_3 = sorted(all_combinations, key=lambda x: x[2], reverse=True)[:3]
    
    for i, (layer, pooling, accuracy) in enumerate(top_3, 1):
        print(f"{i}. Layer {layer:<2} with {pooling:<5} pooling: {accuracy:.4f}")
    
    # Print pooling strategy summaries
    print("\nPooling Strategy Summaries:")
    print("-"*40)
    for pooling in pooling_strategies:
        accuracies = [results[layer][pooling] for layer in layers]
        avg_acc = np.mean(accuracies)
        best_layer = layers[np.argmax(accuracies)]
        max_acc = np.max(accuracies)
        print(f"{pooling:<5} pooling:")
        print(f"  Average accuracy: {avg_acc:.4f}")
        print(f"  Best layer: {best_layer} ({max_acc:.4f})")
    
    print("\nAnalysis complete! Plots have been saved as 'layer_pooling_heatmap.png' and 'layer_pooling_lines.png'")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

# def main():
#     # Load model and tokenizer
#     print("Loading model and tokenizer...")
#     tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
#     model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t12_35M_UR50D")
    
#     # Load and split data
#     print("Loading and preprocessing data...")
#     membrane_sequences, cytosolic_sequences = load_data()
#     membrane_train, membrane_test = train_test_split(membrane_sequences, test_size=0.2, random_state=42)
#     cytosolic_train, cytosolic_test = train_test_split(cytosolic_sequences, test_size=0.2, random_state=42)
    
#     # Define pooling strategies and layers to test
#     pooling_strategies = ['mean', 'max', 'min', 'first', 'last']
#     layers = list(range(13))  # 13 layers total
    
#     # Evaluate all combinations
#     results = {}
#     total_combinations = len(layers) * len(pooling_strategies)
#     current = 0
    
#     for layer in layers:
#         results[layer] = {}
#         for pooling in pooling_strategies:
#             current += 1
#             print(f"Processing combination {current}/{total_combinations}: Layer {layer}, {pooling} pooling")
            
#             accuracy = evaluate_layer_pooling(
#                 model, tokenizer,
#                 membrane_train, cytosolic_train,
#                 membrane_test, cytosolic_test,
#                 layer, pooling
#             )
#             results[layer][pooling] = accuracy
    
#     # Create visualizations
#     print("Generating plots...")
#     plot_layer_pooling_comparison(results)
    
#     # Print best combinations
#     print("\nTop 3 Layer-Pooling Combinations:")
#     all_combinations = [(layer, pooling, results[layer][pooling]) 
#                        for layer in layers 
#                        for pooling in pooling_strategies]
#     top_3 = sorted(all_combinations, key=lambda x: x[2], reverse=True)[:3]
    
#     for layer, pooling, accuracy in top_3:
#         print(f"Layer {layer} with {pooling} pooling: {accuracy:.3f}")

# if __name__ == "__main__":
#     main()