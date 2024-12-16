from transformers import AutoTokenizer, AutoModelForMaskedLM
import requests
from io import BytesIO
import pandas
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
import json

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t12_35M_UR50D")

print("Downloading and processing UniProt dataset...")
query_url ="https://rest.uniprot.org/uniprotkb/stream?compressed=true&fields=accession%2Csequence%2Ccc_subcellular_location&format=tsv&query=%28%28organism_id%3A9606%29%20AND%20%28reviewed%3Atrue%29%20AND%20%28length%3A%5B80%20TO%20500%5D%29%29"
uniprot_request = requests.get(query_url)
bio = BytesIO(uniprot_request.content)
df = pandas.read_csv(bio, compression='gzip', sep='\t')

cytosolic = df['Subcellular location [CC]'].str.contains("Cytosol") | df['Subcellular location [CC]'].str.contains("Cytoplasm")
membrane = df['Subcellular location [CC]'].str.contains("Membrane") | df['Subcellular location [CC]'].str.contains("Cell membrane")

membrane_df = df[membrane & ~cytosolic]
cytosolic_df = df[cytosolic & ~membrane]

membrane_sequences = membrane_df["Sequence"].tolist()
cytosolic_sequences = cytosolic_df["Sequence"].tolist()

membrane_train, membrane_test = train_test_split(membrane_sequences, test_size=0.2, random_state=42)
cytosolic_train, cytosolic_test = train_test_split(cytosolic_sequences, test_size=0.2, random_state=42)
membrane_train = membrane_train[:200]
cytosolic_train = cytosolic_train[:200]

print(f"Processing training data: {len(membrane_train)} membrane sequences and {len(cytosolic_train)} cytosolic sequences")

membrane_train_tokens = [tokenizer(s, return_tensors="pt") for s in membrane_train]
cytosolic_train_tokens = [tokenizer(s, return_tensors="pt") for s in cytosolic_train]

NUM_HIDDEN_LAYERS = 13

print("Calculating hidden states for membrane training proteins...")
membrane_train_hidden_states = dict()
for i, membrane_example in enumerate(membrane_train_tokens):
    if i % 25 == 0:  # Print progress every 25 sequences
        print(f"Processing membrane sequence {i}/{len(membrane_train_tokens)}")
    inputs = membrane_example
    inputs['output_hidden_states'] = True
    outputs = model.forward(**inputs)

    for i, layer in enumerate(outputs.hidden_states):
        layer = layer.detach().numpy()
        layer = layer[:, -1, :].squeeze(0) # Use last token based on previous analysis
        membrane_train_hidden_states[i] = membrane_train_hidden_states.get(i, []) + [layer]

print("Calculating hidden states for cytosolic training proteins...")
cytosolic_train_hidden_states = dict()
for i, cytosolic_example in enumerate(cytosolic_train_tokens):
    if i % 25 == 0:  # Print progress every 25 sequences
        print(f"Processing membrane sequence {i}/{len(cytosolic_train_tokens)}")
    inputs = cytosolic_example
    inputs['output_hidden_states'] = True
    outputs = model.forward(**inputs)

    for i, layer in enumerate(outputs.hidden_states):
        layer = layer.detach().numpy()
        layer = layer[:, -1, :].squeeze(0) # Use last token based on previous analysis
        cytosolic_train_hidden_states[i] = cytosolic_train_hidden_states.get(i, []) + [layer]

print("Computing steering vectors...")
STEERING_VECTOR_LAYER_NUMBER = 3
membrane_train_avg_hidden_states = np.array(membrane_train_hidden_states[STEERING_VECTOR_LAYER_NUMBER])
membrane_train_avg_hidden_states = np.sum(membrane_train_avg_hidden_states, axis=0) / len(membrane_train_hidden_states[STEERING_VECTOR_LAYER_NUMBER])
cytosolic_train_avg_hidden_states = np.array(cytosolic_train_hidden_states[STEERING_VECTOR_LAYER_NUMBER])
cytosolic_train_avg_hidden_states = np.sum(cytosolic_train_avg_hidden_states, axis=0) / len(cytosolic_train_hidden_states[STEERING_VECTOR_LAYER_NUMBER])

final_steering_vector = membrane_train_avg_hidden_states - cytosolic_train_avg_hidden_states

# Calculate training projections to determine threshold
print("Calculating training projections to determine optimal threshold...")

def project_vector(vector, steering_vector):
    return np.dot(vector, steering_vector) / np.linalg.norm(steering_vector)

# Calculate projections for training data
membrane_train_projections = []
cytosolic_train_projections = []

membrane_train_layer = membrane_train_hidden_states[STEERING_VECTOR_LAYER_NUMBER]
cytosolic_train_layer = cytosolic_train_hidden_states[STEERING_VECTOR_LAYER_NUMBER]

for hidden_state in membrane_train_layer:
    projection = project_vector(hidden_state, final_steering_vector)
    membrane_train_projections.append(projection)

for hidden_state in cytosolic_train_layer:
    projection = project_vector(hidden_state, final_steering_vector)
    cytosolic_train_projections.append(projection)

# Find optimal threshold using training data
def find_optimal_thresholds(membrane_projections, cytosolic_projections):
    # Prepare data for ROC and other analyses
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
    return thresholds[best_idx], accuracies[best_idx]

SIMILARITY_THRESHOLD, train_accuracy = find_optimal_thresholds(membrane_train_projections, cytosolic_train_projections)

print(f"Determined optimal threshold from training data: {SIMILARITY_THRESHOLD}")
print(f"Training accuracy at optimal threshold: {train_accuracy}")

# Save final steering vector to file
with open("final_steering_vector.json", "w") as f:
    json.dump(final_steering_vector.tolist(), f)

print("Testing steering vector predictions...")
membrane_test_tokens = [tokenizer(s, return_tensors="pt") for s in membrane_test]
cytosolic_test_tokens = [tokenizer(s, return_tensors="pt") for s in cytosolic_test]

membrane_test_tokens = membrane_test_tokens[:100]
cytosolic_test_tokens = cytosolic_test_tokens[:100]

membrane_test_hidden_states = dict()
for membrane_example in membrane_test_tokens:
    inputs = membrane_example
    inputs['output_hidden_states'] = True
    outputs = model.forward(**inputs)

    for i, layer in enumerate(outputs.hidden_states):
        layer = layer.detach().numpy()
        layer = layer[:, -1, :].squeeze(0) # Use last token based on previous analysis
        membrane_test_hidden_states[i] = membrane_test_hidden_states.get(i, []) + [layer]

cytosolic_test_hidden_states = dict()
for cytosolic_example in cytosolic_test_tokens:
    inputs = cytosolic_example
    inputs['output_hidden_states'] = True
    outputs = model.forward(**inputs)

    for i, layer in enumerate(outputs.hidden_states):
        layer = layer.detach().numpy()
        layer = layer[:, -1, :].squeeze(0) # Use last token based on previous analysis
        cytosolic_test_hidden_states[i] = cytosolic_test_hidden_states.get(i, []) + [layer]

print("Making final predictions...")
membrane_predictions = []
cytosolic_predictions = []

membrane_example = membrane_test_hidden_states[STEERING_VECTOR_LAYER_NUMBER]
cytosolic_example = cytosolic_test_hidden_states[STEERING_VECTOR_LAYER_NUMBER]

for i in range(len(membrane_example)):
    membrane_hidden_state = membrane_example[i]
    membrane_projection = project_vector(membrane_hidden_state, final_steering_vector)
    membrane_predictions.append(membrane_projection)

for i in range(len(cytosolic_example)):
    cytosolic_hidden_state = cytosolic_example[i]
    cytosolic_projection = project_vector(cytosolic_hidden_state, final_steering_vector)
    cytosolic_predictions.append(cytosolic_projection)

membrane_test_avg_projection = np.mean(membrane_predictions)
cytosolic_test_avg_projection = np.mean(cytosolic_predictions)

membrane_correct = sum([1 for p in membrane_predictions if p > SIMILARITY_THRESHOLD])
cytosolic_correct = sum([1 for p in cytosolic_predictions if p < SIMILARITY_THRESHOLD])

print("\nFinal Results:")
print(f"Membrane correct: {membrane_correct} / {len(membrane_predictions)}")
print(f"Cytosolic correct: {cytosolic_correct} / {len(cytosolic_predictions)}")
print("--------")

accuracy = (membrane_correct + cytosolic_correct) / (len(membrane_predictions) + len(cytosolic_predictions))
print(f"Test accuracy: {accuracy}")

membrane_accuracy = membrane_correct / len(membrane_predictions)
cytosolic_accuracy = cytosolic_correct / len(cytosolic_predictions)
print(f"Membrane accuracy: {membrane_accuracy}")
print(f"Cytosolic accuracy: {cytosolic_accuracy}")
print("--------")

print(f"Test membrane average projection: {membrane_test_avg_projection}")
print(f"Test cytosolic average projection: {cytosolic_test_avg_projection}")
print("--------")

print(f"Saving steering vector from layer #{STEERING_VECTOR_LAYER_NUMBER} to final_steering_vector.json")
with open("final_steering_vector.json", "w") as f:
    json.dump(final_steering_vector.tolist(), f)