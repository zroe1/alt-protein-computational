from transformers import AutoTokenizer, AutoModelForMaskedLM
import requests
from io import BytesIO
import pandas
from sklearn.model_selection import train_test_split
import numpy as np
import json

# load model and tokenizer
# ------------------------
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t12_35M_UR50D")


# load sample dataset
# -------------------
# Use this souce as a reference for lines (18-37): https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/protein_language_modeling.ipynb#scrollTo=c718ffbc
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

# Split membrane and cytosolic sequences
membrane_train, membrane_test = train_test_split(membrane_sequences, test_size=0.2, random_state=42)
cytosolic_train, cytosolic_test = train_test_split(cytosolic_sequences, test_size=0.2, random_state=42)

membrane_train_tokens = [tokenizer(s, return_tensors="pt") for s in membrane_train]
cytosolic_train_tokens = [tokenizer(s, return_tensors="pt") for s in cytosolic_train]


# average hidden layer states
# -------------------
NUM_HIDDEN_LAYERS = 13

membrane_train_hidden_states = dict()
for membrane_example in membrane_train_tokens[:100]: # first 100 examples
    inputs = membrane_example
    inputs['output_hidden_states'] = True
    outputs = model.forward(**inputs)

    for i, layer in enumerate(outputs.hidden_states):
        layer = layer.detach().numpy()
        # we try mean pooling the hidden states (could also do max pooling, last token, etc)
        layer = layer.mean(axis=1).squeeze(0)
        membrane_train_hidden_states[i] = membrane_train_hidden_states.get(i, []) + [layer]

cytosolic_train_hidden_states = dict()
for cytosolic_example in cytosolic_train_tokens[:100]: # first 100 examples
    inputs = cytosolic_example
    inputs['output_hidden_states'] = True
    outputs = model.forward(**inputs)

    for i, layer in enumerate(outputs.hidden_states):
        layer = layer.detach().numpy()
        # we try mean pooling the hidden states (could also do max pooling, last token, etc)
        layer = layer.mean(axis=1).squeeze(0)
        cytosolic_train_hidden_states[i] = cytosolic_train_hidden_states.get(i, []) + [layer]

# find average vector for membrane hidden layer state for each layer
membrane_train_avg_hidden_states = dict()
cytosolic_train_avg_hidden_states = dict()
for i in range(NUM_HIDDEN_LAYERS):
    # membrane avgs
    all_layaer_hidden_states = np.array(membrane_train_hidden_states[i])
    sum_hidden_states = np.sum(all_layaer_hidden_states, axis=0)
    avg_hidden_states = sum_hidden_states / len(membrane_train_hidden_states[i])
    membrane_train_avg_hidden_states[i] = avg_hidden_states

    # cytosolic avgs
    all_layaer_hidden_states = np.array(cytosolic_train_hidden_states[i])
    sum_hidden_states = np.sum(all_layaer_hidden_states, axis=0)
    avg_hidden_states = sum_hidden_states / len(cytosolic_train_hidden_states[i])
    cytosolic_train_avg_hidden_states[i] = avg_hidden_states

steering_vectors = dict()
for i in range(NUM_HIDDEN_LAYERS):
    # this encode the difference between membrane and cytosolic hidden states
    steering_vectors[i] = membrane_train_avg_hidden_states[i] - cytosolic_train_avg_hidden_states[i]


# write steering vectors to file as json
# --------------------------------------
#make steering vectors JSON serializable
for k, v in steering_vectors.items():
    steering_vectors[k] = v.tolist()

with open("steering_vectors.json", "w") as f:
    json.dump(steering_vectors, f)




# test steering vector prediction
# -------------------------------
SIMILARITY_THRESHOLD = 0.5

membrane_test_tokens = [tokenizer(s, return_tensors="pt") for s in membrane_test]
cytosolic_test_tokens = [tokenizer(s, return_tensors="pt") for s in cytosolic_test]

membrane_test_hidden_states = dict()
for membrane_example in membrane_test_tokens: # run on all test examples
    inputs = membrane_example
    inputs['output_hidden_states'] = True
    outputs = model.forward(**inputs)

    for i, layer in enumerate(outputs.hidden_states):
        layer = layer.detach().numpy()
        layer = layer.mean(axis=1).squeeze(0)
        membrane_test_hidden_states[i] = membrane_test_hidden_states.get(i, []) + [layer]

cytosolic_test_hidden_states = dict()
for cytosolic_example in cytosolic_test_tokens: # run on all test examples
    inputs = cytosolic_example
    inputs['output_hidden_states'] = True
    outputs = model.forward(**inputs)

    for i, layer in enumerate(outputs.hidden_states):
        layer = layer.detach().numpy()
        layer = layer.mean(axis=1).squeeze(0)
        cytosolic_test_hidden_states[i] = cytosolic_test_hidden_states.get(i, []) + [layer]

STEERING_VECTOR_LAYER_NUMBER = 6

# for each test example, project the hidden states onto the steering vector
membrane_predictions = []
cytosolic_predictions = []

# function to project a vector onto the steering vector (scalar projection)
def project_vector(vector, steering_vector):
    return np.dot(vector, steering_vector) / np.linalg.norm(steering_vector)

membrane_example = membrane_test_hidden_states[STEERING_VECTOR_LAYER_NUMBER]
cytosolic_example = cytosolic_test_hidden_states[STEERING_VECTOR_LAYER_NUMBER]


# test on full dataset of membrane and cytosolic examples
# -------------------------------------------------------
for i in range(len(membrane_example)):
    membrane_hidden_state = membrane_example[i]
    membrane_projection = project_vector(membrane_hidden_state, steering_vectors[STEERING_VECTOR_LAYER_NUMBER])
    membrane_predictions.append(membrane_projection)

for i in range(len(cytosolic_example)):
    cytosolic_hidden_state = cytosolic_example[i]
    cytosolic_projection = project_vector(cytosolic_hidden_state, steering_vectors[STEERING_VECTOR_LAYER_NUMBER])
    cytosolic_predictions.append(cytosolic_projection)

# calculate the number of correct predictions
membrane_correct = sum([1 for p in membrane_predictions if p > SIMILARITY_THRESHOLD])
cytosolic_correct = sum([1 for p in cytosolic_predictions if p < SIMILARITY_THRESHOLD])

print(f"Membrane correct: {membrane_correct} / {len(membrane_predictions)}")
print(f"Cytosolic correct: {cytosolic_correct} / {len(cytosolic_predictions)}")

# print the accuracy
accuracy = (membrane_correct + cytosolic_correct) / (len(membrane_predictions) + len(cytosolic_predictions))
print(f"Accuracy: {accuracy}")

# print accuracy for just membrane and just cytosolic
membrane_accuracy = membrane_correct / len(membrane_predictions)
cytosolic_accuracy = cytosolic_correct / len(cytosolic_predictions)
print(f"Membrane accuracy: {membrane_accuracy}")
print(f"Cytosolic accuracy: {cytosolic_accuracy}")

