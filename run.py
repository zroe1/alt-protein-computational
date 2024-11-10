from transformers import AutoTokenizer, AutoModelForMaskedLM
import requests
from io import BytesIO
import pandas
from sklearn.model_selection import train_test_split
import numpy as np

# load model and tokenizer
# ------------------------
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t12_35M_UR50D")


# load sample dataset
# -------------------
# Use this souce as a reference: https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/protein_language_modeling.ipynb#scrollTo=c718ffbc
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
for membrane_example in membrane_train_tokens[:10]:
    inputs = membrane_example
    inputs['output_hidden_states'] = True
    outputs = model.forward(**inputs)

    for i, layer in enumerate(outputs.hidden_states):
        # print(i, "->", layer.shape)
        # we try mean pooling the hidden states (could also do max pooling, last token, etc)
        layer = layer.detach().numpy()
        layer = layer.mean(axis=1).squeeze(0)
        print(layer.shape)
        # print(layer)
        
        membrane_train_hidden_states[i] = membrane_train_hidden_states.get(i, []) + [layer]

cytosolic_train_hidden_states = dict()
for cytosolic_example in cytosolic_train_tokens[:10]:
    inputs = cytosolic_example
    inputs['output_hidden_states'] = True
    outputs = model.forward(**inputs)

    for i, layer in enumerate(outputs.hidden_states):
        layer = layer.detach().numpy()
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
    print(avg_hidden_states)
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
import json

#make steering vectors JSON serializable
for k, v in steering_vectors.items():
    steering_vectors[k] = v.tolist()


with open("steering_vectors.json", "w") as f:
    json.dump(steering_vectors, f)

# function to project a vector onto the steering vector (scalar prediction where the function returns a scalar)
def project_vector(vector, steering_vector):
    return np.dot(vector, steering_vector) / np.linalg.norm(steering_vector)







# print(outputs.hidden_states[0].shape)
# print()
# print(outputs.hidden_states[1].shape)
# print(outputs.hidden_states[2].shape)
# print(outputs.hidden_states[3].shape)

# print(len(outputs.hidden_states))
# print(outputs.hidden_states)