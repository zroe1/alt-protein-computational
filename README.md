# Steering Vectors for Low-Data Protein Classification

This project introduces a novel approach to protein classification using steering vectors derived from protein language models. By computing directional vectors in the model's hidden space that distinguish between different protein properties (such as membrane vs. cytosolic proteins), the framework achieves high accuracy with minimal training data. The implementation demonstrates this technique using the ESM-2 protein language model, achieving approximately 90% accuracy in protein localization classification with only 200 training examples per class. This proof-of-concept shows promise for efficiently analyzing and classifying proteins for various applications in biotechnology and alternative protein development.

## Table of Contents:
1. `run.py`: run the most recent interation of our code (~90% accuracy)
2. `experiments`: experiments to learn which training procedures are most effective. (Note that all experiments rely on using test data to learn the threshold perameter to better isolate the variables we are testing for; this would obviously not be possible in a real world setting because we wouldn't have labels.)
3. `figures`: graphs and tables produced by the experiments.
4. `final_steering_vector.json`: the final steering vector (you can think of this as a learned model) produced by run.py

## Methods
The protein classification framework uses steering vectors derived from ESM-2, a protein language model, to distinguish between membrane and cytosolic proteins. The method consists of the following key steps:

### Data Collection and Preprocessing

Protein sequences are retrieved from UniProt, filtering for human proteins (organism ID: 9606) that have lengths between 80-500 amino acids. Sequences are categorized into two classes:

1. Membrane proteins: Those annotated with "Membrane" or "Cell membrane" locations
2. Cytosolic proteins: Those annotated with "Cytosol" or "Cytoplasm" locations




### Hidden State Extraction

200 sequences per class are used for training.

1. Each protein sequence is tokenized using the ESM-2 tokenizer
2. The sequences are processed through the ESM-2 T12 35M model (35 million parameters)
3. Hidden states are extracted from all 13 layers of the model
4. The final token's hidden state representation is used for each sequence

### Steering Vector Computation

For a selected layer (layer 3 in the implementation), the average hidden state is computed for each class:

Membrane protein vector: mean of all membrane protein hidden states
Cytosolic protein vector: mean of all cytosolic protein hidden states

The steering vector is calculated as the difference between these averages:

```python
final_steering_vector = membrane_train_avg_hidden_states - cytosolic_train_avg_hidden_states
```


### Classification Procedure

Next, an optimal classification threshold is determined using the training data.

For each protein in the validation set, the protien is sent through the model and the vector at the selected layer of the model is extracted. A projection score is calculated by projecting this hidden state onto the steering vector using a scalar projection:

```python
def project_vector(vector, steering_vector):
    return np.dot(vector, steering_vector) / np.linalg.norm(steering_vector)
```

Proteins are classified based on their projection scores:

* Scores above threshold → Membrane protein
* Scores below threshold → Cytosolic protein

### Model Evaluation

The classifier is evaluated on a held-out test set of membrane and cytosolic proteins. Performance metrics include:

1. Overall accuracy
2. Per-class accuracy
3. Average projection values for each class (for debugging purposes)

The steering vector is saved to final_steering_vector.json for future use in protein classification tasks.

## Steps to take after cloning

1. Run the following command:
   
   ```
   git clone https://huggingface.co/facebook/esm2_t12_35M_UR50D
   ```
    This will download everything you need to run the most recent version of 35 million parameter protein language model from MetaAI. More information <a href ="https://huggingface.co/facebook/esm2_t12_35M_UR50D">here</a>. You can also download larger or smaller models (8 million - 15 billion parameters) <a href="https://huggingface.co/facebook/esm2_t12_35M_UR50D">here</a>.
2. Then you can run the following if you don't already have the needed packages installed on whichever environment you are working on (they are all common ML packages):
   ```
   pip install -r requirements.txt
   ```
3. Then you can run the code:
   ```
   python3 run.py
   ```
