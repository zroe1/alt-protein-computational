# Steering Vectors for Low-Data Protein Classification

This project introduces a novel approach to protein classification using steering vectors derived from protein language models. By computing directional vectors in the model's hidden space that distinguish between different protein properties (such as membrane vs. cytosolic proteins), the framework achieves high accuracy with minimal training data. The implementation demonstrates this technique using the ESM-2 protein language model, achieving approximately 90% accuracy in protein localization classification with only 200 training examples per class. This proof-of-concept shows promise for efficiently analyzing and classifying proteins for various applications in biotechnology and alternative protein development.

## Table of Contents:
1. `run.py`: run the most recent interation of our code (~90% accuracy)
2. `experiments`: experiments to learn which training procedures are most effective. (Note that all experiments rely on using test data to learn the threshold perameter to better isolate the variables we are testing for; this would obviously not be possible in a real world setting because we wouldn't have labels.)
3. `figures`: graphs and tables produced by the experiments.
4. `final_steering_vector.json`: the final steering vector (you can think of this as a learned model) produced by run.py

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
