# Alt Protein Computational Track Proposal Code

## Running the 8 million parameter model:
The 8 million parameter model is pushed as part of the codebase temporarily so there is no need to download the model via hugging face.
1. Switch to the branch where the 8 million parameter model is already uploaded:
   ```
   run-8-million
   ```
3. Just install the required packages:
   ```
   pip install -r requirements.txt
   ```
4. Then run the code:
   ```
   python3 run.py
   ```
## Running the 35 million parameter model:

Steps to take after cloning:

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
