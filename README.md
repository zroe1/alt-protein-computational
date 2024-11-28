# Alt Protein Computational Track Proposal Code

## Running the 8 million parameter model:
The 8 million parameter model is pushed as part of the codebase temporarily so there is no need to download the model via hugging face.
1. Just install the required packages:
   ```
   pip install -r requirements.txt
   ```
2. Then run the code:
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
3. You will need to change these lines from run.py from this:

   ```python
   tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
   model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
   ```
   to this:

   ```python
   tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
   model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t12_35M_UR50D")
   ```

   I would also recommend changing this line:
   ```python
   STEERING_VECTOR_LAYER_NUMBER = 3
   ```
   To this line:
   ```python
   STEERING_VECTOR_LAYER_NUMBER = 6
   ```
4. Then you can run the code!!!
   ```
   python3 run.py
   ```
