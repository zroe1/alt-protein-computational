# Alt Protein Computational Track Proposal Code

## NOTE: code has been temporarily modified from the 35 million parameter model to the 8 million for testing.

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
