# Dependencies
import json
import math
import torch
import warnings
import numpy as np
from peft import PeftModel
from transformers import AutoModel
from sentence_transformers import util
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from src.prompts.prompts import llm_finetuning_prep
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer 

# Supress warnings
warnings.filterwarnings("ignore")

# Define the device 
device                               = "mps" #"cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Load model and tokenizer for Falcon LLM
MODEL_PATH                           = "base_models/falcon1b/model"
TOKENIZER_PATH                       = "base_models/falcon1b/tokenizer"
LORA_PATH                            = "../results/llm_results/pipeline_finetuning_v9"

tokenizer                            = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = TOKENIZER_PATH)

# Ensure tokenizer uses EOS token for padding
tokenizer.pad_token                  = tokenizer.eos_token

base_model                           = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path = MODEL_PATH, 
                                                                            torch_dtype                   = torch.float16 if device == "cuda" else torch.float32, 
                                                                            device_map                    = device)

model                                = PeftModel.from_pretrained(base_model, '../results/llm_results/pipeline_finetuning_v9')

# Ensure the model recognizes the pad token
model.generation_config.pad_token_id = tokenizer.eos_token_id


# Move model explicitly to MPS if needed
if (device == "mps"):
    model = model.to(torch.float32)
else:
    model = model

# Load Sentence Transformer for semantic similarity
embedder                             = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to(device)

# Load test dataset (Modify the path based on your actual dataset)
TEST_DATA_PATH                       = "../data/evaluation_data/llm_test_dataset.json"

# Store all evaluation metrics
perplexities                         = list()
coherence_scores                     = list()
bleu_scores                          = list()


# Load evaluation samples (should be a JSON list of test prompts)
with open(TEST_DATA_PATH, "r") as f:
    test_samples = json.load(f)

print("\nEvaluating Perplexity Across Multiple Samples...\n")


# Process each test sample
for i, sample in enumerate(test_samples):
    # Convert the structured data (dictionary) into a string prompt
    test_text = f"""### Input:
                    - Company Name: Itobuz
                    - Platform: {sample['platform']}
                    - Heading: {sample['post_heading']}
                    - Content: {sample['post_contents']}
                    - Emojis: {", ".join(sample['emoji'])}
                    - Hashtags: {", ".join(sample['hashtags'])}
                    - Response: {sample['raw_post_content']}
                 """

    # Ensure proper formatting
    test_text = test_text.strip() 
    
    # Tokenize input text
    inputs    = tokenizer(test_text, 
                          return_tensors = "pt", 
                          padding        = True, 
                          truncation     = True, 
                          max_length     = 1024).to(device)
    
    # Convert inputs to float32
    if device == "mps":
        inputs["input_ids"] = inputs["input_ids"].to(torch.long)
        inputs["attention_mask"] = inputs["attention_mask"].to(torch.float32)
    else:
        inputs = inputs

    # Set model to no gradient adjusting mode
    with torch.no_grad():
        # Generate model output
        outputs        = model(**inputs, 
                               labels = inputs["input_ids"])

        # Calculate the loss
        loss           = outputs.loss

        # Generate response encoded text
        output_tokens  = model.generate(**inputs, 
                                        max_length           = 1024, 
                                        temperature          = 0.3, 
                                        do_sample            = True,
                                        top_p                = 0.95,
                                        top_k                = 40,
                                        no_repeat_ngram_size = 3,
                                        repetition_penalty   = 1.2)
        
        # Decode the model generated text
        generated_text = tokenizer.decode(output_tokens[0], 
                                          skip_special_tokens = True)

    # Compute perplexity
    perplexity          = math.exp(loss.item())

    # Compute sentence embedding similarity (semantic coherence)
    original_embedding  = embedder.encode(test_text, 
                                          convert_to_tensor = True)

    generated_embedding = embedder.encode(generated_text, 
                                          convert_to_tensor = True)

    similarity_score    = util.pytorch_cos_sim(original_embedding, 
                                               generated_embedding).item()

    # Compute BLEU Score (word-level coherence check)
    reference           = [test_text.split()]
    candidate           = generated_text.split()
    bleu_score          = sentence_bleu(reference, candidate)
    
    # Append all scores to their respective lists
    coherence_scores.append(similarity_score)
    perplexities.append(perplexity)
    bleu_scores.append(bleu_score)

    print(f"Sample {i+1}:\n Perplexity = {perplexity:.2f},\n Coherence = {similarity_score:.4f},\n BLEU = {bleu_score:.4f}\n\n")

# Compute aggregated statistics
avg_perplexity  = np.mean(perplexities)
avg_coherence   = np.mean(coherence_scores)
avg_bleu        = np.mean(bleu_scores)

print("\nFinal Evaluation Summary:")
print(f"Average Perplexity (Lower is better): {avg_perplexity:.2f}")
print(f"Average Coherence (Higher is better): {avg_coherence:.4f}")
print(f"Average BLEU Score (Higher is better): {avg_bleu:.4f}\n")


############################ EVALUATE ############################

# High perplexity → Model struggles with fluency.

# Low coherence   → Model isn’t generating relevant responses.

# Low BLEU        → Model is losing key details.

##################################################################