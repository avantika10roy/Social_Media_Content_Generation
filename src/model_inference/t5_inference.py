## ----- DONE BY PRIYAM PAL -----

## ----- INFERENCE OF FINE-TUNED T5-BASE MODEL -----

# DEPENDENCIES

import re
import os
import torch
from peft import PeftModel
from peft import PeftConfig
from transformers import T5Tokenizer
from transformers import T5ForConditionalGeneration
from src.prompts.llm_prompt_template import generate_prompt

import warnings
warnings.filterwarnings(action = 'ignore')


class T5LoRAInference:
    
    """
    Class to handle inference with a fine-tuned T5 model using LoRA adapters.
    
    Attributes:
    -----------
        model_path             {str}          : Path to the model directory containing adapter files
        
        device             {torch.device}     : Device to run inference on (CPU, CUDA, or MPS)
        
        tokenizer          {T5Tokenizer}      : Tokenizer for the T5 model
        
        model               {PeftModel}       : The loaded T5 model with LoRA adapters
    
    """
    
    def __init__(self, model_path: str, device = None) -> None:
        
        """
        Initialize the T5LoRAInference with the specified model path.
        
        Arguments:
        ----------
            model_path                 {str}               : Path to the directory containing the adapter files
            
            device             {torch.device, optional}    : Device to run inference on. If None, uses MPS if available,
                                                             then CUDA if available, otherwise CPU
        
        Raises:
        --------
            FileNotFoundError: If model files are not found at the specified path
            
            RuntimeError: If there's an issue loading the model or adapter
        
        """
        self.model_path        = model_path
        
        if device is not None:
            self.device        = device
        
        else:
            if torch.backends.mps.is_available():
                self.device    = torch.device("mps")
            
            elif torch.cuda.is_available():
                self.device    = torch.device("cuda")
            
            else:
                self.device    = torch.device("cpu")
        
        print(f"Using device: {self.device}")
        

        required_files         = ['adapter_config.json', 'tokenizer.json', 'adapter_model.safetensors']
        
        for file in required_files:
            file_path          = os.path.join(model_path, file)
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        try:
            self.tokenizer     = T5Tokenizer.from_pretrained(model_path, 
                                                             local_files_only  = True, 
                                                             legacy            = False
                                                             )
        
        except Exception as e:
            raise FileNotFoundError(f"Failed to load tokenizer from {model_path}: {str(e)}")
        
        try:
            self.config        = PeftConfig.from_pretrained(model_path, local_files_only = True)
       
        except Exception as e:
            raise FileNotFoundError(f"Failed to load adapter config from {model_path}: {str(e)}")
        
        try:
            base_model         = T5ForConditionalGeneration.from_pretrained(self.config.base_model_name_or_path)
            
            self.model         = PeftModel.from_pretrained(base_model, model_path, local_files_only = True)
            self.model.to(self.device)
            self.model.eval()
        
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    # def create_prompt(self, platform: str, topic: str, brief: str, extras: str, word_limit: str, target_audience: str, tone: str) -> str:
    #     """
    #     Create a prompt for generating social media content using chain of thought approach.
        
    #     Arguments:
    #     ----------
    #         platform                    {str}            : Social media platform (e.g., "Facebook", "LinkedIn", "Instagram")
            
    #         topic                       {str}            : Main topic of the post
            
    #         brief                       {str}            : Brief description or key points to include
            
    #         extras                 {str, optional}       : Any additional information to consider
            
    #         word_limit             {int, optional}       : Maximum word count for the post
            
    #         target_audience        {str, optional}       : Target audience for the post
            
    #         tone                   {str, optional}       : Desired tone of the post
        
    #     Returns:
    #     --------
    #         post                         {str}           : Formatted prompt for the model
        
    #     """
        
    #     prompt = f"""
    #     Task: Create a {platform} post about {topic}.

    #     Let me think through this step by step:

    #         1. Platform considerations:
    #             - Platform: {platform}
    #             - Ideal post length: Within {word_limit} words
    #             - Format needed: Post heading, post content, 5 hashtags, at least 2 emojis

    #         2. Content planning:
    #             - Main topic: {topic}
    #             - Key points from brief: {brief}
    #             - Target audience: {target_audience}
    #             - Tone to use: {tone}
    #     """
                
    #     if extras:
    #         prompt += f"""- Additional considerations: {extras}"""
                    
    #     prompt += f"""
    #         3. Content structure:
    #             - Start with an attention-grabbing heading
    #             - Follow with informative and engaging content
    #             - End with 5 relevant hashtags
    #             - Include at least 2 appropriate emojis
    #             - Ensure the total post stays within {word_limit} words

    #         Now, create a {platform} post about {topic} for {target_audience} audience in a {tone} tone that includes:
    #             - A compelling heading
    #             - Engaging content
    #             - 5 relevant hashtags
    #             - At least 2 emojis
                
    #         Final Post:
    #     """
        
    #     return prompt.strip()
    
    def generate_post(self, platform: str, topic: str, brief: str, extras: str, word_limit: str, target_audience: str, tone: str) -> str:
        """
        Generate a social media post using the fine-tuned T5 model.
        
        Arguments:
        ----------
            platform                   {str}           : Social media platform (e.g., "Facebook", "LinkedIn", "Instagram")
            
            topic                      {str}           : Main topic of the post
            
            brief                      {str}           : Brief description or key points to include
            
            extras                {str, optional}      : Any additional information to consider
            
            word_limit            {int, optional}      : Maximum word count for the post
            
            target_audience       {str, optional}      : Target audience for the post
            
            tone                  {str, optional}      : Desired tone of the post
        
        Returns:
        --------
            generated_text             {str}           : Generated social media post
        
        Raises:
        -------
            RuntimeError: If generation fails at the runtime
        """
        
        # prompt              = self.create_prompt(platform         = platform,
        #                                          topic            = topic,
        #                                          brief            = brief,
        #                                          extras           = extras,
        #                                          word_limit       = word_limit,
        #                                          target_audience  = target_audience,
        #                                          tone             = tone
        #                                          )
        
        prompt              = generate_prompt(occasion          = "Republic Day",
                                              brief             = "Celebrate the spirit of patriotism",
                                              platform          = "LinkedIn",
                                              target_audience   = "professionals",
                                              tone              = "inspirational",
                                              extra_details     = ""
                                              )
        
        try:
            inputs          = self.tokenizer(prompt, 
                                             return_tensors  = "pt", 
                                             padding         = True, 
                                             truncation      = True,
                                             max_length      = 512 
                                             ).to(self.device)
            
            with torch.no_grad():
                outputs      = self.model.generate(**inputs,
                                                   max_length              = 1024,      # Maximum length of the generated sequence.
                                                   min_length              = 100,       # Minimum length of the generated sequence to avoid very short outputs.
                                                   temperature             = 0.5,       # Controls randomness; lower values make output more deterministic.
                                                   top_p                   = 0.92,      # Nucleus sampling: limits sampling to top tokens whose cumulative probability exceeds `top_p`.
                                                   do_sample               = True,      # Enables sampling instead of greedy/beam search for more diverse outputs.
                                                   no_repeat_ngram_size    = 5,         # Prevents repetition of n-grams to reduce redundancy.
                                                   repetition_penalty      = 1.2,       # Penalizes repeated tokens to encourage diversity
                                                   early_stopping          = False,     # Stops generation once EOS token is predicted or max/min lengths are met.
                                                   eos_token_id            = self.tokenizer.eos_token_id
                                                   )
            
            # Decode output
            generated_text    = self.tokenizer.decode(outputs[0], skip_special_tokens = True)
            
            # final_post        = self._extract_final_post(generated_text)
            
            # if not self._is_valid_post(final_post):
            #     final_post    = self._extract_from_raw_text(generated_text)
            
            return generated_text
            
            # return final_post
            
            
             
        except Exception as e:
            raise RuntimeError(f"Generation failed: {str(e)}")
    
    
    # def _is_valid_post(self, post_text: str) -> bool:
    #     """
    #     Check if the extracted post has the expected social media post structure.
        
    #     Arguments:
    #     ----------
    #         post_text               {str}           : Text to validate
            
    #     Returns:
    #     --------
    #         is_valid                {bool}          : Whether the text appears to be a valid post
    #     """
        
    #     has_hashtags          = '#' in post_text
        
    #     has_sufficient_length = len(post_text.split()) >= 20
        
    #     no_planning_language  = not any(marker in post_text.lower() for marker in 
    #                                   ["platform considerations", "content planning", "content structure",
    #                                    "let me think", "step by step"])
        
    #     return has_hashtags and has_sufficient_length and no_planning_language
    
    
    # def _extract_from_raw_text(self, text: str) -> str:
    #     """
    #     Extract social media post from raw generated text by identifying post-like features.
        
    #     Arguments:
    #     ----------
    #         text                   {str}           : Raw generated text
            
    #     Returns:
    #     --------
    #         post                   {str}           : Extracted social media post
    #     """
        
    #     lines                   = text.split('\n')
    #     hashtag_line_indices    = [i for i, line in enumerate(lines) if '#' in line]
        
    #     if hashtag_line_indices:
    #         first_hashtag_index = min(hashtag_line_indices)
            
    #         start_index         = max(0, first_hashtag_index - 10)
            
    #         for i in range(start_index, first_hashtag_index):
    #             line            = lines[i].strip()
    #             if line and (line.isupper() or line.startswith('## ') or line.startswith('# ') or 
    #                        any(emoji in line for emoji in ['📣', '🔥', '💡', '✨', '🚀'])):
    #                 start_index = i
    #                 break
            
    #         end_index           = max(hashtag_line_indices) + 3 
    #         end_index           = min(end_index, len(lines))
            
    #         return '\n'.join(lines[start_index:end_index]).strip()
        
    #     emoji_pattern           = re.compile(r'[\U0001F300-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251\U0001f926-\U0001f937]')
    #     emoji_lines             = [i for i, line in enumerate(lines) if emoji_pattern.search(line)]
        
    #     if emoji_lines:
    #         start_line          = max(0, min(emoji_lines) - 5)
    #         end_line            = min(len(lines), max(emoji_lines) + 3)
            
    #         return '\n'.join(lines[start_line:end_line]).strip()
        
    #     text_chunks             = [chunk for chunk in text.split('\n\n') if len(chunk.strip()) > 50]
        
    #     if text_chunks:
    #         return text_chunks[-1].strip()
        
    #     for marker in ["Final Post:", "Here's the post:", "The post:"]:
            
    #         if marker in text:
    #             return text.split(marker)[-1].strip()
                
    #     return text.strip()
    
    # def _extract_final_post(self, text: str) -> str:
    #     """
    #     Helper method to extract the final post content if the model 
    #     generated content that includes planning steps.
        
    #     Arguments:
    #     ----------
    #         text                      {str}           : Generated text that may include planning
        
    #     Returns:
    #     --------
    #         final_post                {str}           : Extracted final post content
        
    #     """
    #     # Check for explicit post markers
    #     markers              = ["Final Post:",
    #                             "Here's the post:",
    #                             "The post:",
    #                             "Here's the final post:",
    #                             "Final content:"
    #                             ]
        
    #     for marker in markers:
            
    #         if marker in text:
    #             post_content  = text.split(marker, 1)[1].strip()
                
    #             if "\n\n" in post_content and len(post_content.split("\n\n")[0]) > 50:
    #                 return post_content.split("\n\n")[0].strip()
                
    #             return post_content
        
    #     lines                 = text.strip().split('\n')
        
    #     hashtag_indices       = [i for i, line in enumerate(lines) if '#' in line and not line.startswith('###')]
        
    #     if hashtag_indices:
    #         last_hashtag_idx  = max(hashtag_indices)
    #         first_hashtag_idx = min(hashtag_indices)
            
    #         post_start_idx    = 0
    #         for i in range(first_hashtag_idx-1, -1, -1):
    #             if i < 0:
    #                 break
                    
    #             line          = lines[i].strip()
                
    #             if (line.startswith("3. Content structure:") or 
    #                 line.startswith("Now, create a") or
    #                 "Final Post:" in line):
    #                 post_start_idx = i + 1
                    
    #                 break
                    
    #             if line == "" and i > 0 and lines[i-1].strip() != "":
    #                 post_start_idx = i + 1
    #                 break
            
    #         post_end_idx           = min(last_hashtag_idx + 3, len(lines))
            
    #         extracted_post         = '\n'.join(lines[post_start_idx:post_end_idx]).strip()
    #         return extracted_post
        
    #     heading_indicators         = ["##", "!", "?", "💡", "🔥", "📢", "✨"]
    #     heading_indices            = [i for i, line in enumerate(lines) if any(ind in line for ind in heading_indicators)]
        
    #     if heading_indices:
    #         start_idx              = min(heading_indices)
    
    #         for i in range(start_idx, len(lines)):
                
    #             if "platform considerations:" in lines[i].lower() or "content planning:" in lines[i].lower():
    #                 return '\n'.join(lines[start_idx:i]).strip()
            
    #         return '\n'.join(lines[start_idx:]).strip()
        
    #     clean_text                 = re.sub(r'(?i)Task: Create a.*step by step:', '', text)
    #     clean_text                 = re.sub(r'(?i)(1\. Platform considerations:|2\. Content planning:|3\. Content structure:).*?(?=\n\n|$)', '', clean_text)
        
    #     # Remove any remaining numbered list items that are likely part of planning
    #     clean_text                 = re.sub(r'^\s*-\s*(Platform|Ideal post|Format needed|Main topic|Key points|Target audience|Tone|Start with|Follow with|End with).*$', '', clean_text, flags=re.MULTILINE)
        
    #     return clean_text.strip()