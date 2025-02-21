#     ::::::::----------------------------Done By Amrit Bag--------------------------------:::::::
#     --------------------------------------------------------------------------------------------

from fastapi import FastAPI
from pydantic import BaseModel
from src.prompts.prompts import llm_prompt
from src.model_inference.llm_inference import LLMInference



app = FastAPI()
class Prompt(BaseModel):
    company_name      : str
    platform          : str
    topic             : str
    extra_details     : str
    occasion          : str



@app.get("/")
def start():
    return {"Meaasge":"Welcome to LLMs"}

@app.post("/llm")
def gen_prompt(prompt : Prompt):
    company_name     = prompt.company_name
    platform         = prompt.platform
    topic            = prompt.topic
    extra_details    = prompt.extra_details
    occation         = prompt.occasion


    prompt = llm_prompt(platform        = platform,
                        topic           = topic,
                        company_name   = company_name,
                        extra_details  = extra_details,
                        occasion       = occation )
    
    #return {"prompts":prompt}

    
    infer = LLMInference(model_path='src/base_models/falcon1b/model',
                     tokenizer_path='src/base_models/falcon1b/tokenizer', 
                     lora_path='results/llm_results/pipeline_finetuning_v9')
    
    result = infer.generate(prompt = prompt)

    return {"Generaed Prompt":prompt,
            
            "Genarated Content ":result}



