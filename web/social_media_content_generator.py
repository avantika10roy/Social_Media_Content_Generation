# Dependencies
from fastapi import FastAPI
from fastapi import HTTPException
from pydantic_inputs import TextGenerationRequest
from fastapi.middleware.cors import CORSMiddleware
from pydantic_outputs import TextGenerationResponse
from text_generation_inference import text_generator
# from image_generation_inference import image_generator


app = FastAPI(title       = "Text Generation API", 
              description = "Generate AI-powered social media posts.")


# Enable CORS for Streamlit frontend
app.add_middleware(CORSMiddleware,
                   allow_origins     = ["*"],  # Add the frontend URL in production here
                   allow_credentials = True,
                   allow_methods     = ["*"],
                   allow_headers     = ["*"],
                  )

@app.post("/generate_text", response_model = TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    try:
        generated_text = text_generator.generate_post(
                                                 company_name    = request.company_name,
                                                 occasion        = request.occasion,
                                                 topic           = request.topic,
                                                 brief           = request.brief,
                                                 extra_details   = request.extra_details,
                                                 platform        = request.platform,
                                                 tone            = request.tone,
                                                 target_audience = request.target_audience,
                                                )
        return TextGenerationResponse(generated_text = generated_text)
    
    except Exception as e:
        raise HTTPException(status_code = 500, 
                            detail      = str(e))

