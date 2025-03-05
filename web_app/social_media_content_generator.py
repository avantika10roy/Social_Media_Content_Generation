# Dependencies
from fastapi import FastAPI
from fastapi import HTTPException
from pydantic_inputs import TextGenerationRequest
from pydantic_inputs import ImageGenerationRequest
from fastapi.middleware.cors import CORSMiddleware
from pydantic_outputs import TextGenerationResponse
from pydantic_outputs import ImageGenerationResponse
from text_generation_inference import text_generator
from image_generation_inference import image_generator


app = FastAPI(title       = "Text Generation API", 
              description = "Generate AI-powered social media posts.")


# Enable CORS for Streamlit frontend
app.add_middleware(CORSMiddleware,
                   allow_origins     = ["*"],  # Add the frontend URL in production here
                   allow_credentials = True,
                   allow_methods     = ["*"],
                   allow_headers     = ["*"],
                  )

@app.post("/generate-text", response_model = TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    try:
        generated_text = text_generator.generate_post(occasion        = request.occasion,
                                                      brief           = request.brief,
                                                      platform        = request.platform,
                                                      target_audience = request.target_audience,
                                                      tone            = request.tone,
                                                      extra_details   = request.extra_details,
                                                      max_length      = request.max_length,
                                                      temperature     = request.temperature,
                                                      top_p           = request.top_p,
                                                )
        return TextGenerationResponse(generated_text = generated_text)
    
    except Exception as e:
        raise HTTPException(status_code = 500, 
                            detail      = str(e))
    
@app.post("/generate-image",response_model = ImageGenerationResponse)
async def generate_image(request:ImageGenerationRequest):
    try:
        generated_image = image_generator.generate_image()

        pass
    except Exception as e:
        raise HTTPException(status_code = 500,
                            detail = str(e))
# Run using: uvicorn main:app --reload
