## ----- DONE BY SUBHAS MUKHERJEE -----

# Dependencies
import re
import os
import cv2
import json
import torch
import textwrap
import warnings
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from peft import PeftModel
from llama_cpp import Llama
import matplotlib.font_manager as fm
from diffusers import StableDiffusionXLPipeline
from diffusers import DPMSolverMultistepScheduler
from diffusers import StableDiffusionXLImg2ImgPipeline


# Clear MPS Cache
torch.mps.empty_cache()

# Suppress extra logs
os.environ["LLAMA_CPP_LOG_LEVEL"] = "ERROR"

# Suppress warnings
warnings.filterwarnings("ignore")


# ------------------------------------------------------------------------------------
#  Prompt Formatting using Zephyr LLM
# ------------------------------------------------------------------------------------
def zephyr_prompt_enhancer(zephyr_model_path: str, positive_prompt: str, negative_prompt: str, color_scheme: str) -> dict:
    """
    Uses Zephyr-7B locally via Llama.cpp for SDXL model's raw prompt enhancement
    """
    # Initialize the Zephyr model
    zephyr_model = Llama(model_path   = zephyr_model_path,
                         n_ctx        = 8192,
                         verbose      = False,
                         n_batch      = 1024,
                        )

    system_prompt = (f"""You are an advanced AI model specializing in extracting structured information from natural language descriptions and optimizing prompts for Stable Diffusion XL (SDXL).
                         
                         Your task is to:
                         1. **Extract the following elements from the user's input prompt using natural language understanding**:
                            - `image subject`
                            - `image content`
                            - `context of the image`
                            - `color theme of the image` (if color_scheme is provided)
                            - `image layout` (if applicable)
    
                         2. **Construct the final positive prompt using the extracted components** in this format:
                            ```
                            positive_prompt = image subject + image content + context of the image + color theme of the image + image layout
                            ```
                         3. Final positive prompt should be strictly containing at most **77 tokens**.
                         
                         4. **Generate an SDXL-optimized negative prompt** to reduce distortions and unwanted artifacts.

                         5. **Output the result as a strict JSON object** in this format:
                            ```
                            {{
                              "positive_prompt": "generated positive prompt text here",
                              "negative_prompt": "generated negative prompt text here"
                            }}
                            ```

                         Ensure **correct extraction and formatting** while keeping the output optimized for SDXL.
                      """).strip()


    user_prompt   = f"""User Input Prompt:
                        "{positive_prompt}"

                        User Negative Prompt:
                        "{negative_prompt}"

                        Preferred Color Scheme:
                        "{color_scheme}" 
                     """ 

    full_prompt   = system_prompt + "\n\n" + user_prompt

    try:
        # Run inference
        response   = zephyr_model(prompt      = full_prompt, 
                                  stream      = False, 
                                  max_tokens  = 2048,
                                  top_p       = 0.9,
                                  top_k       = 85,
                                  temperature = 0.5,
                                 )

        # Extract JSON from the response
        json_match = re.search(r'\{.*\}', response["choices"][0]["text"], re.DOTALL)

        if json_match:
            enhanced_prompt = json_match.group(0).strip()

        else:
            return {"error": "No valid JSON found", "raw_text": response["choices"][0]["text"]}

        # Try to parse the response as JSON
        try:
            json_response = json.loads(enhanced_prompt)
            return json_response

        except json.JSONDecodeError as e:
            return {"error": f"JSON parsing failed: {str(e)}", "raw_text": enhanced_prompt}

    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


# ----------------------------------------------------------
# TEXT IMAGE GENERATION USING PIL
# ----------------------------------------------------------
class TextImageGenerator:
    """
    A class to generate text overlays and apply them to AI-generated images, which
    supports HTML-like text formatting and dynamic word wrapping etc.
    """
    def __init__(self, image_size: tuple = (800, 600), default_font: str = "Arial", default_color: str = "black"):
        """
        Initialize the text overlay generator
        
        Arguments:
        ----------
            image_size   { tuple } : Tuple representing the width and height of the image
            
            default_font  { str }  : Default font name for text rendering

            default_color { str }  : Default color of the text

        """
        self.image_size    = image_size
        self.fonts         = self.get_available_fonts()
        self.default_font  = self.fonts.get(default_font, "arial.ttf")
        self.default_color = default_color
        
        # Predefined font sizes
        self.FONT_SIZES    = {"tiny"        : 16, 
                              "small"       : 24, 
                              "medium"      : 32,
                              "large"       : 48, 
                              "extra-large" : 64, 
                              "huge"        : 80,
                             }
        
        # HTML-like tag sizes
        self.TAG_SIZES     = {"H1"   : "huge", 
                              "H2"   : "extra-large", 
                              "H3"   : "large", 
                              "H4"   : "medium", 
                              "body" : "small",
                              "note" : "small",
                             }

        # Bracket-level font sizes
        self.BRACKET_SIZES = {1 : "medium", 
                              2 : "large", 
                              3 : "extra-large", 
                              4 : "huge"}


    def get_available_fonts(self) -> dict:
        """
        Retrieve all available system fonts in a readable format

        Returns:
        --------
            { dict } : A dictionary mapping font names to file paths
        """
        font_dict = dict()

        for font_path in fm.findSystemFonts(fontext='ttf'):
            try:
                font_name            = fm.FontProperties(fname = font_path).get_name()
                font_dict[font_name] = font_path
            
            except RuntimeError:
                # Skip unreadable fonts
                continue

        return font_dict


    def parse_bracket_sized_text(self, text: str) -> list:
        """
        Identify words in brackets and assign appropriate font sizes
        
        Arguments:
        ----------
            text { str } : Input text with bracket-based size formatting

        Returns:
        --------
             { list }    : A list of tuples (word, font_size)
        """
        words           = text.split()
        formatted_words = list()

        for word in words:
            bracket_count = word.count("(")
            clean_word    = word.strip("()")  
            font_size     = self.BRACKET_SIZES.get(bracket_count, "small")  

            formatted_words.append((clean_word, font_size))

        return formatted_words


    def parse_html_like_tags(self, text: str) -> list:
        """
        Detect headings, body text, and notes, and assign formatting

        Arguments:
        ----------
            text { str } : Input text with HTML-like tags
        
        Returns:
        --------
               { list }  : A structured list of (content, font_size, is_new_line)
        """
        structured_text = list()
        lines           = text.split("\n")

        for line in lines:
            match = re.match(r"<(H[1-4]|body|note)>(.*?)</\1>", line.strip(), re.IGNORECASE)
            
            if match:
                tag, content = match.groups()
                font_size    = self.TAG_SIZES.get(tag.upper(), "small")
                structured_text.append((content.strip(), font_size, True)) 
            
            else:
                # Process bracket-based font sizing for regular text
                bracket_formatted = self.parse_bracket_sized_text(line.strip())
                structured_text.extend([(word, size, False) for word, size in bracket_formatted]) 

        return structured_text


    def wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list:
        """
        Wrap text to fit within the image width
        
        Arguments:
        ----------
            text             { str }        : Input text to wrap

            font { ImageFont.FreeTypeFont } : The font used for measurement

            max_width        { int }        : The maximum allowed width for text

        Returns:
        --------
                     { list }               : A list of (wrapped line, text_width)
        """
        lines             = list()

        # Use textwrap as an initial approximation
        estimated_wrapped = textwrap.wrap(text, width = 40)

        for esttimated_line in estimated_wrapped:
            words        = esttimated_line.split()
            current_line = ""

            for word in words:
                # Avoid extra spaces
                test_line  = f"{current_line} {word}".strip()  
                
                # Get pixel width
                text_width = font.getlength(test_line)  

                # Add word if within limit
                if (text_width <= max_width):
                    current_line = test_line 
                
                else:
                    lines.append((current_line, font.getlength(current_line)))
                    # Start new line
                    current_line = word
            
            # Add last line
            if current_line:  
               lines.append((current_line, font.getlength(current_line)))
        
        return lines

    
    def generate_text_overlay(self, text: str, font_name: str = "Arial", font_color: str = None) -> Image.Image:
        """
        Creates a transparent text overlay with user-defined font and color
        
        Arguments:
        ----------
        text          { str }  : The text content to render

        font_name     { str }  : The font to use

        font_color    { str }  : The text color

        Returns:
        --------
               { Image.Image } : A PIL Image object with the text overlay
        """
        image          = Image.new(mode  = "RGBA", 
                                   size  = self.image_size, 
                                   color = (255, 255, 255, 0)) 

        draw           = ImageDraw.Draw(image)

        # Choose the correct font file
        font_path      = self.fonts.get(font_name, self.default_font)

        # Use default color if none provided
        text_color     = font_color if font_color else self.default_color

        # Parse structured text
        text_structure = self.parse_html_like_tags(text = text)
        
        # Start position
        y_offset       = 50
        
        for content, font_size, is_new_line in text_structure:
            font          = ImageFont.truetype(font = font_path, 
                                               size = self.FONT_SIZES[font_size])

            # Wrap text to fit within image width
            wrapped_lines = self.wrap_text(text      = content, 
                                           font      = font, 
                                           max_width = self.image_size[0] - 100)

            for line, text_width in wrapped_lines:
                # Center alignment
                x_offset = (self.image_size[0] - text_width) // 2 

                draw.text(xy   = (x_offset, y_offset), 
                          text = line, 
                          font = font, 
                          fill = text_color,
                         )

                # Adjust line spacing
                y_offset += font.size + 5 
            
            # Extra spacing for new paragraphs 
            if is_new_line:
                y_offset += 10 
        
        return image


# ----------------------------------------------------------------------
# Image Processing : Logo and Text Image Overlay on AI generated image 
# ----------------------------------------------------------------------
class ImageProcessor:
    """
    Handles image processing operations like overlaying logos and text
    """
    
    @staticmethod
    def overlay_logo_on_image(base_image: Image.Image, logo_path: str, logo_position: str = "bottom-right", margin: int = 5) -> Image.Image:
        """
        Overlays a logo on the base image at the specified position
        
        Arguments:
        ----------
            base_image { Image.Image } : Base image to overlay the logo on

            logo_path      { str }     : Path to the logo image

            logo_position  { str }     : Position for logo placement ('top-left', 'top-right', 'bottom-left', 'bottom-right')

            margin         { int }     : Margin from the edge in pixels
            
        Returns:
        --------
              { Image.Image }          : Image with logo overlaid
        """
        # Ensure base image is in RGBA mode for transparency
        if (base_image.mode != 'RGBA'):
            base_image = base_image.convert(mode = 'RGBA')
        
        # Load and resize logo
        try:
            logo = Image.open(logo_path).convert(mode = 'RGBA')
            
            # Resize logo to be at most 10% of the base image width
            max_width = int(base_image.width * 0.1)
            if (logo.width > max_width):
                ratio    = max_width / logo.width
                new_size = (max_width, int(logo.height * ratio))
                logo     = logo.resize(new_size, Image.LANCZOS)
            
            # Determine position
            if (logo_position == "top-left"):
                final_logo_position = (margin, margin)

            elif (logo_position == "top-right"):
                final_logo_position = (base_img.width - logo.width - margin, margin)

            elif (logo_position == "bottom-left"):
                final_logo_position = (margin, base_img.height - logo.height - margin)
                
            else:
                # Default to bottom-right  
                final_logo_position = (base_img.width - logo.width - margin, base_img.height - logo.height - margin)
            
            # Create a new image with the same size as the base image
            logo_overlayed_image = Image.new(mode = 'RGBA', 
                                             size = base_image.size)
            
            # Paste the base image
            logo_overlayed_image.paste(base_image, (0, 0))
            
            # Paste the logo with transparency
            logo_overlayed_image.paste(logo, final_logo_position, logo)
            
            return logo_overlayed_image
            
        except Exception as e:
            print(f"Error overlaying logo: {e}")
            return base_image
    
    @staticmethod
    def overlay_text_image_on_ai_image(ai_image: Image.Image, text_image: Image.Image) -> Image.Image:
        """
        Fully overlays the transparent text image on an AI-generated image

        Arguments:
        ----------
        ai_image   { Image.Image } : The AI-generated image

        text_image { Image.Image } : The text overlay image

        Returns:
        --------
               { Image.Image }     : The final PIL Image object with the text overlay
        """
        try: 
            ai_image    = ai_image.convert("RGBA")

            # Ensure both images have the same size
            text_image  = text_image.resize(ai_image.size) 

            # Use alpha compositing to overlay text_image fully
            final_image = Image.alpha_composite(ai_image, text_image)

            return final_image
        
        except Exception as e:
            print(f"Error overlaying text image: {e}")
            raise


# ---------------------------------------------
# SDXL Image Generation with LoRA Adapter
# ---------------------------------------------
class SDXLGenerator:
    """
    A class to handle LoRA fine-tuned SDXL image generation
    """
    def __init__(self, device: str, precision: torch.dtype, sdxl_base_path: str, finetuned_lora_path: str, zephyr_model_path: str, image_size: tuple):
        """
        Initializes the SDXL pipeline and loads LoRA adapter weights

        Arguments:
        ----------
            device 

            precision

            sdxl_base_path

            finetuned_lora_path

            image_size                    { tuple }  : Desired image resolution ("512x512" or "1024x1024")
        """
        try:
            # Set device and precision as the class attributes
            self.device                  = device
            self.precision               = precision
            self.image_size              = image_size
            
            print(f"\nLoading SDXL-base model ...")
            # Load the base SDXL model
            self.sdxl_pipeline           = StableDiffusionXLPipeline.from_pretrained(pretrained_model_name_or_path = sdxl_base_path,
                                                                                     torch_dtype                   = self.precision,
                                                                                    )

            # Move the SDXL Pipeline to desired device
            self.sdxl_pipeline.to(self.device)

            # Load the SDXL Finetuned LoRA adapter
            print(f"\nLoading LoRA adapter from {finetuned_lora_path}...")
            self.sdxl_pipeline.unet      = PeftModel.from_pretrained(self.sdxl_pipeline.unet, 
                                                                     finetuned_lora_path)

            # Adjust LoRA influence
            lora_scale                   = 0.5
            # Manually scale LoRA influence instead of set_adapter_scales()
            for name, param in self.sdxl_pipeline.unet.named_parameters():
                if "lora" in name:  # Ensure only LoRA layers are affected
                    param.data *= lora_scale  # Adjust 0.5 to control LoRA influence

            print(f"\nLoRA adapter loaded successfully with LoRA Scale: {lora_scale} ....\n")
            
            print(f"\nLoading SDXL-refiner model ...")
            # Load the refiner model
            self.refiner                 = StableDiffusionXLImg2ImgPipeline.from_pretrained(pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-refiner-1.0",
                                                                                            torch_dtype                   = precision)
            # Move the refiner model to desired device
            self.refiner.to(self.device)
            print(f"\nSDXL refiner model has been loaded successfully")

            # Set the general image generation scheduler
            self.sdxl_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.sdxl_pipeline.scheduler.config)
            print(f"\nDPM++ 2M Karras Scheduler added successfully")
            
            self.zephyr_model_path       = zephyr_model_path

            # Set up the image processor
            self.image_processor         = ImageProcessor()

            # Setup Text Image Generator
            self.text_image_generator    = TextImageGenerator(image_size = self.image_size)
        
        except Exception as e:
            print(f"\nError loading SDXL Model with LoRA: {e}")
            raise


    def generate_image(self, positive_prompt: str, negative_prompt: str, color_theme: str, num_inference_steps: int = 20, 
                       guidance_scale: float = 7.5, logo_presence: str = "yes", logo_path: str = None, logo_position: str = "bottom-right",
                       seed: int = None, font_style: str = "Arial", font_color: str = "black", image_text: str = None) -> Image.Image:
        """
        Generates an image using SDXL LoRA fine-tuned model
        
        Arguments:
        ----------
            positive_prompt                { str }   : Key features to emphasize in the image generation

            negative_prompt                { str }   : Elements to avoid in the image generation

            color_theme                    { str }   : Preferred color theme (e.g., "navy blue", "pastel orange")
            
            num_inference_steps            { int }   : Number of denoising steps
            
            guidance_scale                { float }  : Scale for classifier-free guidance
            
            logo_presence                  { str }   : Presence of the brand logo in the generated image
            
            logo_path                      { str }   : Path to the pre-saved logo image (if any)
            
            logo_position                  { str }   : Position of the brand logo placement in the image such as: 
                                                       top-left, top-right, bottom-left, bottom-right etc. Default is top-left
            
            seed                           { int }   : Random seed for reproducibility

            font_style                     { str }   : The font to use for writing text on the image

            font_color                     { str }   : The color of the text font on the image

            image_text                     { str }   : The text to be written on the image
            
        Returns:
        --------
                        { Image.Image }              : AI Generated image with color theme, desired properties and optional brand logo on it
        """
        # Format user provided prompts as per conventions of SDXL prompting 
        try:
            enhanced_prompts_dict                  = zephyr_prompt_enhancer(zephyr_model_path = self.zephyr_model_path, 
                                                                            positive_prompt   = positive_prompt,
                                                                            negative_prompt   = negative_prompt,
                                                                            color_scheme      = color_theme,
                                                                           )

            formatted_positive, formatted_negative = enhanced_prompts_dict["positive_prompt"], enhanced_prompts_dict["negative_prompt"]
        
        except: 
            formatted_positive                     = positive_prompt
            formatted_negative                     = negative_prompt
        
            # Append color theme to the positive prompt
            if color_theme:
                formatted_positive = f"{formatted_positive} with color scheme {color_theme}"
        """
        formatted_positive                     = positive_prompt
        formatted_negative                     = negative_prompt
        """
        print(f"\nUsing positive prompt: {formatted_positive}")
        print(f"\nUsing negative prompt: {formatted_negative}\n")
        
        # Set random seed if provided
        if seed is not None:
            generator = torch.Generator(device = self.device).manual_seed(seed)
        
        else:
            generator = None

        # Generate the required image using SDXL LoRA Fine-tuned model
        print(f"\nProcess of the making of AI-generated image is running ... ")
        sdxl_output          = self.sdxl_pipeline(prompt              = formatted_positive,
                                                  negative_prompt     = formatted_negative,
                                                  width               = self.image_size[0],
                                                  height              = self.image_size[1],
                                                  num_inference_steps = num_inference_steps,
                                                  guidance_scale      = guidance_scale,
                                                  generator           = generator
                                                 )
        
        # Get the generated image
        sdxl_generated_image = sdxl_output.images[0]

        # Pass the SDXL Generated image through refiner
        refiner_output       = self.refiner(prompt              = formatted_positive, 
                                            negative_prompt     = formatted_negative,
                                            generator           = generator, 
                                            image               = sdxl_generated_image,
                                            strength            = 0.6,
                                            num_inference_steps = num_inference_steps,
                                            guidance_scale      = guidance_scale,
                                           )

        refined_image        = refiner_output.images[0]

        print(f"\nGot AI-Generated image by SDXL Fine-Tuned model successfully.\n")
        
        print (f"Writing the user provided text on the AI-Generated image ...")
        if (image_text != None):
            # Generate text on the image with transparent background
            text_generated_image = self.text_image_generator.generate_text_overlay(text       = image_text, 
                                                                                   font_name  = font_style, 
                                                                                   font_color = font_color,
                                                                                  )

            # Overlay text on AI-generated image
            text_overlayed_image = self.image_processor.overlay_text_image_on_ai_image(ai_image   = refined_image, 
                                                                                       text_image = text_generated_image,
                                                                                      )
        
        else:
            text_overlayed_image = sdxl_generated_image

        print(f"\nAdding logo to the AI-generated image ...\n")
        # Overlay logo if provided on the SDXL Generated + Text overlayed image
        if ((logo_presence == "yes") and (logo_path and os.path.exists(logo_path))):
            logo_added_image = self.image_processor.overlay_logo_on_image(base_image    = text_overlayed_image, 
                                                                          logo_path     = logo_path, 
                                                                          logo_position = logo_position)
        else:
            print(f"\nWarning: Logo cannot be added as: either logo path is incorrect or you haven't provided logo presence = yes")
            logo_added_image = text_overlayed_image

        final_generated_image = logo_added_image
        print (f"\nImage generation process completed")
        return final_generated_image



# Main execution
if __name__ == "__main__":

    # Paths to models
    BASE_MODEL_PATH       = "models/sdxl_finetuned/base_model"
    ADAPTER_MODEL_PATH    = "models/sdxl_finetuned/sdxl_lora_adapter"
    ZEPHYR_MODEL_PATH     = "models/zephyr_7b_beta/zephyr-7b-beta.Q5_K_M.gguf"
    IMAGE_SIZE            = (1024, 1024)

    # System parameters
    DEVICE                = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    PRECISION             = torch.float32 if DEVICE == "mps" else torch.float16
    
    # Parameters: Given by User
    positive_prompt       = "Futuristic AI-driven trading app named ITO-TRADE UI displayed on only one smartphone screen with clear English fonts, real-time stock graphs or trading insights or financial analytics; everything in intricate and clear english text fonts. Dark theme, sleek & minimal. White background, black with red & green highlights. High-tech digital elements."
    negative_prompt       = "blurry, cluttered, too many objects, distorted, low quality, unnatural colors, bad composition"
    num_inference_steps   = 200
    guidance_scale        = 9.0
    color_theme           = "white and green"
    logo_presence         = "yes"
    logo_path             = "data/company_logo/logo.jpg"
    logo_position         = "top-left"
    image_generation_seed = 42
    font_style            = "Arial"
    font_color            = "red"
    image_text            = """<H1>Ito-Trade</H1>\n<H3>Your Reliable Companion for Security Trading</H3>\n<body>Join us in our innovation journey through AI</body>"""
    save_path             = "results/image_generation_results/trading_app_launching_image.png"
    
    # Ensure save directory exists
    os.makedirs(os.path.dirname("results/image_generation_results"), exist_ok = True)

    # Initialize generators
    sdxl_generator        = SDXLGenerator(device              = DEVICE, 
                                          precision           = PRECISION, 
                                          sdxl_base_path      = BASE_MODEL_PATH,
                                          finetuned_lora_path = ADAPTER_MODEL_PATH,
                                          zephyr_model_path   = ZEPHYR_MODEL_PATH,
                                          image_size          = IMAGE_SIZE,
                                         )
    
    ai_generated_image    = sdxl_generator.generate_image(positive_prompt     = positive_prompt, 
                                                          negative_prompt     = negative_prompt,
                                                          color_theme         = color_theme,
                                                          num_inference_steps = num_inference_steps,
                                                          guidance_scale      = guidance_scale, 
                                                          logo_presence       = logo_presence, 
                                                          logo_path           = logo_path,
                                                          logo_position       = logo_position,
                                                          seed                = image_generation_seed, 
                                                          font_style          = font_style, 
                                                          font_color          = font_color,
                                                          image_text          = image_text,
                                                         )
    # Save the final result
    ai_generated_image.save(save_path)

