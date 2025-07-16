import os
from dotenv import load_dotenv
from google import genai
from google.genai.types import HttpOptions
# Load environment variables from .env file
load_dotenv()

def generate_veo3_prompt_with_gemini(product_name, product_description, ad_brief):
    print("\n--- Generating Veo3 Prompt with Gemini 1.5 Pro ---")

    # Initialize the client within the function
    try:
        gcp_project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        gcp_location = "us-central1" # Using a specific region known to work
        
        if not gcp_project_id:
            raise ValueError("GCP_PROJECT_ID environment variable not set.")
        client = genai.Client(http_options=HttpOptions(api_version="v1"))
        # client = genai.Client(vertexai=True, project=gcp_project_id, location=gcp_location)
    except Exception as e:
        print(f"Error initializing Gemini Client: {e}")
        return None
    llm_prompt = f"""
    You are an expert video prompt engineer for Veo, a cutting-edge text-to-video AI model.
    Your task is to create a highly detailed and visually rich prompt for Veo to generate an 8-second advertisement video.
    The video must be concise, impactful, and clearly convey the product's essence within the 8-second limit.
    Focus on dynamic visual animations, seamless transitions, and evocative imagery.

    **Product Name:** {product_name}
    **Product Description:** {product_description}
    **Advertisement Brief (User's request for tone/theme):** {ad_brief}

    ---
    **Guidelines for Veo Prompt:**
    1.  **Duration:** Strictly 8 seconds.
    2.  **Visual Focus:** Prioritize visual storytelling. Use vivid, descriptive language for scenes, animations, lighting, and transitions.
    3.  **Conciseness:** Every word matters. Be precise.
    4.  **No Explanations:** The output must ONLY be the Veo prompt text.
    5.  **Brand Integration:** Ensure the product name is clearly featured.
    6.  **Final Output Format:** Provide the raw prompt text suitable for a "text_to_video" API.
    ---
    **Example Structure:**
    Generate a highly dynamic, visually captivating 8-second video advertisement for **{product_name}**.
    **Overall Visual Tone:** {ad_brief}, cinematic, high-fidelity, dynamic lighting.
    **Scene 1 (0-3s):** A user looking frustrated with a normal online store, the camera zooms into the screen showing flat 2D images of clothes.
    **Scene 2 (3-6s):** A seamless transition into the EastXWest interface. The user's finger swipes, and the 2D image transforms into a stunning, interactive 3D model of a dress on a mannequin, rotating smoothly.
    **Scene 3 (6-8s):** The 3D model is shown on a real-life model walking down a runway. The EastXWest logo and a call to action, "See Fashion in a New Dimension," appear in sleek, futuristic text.
    **Aspect Ratio:** 16:9
    """

    try:
        model = "gemini-2.5-flash"
        response = client.models.generate_content(
            model=model,
            contents=[llm_prompt]
        )
        veo3_prompt = response.text.strip()
        print("Gemini generated prompt successfully!")
        return veo3_prompt
    except Exception as e:
        print(f"Error generating prompt with Gemini: {e}")
        return None
