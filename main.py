import os
import json
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
GCP_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
GCP_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
GCS_BUCKET = os.getenv("GCS_BUCKET")
GEMINI_MODEL_NAME = "gemini-1.5-flash-001"

# Ensure all necessary environment variables are set
if not all([GCP_PROJECT_ID, GCP_LOCATION, GCS_BUCKET]):
    raise ValueError("Missing one or more environment variables. Please check your .env file.")

# Initialize Vertex AI
try:
    from google.cloud import aiplatform
    aiplatform.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
    from vertexai.generative_models import GenerativeModel
    gemini_model = GenerativeModel(GEMINI_MODEL_NAME)
except Exception as e:
    print(f"Error initializing Vertex AI or loading models: {e}")
    print("Please ensure you have authenticated to GCP (gcloud auth application-default login) and enabled Vertex AI API.")
    exit()

from prompt_generator import generate_veo3_prompt_with_gemini
from video_generator import generate_video_with_veo3

# --- Step 1: Collect User Input ---
def get_user_product_details():
    print("--- Product Advertisement Video Generator ---")
    product_name = input("Enter your product's name (e.g., 'NewberryAI', 'EcoGlow Smart Garden'): ").strip()
    product_description = input("Provide a detailed description of your product and its main benefits: ").strip()
    ad_brief = input("Describe what you want to include in the 8-second ad (e.g., 'futuristic, sleek, problem-solution format, highlight efficiency'): ").strip()

    if not all([product_name, product_description, ad_brief]):
        print("All fields are required. Please try again.")
        return get_user_product_details()
    return product_name, product_description, ad_brief

# --- Main Pipeline Execution ---
if __name__ == "__main__":
    product_name, product_description, ad_brief = get_user_product_details()

    # Step 2: Generate Veo3 Prompt
    veo3_prompt = generate_veo3_prompt_with_gemini(gemini_model, product_name, product_description, ad_brief)

    if veo3_prompt:
        print("\n--- Generated Veo3 Prompt ---")
        print(veo3_prompt)
        print("\n-----------------------------")

        # Step 3: Generate Video with Veo3
        output_gcs_uri = f"gs://{GCS_BUCKET}/{product_name.replace(' ', '_')}-{int(time.time())}.mp4"
        final_video_url = generate_video_with_veo3(veo3_prompt, output_gcs_uri)

        if final_video_url:
            print("\n--- Video Generation Complete! ---")
            print(f"Your advertisement video is available at: {final_video_url}")
        else:
            print("\nVideo generation failed or URL not found.")
    else:
        print("\nFailed to generate a valid prompt. Video generation aborted.")