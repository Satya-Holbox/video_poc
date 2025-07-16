import os
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
GCP_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
GCS_BUCKET = os.getenv("GCS_BUCKET")
# The location is now handled in the prompt_generator and video_generator
# to ensure the correct region is used for each service.

# Ensure all necessary environment variables are set
if not all([GCP_PROJECT_ID, GCS_BUCKET]):
    raise ValueError("Missing one or more environment variables (GCP_PROJECT_ID, GCS_BUCKET). Please check your .env file.")

# Initialize Vertex AI (required for the google-genai library to use the Vertex AI backend)
try:
    from google.cloud import aiplatform
    # We use a specific region for initialization, but it can be overridden in the clients.
    aiplatform.init(project=GCP_PROJECT_ID, location="us-central1")
except Exception as e:
    print(f"Error initializing Vertex AI: {e}")
    print("Please ensure you have authenticated to GCP (gcloud auth application-default login) and enabled the Vertex AI API.")
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
    veo3_prompt = generate_veo3_prompt_with_gemini(product_name, product_description, ad_brief)

    if veo3_prompt:
        print("\n--- Generated Veo3 Prompt ---")
        print(veo3_prompt)
        print("\n-----------------------------")

        # Step 3: Generate Video with Veo3
        # Create a unique name for the video file in the GCS bucket
        video_filename = f"{product_name.replace(' ', '_').lower()}_{int(time.time())}.mp4"
        output_gcs_uri = f"gs://{GCS_BUCKET}/{video_filename}"
        
        print(f"Video will be saved to: {output_gcs_uri}")

        final_video_url = generate_video_with_veo3(veo3_prompt, output_gcs_uri)

        if final_video_url:
            print("\n--- Video Generation Complete! ---")
            print(f"Your advertisement video is available at: {final_video_url}")
        else:
            print("\nVideo generation failed or a URL was not returned.")
    else:
        print("\nFailed to generate a valid prompt. Video generation aborted.")
