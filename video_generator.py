import time
from google import genai
from google.genai.types import GenerateVideosConfig

def generate_video_with_veo3(veo3_prompt, output_gcs_uri):
    print("\n--- Sending Prompt to Vertex AI for Video Generation ---")

    try:
        client = genai.Client()

        # --- Configuration ---
        model_to_use = "veo-3.0-generate-preview"

        video_config = GenerateVideosConfig(
            aspect_ratio="16:9",
            output_gcs_uri=output_gcs_uri,
        )

        print(f"Starting video generation for prompt: '{veo3_prompt}'")

        operation = client.models.generate_videos(
            model=model_to_use,
            prompt=veo3_prompt,
            config=video_config,
        )

        print(f"Operation started with name: {operation}")
        print("Polling for completion... (this may take a few minutes)")

        while not operation.done:
            time.sleep(25)
            operation = client.operations.get(operation)
            print(f"Operation status: {'Done' if operation.done else 'In progress'}")

        if operation.response:
            generated_video = operation.result.generated_videos[0]
            print("Video generation complete!")
            return generated_video.video.uri
        else:
            print("Operation failed or returned no response.")
            return None

    except Exception as e:
        print(f"An unexpected error occurred with Vertex AI API: {e}")
        return None
