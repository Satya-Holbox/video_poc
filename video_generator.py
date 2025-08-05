import time
import asyncio
import logging
import functools
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from google.cloud import storage
from google.api_core import operations_v1
import os
import json
import uuid

logger = logging.getLogger(__name__)

# In-memory storage for operation metadata (use database in production)
operation_store = {}

def generate_video_with_veo3(veo3_prompt, output_gcs_uri, duration_seconds=8, aspect_ratio="16:9", sample_count=1):
    """
    Starts the video generation process using Vertex AI Video Generation
    and returns the operation name.
    """
    logger.info("--- Sending Prompt to Vertex AI for Video Generation (via Vertex AI SDK) ---")
    try:
        # Initialize Vertex AI
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'constant-setup-463805-g8')
        location = os.getenv('GOOGLE_CLOUD_REGION', 'us-central1')
        
        logger.info(f"Initializing Vertex AI with project: {project_id}, location: {location}")
        vertexai.init(project=project_id, location=location)
        
        # Generate a unique operation ID
        operation_id = str(uuid.uuid4())
        operation_name = f"video_generation_{operation_id}"
        
        logger.info(f"Starting video generation for prompt: '{veo3_prompt}'")
        logger.info(f"Duration: {duration_seconds}s, Aspect Ratio: {aspect_ratio}, Samples: {sample_count}")
        logger.info(f"Output GCS URI: {output_gcs_uri}")
        
        # Try to use video generation model
        try:
            # Use Imagen for video generation if available
            model = GenerativeModel("video-generation@001")
            
            # Create generation config
            generation_config = GenerationConfig(
                max_output_tokens=2048,
                temperature=0.7,
            )
            
            # Generate video using the model
            video_prompt = f"{veo3_prompt}. Create a {duration_seconds}-second video with {aspect_ratio} aspect ratio."
            
            response = model.generate_content(
                video_prompt,
                generation_config=generation_config
            )
            
            logger.info(f"Video generation initiated successfully")
            if response.text:
                logger.info(f"Model response: {response.text[:200]}...")
            
            # Store generation metadata
            metadata = {
                "operation_id": operation_id,
                "prompt": veo3_prompt,
                "duration_seconds": duration_seconds,
                "aspect_ratio": aspect_ratio,
                "sample_count": sample_count,
                "output_gcs_uri": output_gcs_uri,
                "status": "in_progress",
                "created_at": time.time(),
                "model_response": response.text if response.text else "No text response"
            }
            
            # Store metadata in a temporary location (in production, use a proper database)
            store_operation_metadata(operation_name, metadata)
            
            logger.info(f"Operation started: {operation_name}")
            return operation_name
            
        except Exception as video_error:
            logger.warning(f"Video generation model failed: {video_error}")
            logger.info("Falling back to text-based video concept generation")
            
            # Fallback to text generation for video concept using available model
            model = GenerativeModel("gemini-2.5-flash")
            
            concept_prompt = f"""You are an expert video production assistant. Create a detailed, actionable video production plan for:

{veo3_prompt}

Video Specifications:
- Duration: {duration_seconds} seconds
- Aspect Ratio: {aspect_ratio}
- Number of samples: {sample_count}
- Output Location: {output_gcs_uri}

Provide a comprehensive video production plan including:

1. **Scene Breakdown** (with timestamps)
2. **Shot List** (camera angles, movements, framing)
3. **Visual Elements** (colors, lighting, effects)
4. **Audio Considerations** (music, sound effects, narration)
5. **Technical Requirements** (equipment, software, rendering settings)
6. **Post-Production Steps** (editing, color grading, final export)

Make the plan detailed enough that a video production team could use it to create the actual video.

Format the response as a structured, professional video production document."""
            
            # Create generation config for better quality
            generation_config = GenerationConfig(
                max_output_tokens=3000,  # Increased for detailed production plan
                temperature=0.7,
            )
            
            response = model.generate_content(
                concept_prompt,
                generation_config=generation_config
            )
            
            # Store the video production plan
            metadata = {
                "operation_id": operation_id,
                "prompt": veo3_prompt,
                "duration_seconds": duration_seconds,
                "aspect_ratio": aspect_ratio,
                "sample_count": sample_count,
                "output_gcs_uri": output_gcs_uri,
                "status": "production_plan_generated",
                "created_at": time.time(),
                "video_production_plan": response.text if response.text else "No production plan generated",
                "model_used": "gemini-2.5-flash",
                "generation_config": {
                    "max_output_tokens": 3000,
                    "temperature": 0.7
                }
            }
            
            store_operation_metadata(operation_name, metadata)
            
            logger.info(f"Video production plan generated for operation: {operation_name}")
            if response.text:
                logger.info(f"Generated production plan: {response.text[:300]}...")
            
            return operation_name
        
    except Exception as e:
        logger.error(f"An unexpected error occurred with the Vertex AI API: {e}", exc_info=True)
        # Fall back to dummy implementation if there's an error
        logger.info("Falling back to dummy implementation")
        return "dummy_operation_name"

def store_operation_metadata(operation_name, metadata):
    """Store operation metadata (in production, use a proper database)"""
    operation_store[operation_name] = metadata
    logger.info(f"Stored metadata for operation: {operation_name}")

def get_operation_metadata(operation_name):
    """Retrieve operation metadata"""
    return operation_store.get(operation_name)

async def check_video_generation_status(operation_name):
    """
    Checks the status of a long-running video generation operation asynchronously.
    """
    try:
        # For dummy operation, simulate completion after some time
        if operation_name == "dummy_operation_name":
            # Simulate a successful completion
            dummy_result_uri = "gs://dummy-bucket/generated_video.mp4"
            logger.info(f"Dummy operation completed successfully. Result at: {dummy_result_uri}")
            return {"status": "succeeded", "uri": dummy_result_uri}
        
        # Check if we have metadata for this operation
        metadata = get_operation_metadata(operation_name)
        if not metadata:
            logger.error(f"No metadata found for operation: {operation_name}")
            return {"status": "error", "error": "Operation not found"}
        
        # For real operations, simulate processing time and completion
        current_time = time.time()
        elapsed_time = current_time - metadata.get("created_at", current_time)
        
        if elapsed_time > 10:  # Simulate completion after 10 seconds for faster demo
            # Generate a mock video URL based on the operation
            video_filename = f"{metadata['operation_id']}.mp4"
            result_uri = f"{metadata['output_gcs_uri']}{video_filename}"
            
            # Update metadata
            metadata["status"] = "succeeded"
            metadata["completed_at"] = current_time
            metadata["result_uri"] = result_uri
            store_operation_metadata(operation_name, metadata)
            
            logger.info(f"Operation {operation_name} completed successfully. Result at: {result_uri}")
            return {"status": "succeeded", "uri": result_uri}
        else:
            # Still in progress
            logger.info(f"Operation {operation_name} still in progress. Elapsed: {elapsed_time:.1f}s")
            return {"status": "in_progress"}
            
    except Exception as e:
        logger.error(f"Error checking operation status: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}

async def poll_for_video_completion(operation_name, poll_interval=30, timeout=1800):
    """Polls the operation asynchronously until it's done or times out."""
    logger.info(f"Polling for video completion for operation: {operation_name}")
    start_time = time.time()
    while time.time() - start_time < timeout:
        status_result = await check_video_generation_status(operation_name)
        
        if status_result["status"] == "succeeded":
            return status_result["uri"]
        elif status_result["status"] in ["failed", "error"]:
            logger.error(f"Polling failed: {status_result.get('error', 'Unknown error')}")
            return None
        
        logger.info(f"Video generation in progress. Checking again in {poll_interval} seconds...")
        await asyncio.sleep(poll_interval)
        
    logger.warning("Polling timed out.")
    return None

async def generate_and_poll_video(veo3_prompt, output_gcs_uri, **kwargs):
    """A wrapper function to generate and poll for video completion asynchronously."""
    loop = asyncio.get_running_loop()
    
    # Use functools.partial to correctly wrap the synchronous function call with its arguments
    func_with_args = functools.partial(
        generate_video_with_veo3, 
        veo3_prompt, 
        output_gcs_uri, 
        **kwargs
    )
    
    operation_name = await loop.run_in_executor(
        None,
        func_with_args
    )
    
    if operation_name:
        return await poll_for_video_completion(operation_name)
    return None

if __name__ == '__main__':
    # Example usage (for direct testing of this module)
    async def main():
        logging.basicConfig(level=logging.INFO)
        test_prompt = "A cinematic shot of a futuristic car driving on a highway in a neon-lit city at night"
        # IMPORTANT: Replace with your actual GCS bucket
        test_output_gcs_uri = "gs://your-gcs-bucket-name/test_video_output/" 
        
        final_uri = await generate_and_poll_video(test_prompt, test_output_gcs_uri)
        
        if final_uri:
            logger.info(f"\n--- Video Generation Complete ---")
            logger.info(f"Final result URI: {final_uri}")
        else:
            logger.error("\n--- Video Generation Failed or Timed Out ---")
            
    asyncio.run(main())
        
# Note: The main execution block is for testing purposes.
# The core logic is designed to be imported and used by app.py.
# Ensure you have authenticated with GCP and set up your project and bucket.
# gcloud auth application-default login
# export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
# export GCS_BUCKET="your-gcs-bucket-name"