import os
import time
import asyncio
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
GCP_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
GCS_BUCKET = os.getenv("GCS_BUCKET")

# Ensure all necessary environment variables are set
if not all([GCP_PROJECT_ID, GCS_BUCKET]):
    raise ValueError("Missing one or more environment variables (GCP_PROJECT_ID, GCS_BUCKET). Please check your .env file.")

# Initialize Vertex AI
try:
    from google.cloud import aiplatform
    aiplatform.init(project=GCP_PROJECT_ID, location="us-central1")
    logger.info("Vertex AI initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Vertex AI: {e}")
    raise RuntimeError("Failed to initialize Vertex AI. Please ensure you have authenticated to GCP and enabled the Vertex AI API.")

from prompt_generator import generate_veo3_prompt_with_gemini
from video_generator import generate_and_poll_video

# --- FastAPI App ---
app = FastAPI(
    title="Product Advertisement Video Generator API",
    description="Generate AI-powered product advertisement videos using Google's Veo3 and Gemini",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class ProductVideoRequest(BaseModel):
    product_name: str = Field(..., min_length=1, max_length=100, description="Name of the product")
    product_description: str = Field(..., min_length=10, max_length=1000, description="Detailed description of the product and its benefits")
    ad_brief: str = Field(..., min_length=10, max_length=500, description="Description of what to include in the 8-second ad")
    duration_seconds: Optional[int] = Field(8, ge=5, le=8, description="Video duration in seconds (5-8)")
    aspect_ratio: Optional[str] = Field("16:9", description="Video aspect ratio: 16:9 or 9:16")
    sample_count: Optional[int] = Field(1, ge=1, le=4, description="Number of video samples to generate")
    
    class Config:
        schema_extra = {
            "example": {
                "product_name": "EcoGlow Smart Garden",
                "product_description": "An innovative indoor smart garden system that uses AI to optimize plant growth with automated watering, LED lighting, and nutrient management. Perfect for urban dwellers who want fresh herbs and vegetables year-round.",
                "ad_brief": "Futuristic, sleek design showcasing the problem-solution format. Highlight efficiency, sustainability, and the convenience of fresh produce at home.",
                "duration_seconds": 8,
                "aspect_ratio": "16:9",
                "sample_count": 1
            }
        }

class GeneratedVideoSample(BaseModel):
    uri: str
    encoding: str

class VideoGenerationResponse(BaseModel):
    message: str
    video_id: Optional[str] = None
    operation_name: Optional[str] = None
    generated_prompt: Optional[str] = None

class VideoStatusResponse(BaseModel):
    video_id: str
    status: str
    operation_name: Optional[str] = None
    generated_samples: Optional[list[GeneratedVideoSample]] = None
    error_message: Optional[str] = None
    progress_percentage: Optional[int] = None

# --- In-memory storage for video generation status ---
# In production, use a proper database or cache like Redis
video_status_store = {}

# --- Helper Functions ---
async def generate_video_async(video_id: str, product_name: str, product_description: str, ad_brief: str, duration_seconds: int, aspect_ratio: str, sample_count: int):
    """Async function to handle video generation in the background"""
    try:
        # Update status to processing
        video_status_store[video_id] = {
            "status": "processing",
            "operation_name": None,
            "generated_samples": None,
            "error_message": None,
            "progress_percentage": 10
        }
        
        logger.info(f"Starting video generation for {video_id}")
        
        # Step 1: Generate Veo3 Prompt
        veo3_prompt = generate_veo3_prompt_with_gemini(product_name, product_description, ad_brief)
        
        if not veo3_prompt:
            raise Exception("Failed to generate a valid prompt")
        
        logger.info(f"Generated prompt for {video_id}: {veo3_prompt[:100]}...")
        video_status_store[video_id]["progress_percentage"] = 30
        
        # Step 2: Generate Video with Veo3 and poll for completion
        video_filename_prefix = f"{product_name.replace(' ', '_').lower()}_{int(time.time())}"
        output_gcs_uri = f"gs://{GCS_BUCKET}/{video_filename_prefix}/"
        
        # This function is now fully async and uses the corrected video_generator
        final_result_uri = await generate_and_poll_video(
            veo3_prompt,
            output_gcs_uri,
            duration_seconds=duration_seconds,
            aspect_ratio=aspect_ratio,
            sample_count=sample_count
        )
        
        if final_result_uri:
            video_status_store[video_id].update({
                "status": "completed",
                "generated_samples": [{"uri": final_result_uri, "encoding": "application/json"}],
                "error_message": None,
                "progress_percentage": 100
            })
            logger.info(f"Video generation completed for {video_id}. Result manifest: {final_result_uri}")
        else:
            raise Exception("Video generation failed or timed out.")
            
    except Exception as e:
        logger.error(f"Video generation failed for {video_id}: {str(e)}", exc_info=True)
        video_status_store[video_id].update({
            "status": "failed",
            "error_message": str(e),
            "progress_percentage": 0
        })

# --- API Endpoints ---
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Product Advertisement Video Generator API is running",
        "version": "1.0.0",
        "endpoints": {
            "generate_video": "/generate-video",
            "video_status": "/video-status/{video_id}",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/generate-video", response_model=VideoGenerationResponse)
async def generate_product_video(
    request: ProductVideoRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate a product advertisement video
    
    This endpoint starts the video generation process in the background and returns immediately
    with a video_id that can be used to check the status of the generation.
    
    The process follows Vertex AI's long-running operation pattern:
    1. Submit request to Vertex AI Veo
    2. Receive operation name
    3. Poll operation status until completion
    4. Return generated video URIs
    """
    try:
        # Generate unique video ID
        video_id = f"{request.product_name.replace(' ', '_').lower()}_{int(time.time())}"
        
        # Initialize status
        video_status_store[video_id] = {
            "status": "queued",
            "operation_name": None,
            "generated_samples": None,
            "error_message": None,
            "progress_percentage": 0
        }
        
        # Start background task for video generation
        background_tasks.add_task(
            generate_video_async,
            video_id,
            request.product_name,
            request.product_description,
            request.ad_brief,
            request.duration_seconds,
            request.aspect_ratio,
            request.sample_count
        )
        
        logger.info(f"Video generation queued for {video_id}")
        
        return VideoGenerationResponse(
            status="queued",
            message="Video generation started. Use the video_id to check status.",
            video_id=video_id
        )
        
    except Exception as e:
        logger.error(f"Error starting video generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start video generation: {str(e)}")

@app.get("/video-status/{video_id}", response_model=VideoStatusResponse)
async def get_video_status(video_id: str):
    """
    Get the status of a video generation request
    
    Returns the current status of the video generation process:
    - queued: Generation is queued
    - processing: Generation is in progress
    - completed: Generation is complete, generated_samples available
    - failed: Generation failed, error_message available
    
    For completed requests, generated_samples contains an array of video objects
    with uri (GCS URI) and encoding (video format) fields matching Vertex AI response structure.
    """
    if video_id not in video_status_store:
        raise HTTPException(status_code=404, detail="Video ID not found")
    
    status_info = video_status_store[video_id]
    
    # Convert generated_samples back to Pydantic models if they exist
    generated_samples = None
    if status_info.get("generated_samples"):
        generated_samples = [
            GeneratedVideoSample(**sample) 
            for sample in status_info["generated_samples"]
        ]
    
    return VideoStatusResponse(
        video_id=video_id,
        status=status_info["status"],
        operation_name=status_info.get("operation_name"),
        generated_samples=generated_samples,
        error_message=status_info.get("error_message"),
        progress_percentage=status_info.get("progress_percentage", 0)
    )

@app.delete("/video-status/{video_id}")
async def delete_video_status(video_id: str):
    """
    Delete video status from memory (cleanup)
    """
    if video_id not in video_status_store:
        raise HTTPException(status_code=404, detail="Video ID not found")
    
    del video_status_store[video_id]
    return {"message": f"Video status for {video_id} deleted successfully"}

@app.get("/video-status")
async def list_all_video_status():
    """
    List all video generation requests (for debugging/monitoring)
    """
    return {
        "total_requests": len(video_status_store),
        "requests": video_status_store
    }

# --- Error Handlers ---
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return {"error": "Invalid input", "detail": str(exc)}

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return {"error": "Internal server error", "detail": "An unexpected error occurred"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, log_level="info")