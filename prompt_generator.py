def generate_veo3_prompt_with_gemini(gemini_model, product_name, product_description, ad_brief):
    print("\n--- Generating Veo3 Prompt with Gemini 1.5 Flash ---")

    llm_prompt = f"""
    You are an expert video prompt engineer for Veo3, a cutting-edge text-to-video AI model.
    Your task is to create a highly detailed and visually rich prompt for Veo3 to generate an 8-second advertisement video.
    The video must be concise, impactful, and clearly convey the product's essence within the 8-second limit.
    Focus on dynamic visual animations, seamless transitions, and evocative imagery.

    **Product Name:** {product_name}
    **Product Description:** {product_description}
    **Advertisement Brief (User's request for tone/theme):** {ad_brief}

    ---
    **Guidelines for Veo3 Prompt:**
    1.  **Duration:** Strictly 8 seconds. Structure it into logical segments (e.g., Problem, Solution, Benefit, CTA).
    2.  **Visual Focus:** Prioritize visual storytelling. Use vivid, descriptive language for scenes, animations, lighting, and transitions. Think cinematic, high-fidelity, futuristic, organic, etc.
    3.  **Conciseness:** Every word matters. Be precise.
    4.  **No Explanations:** The output must ONLY be the Veo3 prompt text. Do not include any conversational filler, introductions, or explanations.
    5.  **Placeholders for A/B Testing (Optional, but good practice):** If applicable, suggest where a dynamic text overlay or specific benefit could be swapped out for A/B testing, like `[CALL_TO_ACTION_TEXT]`.
    6.  **Brand Integration:** Ensure the product name is clearly featured.
    7.  **Final Output Format:** Provide the raw prompt text suitable for a "text_to_video" API.

    ---
    **Example Structure (Adapt this heavily):**

    Generate a highly dynamic, visually captivating 8-second video advertisement for **[PRODUCT_NAME]**.

    **Objective:** [Brief statement of video's goal]

    **Overall Visual Tone:** [e.g., Futuristic, Intelligent, Energetic]

    **Scene 1 (0-2.5s): [Scene Title]**
    *   **Visual:** [Detailed description of visual elements, actions, environment]
    *   **Animation Style:** [Specific animation techniques, camera moves, VFX]
    *   **Lighting:** [Mood, color temperature]
    *   **Text Overlay (Optional):** "[TAGLINE_VARIABLE]"

    **Scene 2 (2.5-6s): [Scene Title]**
    *   **Visual:** [Detailed description]
    *   **Animation Style:** [Specific techniques]
    *   **Lighting:** [Mood]
    *   **Text Overlay (Optional):** "[BENEFIT_VARIABLE]"

    **Scene 3 (6-8s): [Scene Title]**
    *   **Visual:** [Detailed description, product logo reveal, final shot]
    *   **Animation Style:** [Specific techniques]
    *   **Lighting:** [Mood]
    *   **Text Overlay (Variable):** "[CALL_TO_ACTION_TEXT]"
    *   **Brand Element:** Integrate the "{product_name}" logo.

    **Visual Keywords for Veo3 (to push quality):** [e.g., Cinematic, macro, high-fidelity, dynamic lighting, subtle VFX, organic transitions, modern, elegant, data visualization.]
    **Aspect Ratio:** 1:1 (Square for social media)
    """

    try:
        # Use a higher temperature for more creative prompts, lower for more structured
        response = gemini_model.generate_content(llm_prompt, generation_config={"temperature": 0.7})
        veo3_prompt = response.candidates[0].content.parts[0].text.strip()
        print("Gemini generated prompt successfully!")
        return veo3_prompt
    except Exception as e:
        print(f"Error generating prompt with Gemini: {e}")
        return None