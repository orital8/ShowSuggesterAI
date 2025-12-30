import os
from google import genai
from PIL import Image

# Initialize Client
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

def generate_fictional_show(source_titles, inspiration_type):
    """
    Generates a title, description, and image poster for a new show.
    """
    print(f"\n--- Creative Studio: Designing a new show based on {inspiration_type} ---")
    
    # 1. Generate Title and Description
    prompt = (
        f"Create a catchy title and a 2-sentence description for a new TV show "
        f"that combines the style and themes of these shows: {', '.join(source_titles)}. "
        f"Format strictly as:\nTitle: <title>\nDescription: <description>"
    )
    
    print("Asking Gemini for a concept...")
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview", 
            contents=prompt
        )
        
        text_output = response.text.strip()
        
        lines = text_output.split('\n')
        show_title = "Unknown Title"
        show_desc = "No description available."
        
        for line in lines:
            if line.startswith("Title:"):
                show_title = line.replace("Title:", "").strip()
            elif line.startswith("Description:"):
                show_desc = line.replace("Description:", "").strip()
                
        print(f"Concept Created: {show_title}")
        print(f"Plot: {show_desc}")
        
    except Exception as e:
        print(f"Error generating text: {e}")
        raise e

    # 2. Generate Image Poster
    image_prompt = (
        f"A high-quality movie poster for a TV show named '{show_title}'. "
        f"The show is about: {show_desc}. Cinematic lighting, professional design."
    )
    
    print("Painting the poster (this may take a moment)...")
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=image_prompt
        )
        
        if response.parts:
            for part in response.parts:
                if part.inline_data:
                    image = part.as_image()
                    print(f"Displaying poster for '{show_title}'...")
                    image.show()
        else:
            print("No image parts found in the response.")
                
    except Exception as e:
        print(f"Error generating image: {e}")
        raise e

    return show_title, show_desc