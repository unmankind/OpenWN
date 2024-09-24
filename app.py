from openai import OpenAI
import os
import json
from flask import Flask, render_template, request, redirect, url_for
from dotenv import load_dotenv
import logging
import re


# Load environment variables from .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Flask app
app = Flask(__name__)

@app.template_filter('nl2br')
def nl2br(text):
    """Converts newlines to <br> tags."""
    return text.replace('\n', '<br>\n')

# Path to memory.json
MEMORY_FILE = "memory.json"

logging.basicConfig(level=logging.DEBUG)

def initialize_memory():
    """Ensure that memory.json exists and has the correct structure."""
    initial_memory = {
        "characters": [],
        "plot": {
            "main_goal": "",
            "current_challenge": "",
            "subplots": [],
            "past_events": [],
            "upcoming_events": []
        },
        "locations": [],
        "theme": {
            "theme": "",
            "tone": "",
            "recurring_motifs": []
        },
        "chapters": [],
        "artifacts": [],
        "unresolved_questions": [],
        "conflicts": [],
        "dialogue_styles": {},
        "notable_quotes": []
    }
    with open(MEMORY_FILE, 'w') as f:
        json.dump(initial_memory, f, indent=4)

# Load the memory.json data
def load_memory():
    """Load the memory.json file into a Python dictionary. Initialize it if empty or invalid."""
    try:
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, 'r') as f:
                content = f.read().strip()
                if content:
                    return json.loads(content)
                else:
                    # If the file is empty, initialize it
                    initialize_memory()
                    return load_memory()
        else:
            initialize_memory()
            return load_memory()
    except json.JSONDecodeError:
        # If the file contains invalid JSON, reinitialize it
        initialize_memory()
        return load_memory()

# Save updated memory back to memory.json
def save_memory(memory):
    """Save the updated memory back into memory.json."""
    with open(MEMORY_FILE, 'w') as f:
        json.dump(memory, f, indent=4)

# Standardized OpenAI API call function
def call_openai_api(system_content, user_content, model="gpt-4o-mini", max_tokens=10000):
    """
    Standard OpenAI API call format for generating content.
    """
    response = client.chat.completions.create(model=model,
    messages=[
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ],
    max_tokens=max_tokens)
    return response.choices[0].message.content.strip()

# Generate the analysis prompt
def generate_analysis_prompt(chapter_text):
    """
    Create a prompt to analyze the chapter and provide relevant details in JSON format.
    """
    if not chapter_text or chapter_text.strip() == "":
        logging.error("Chapter text is missing or empty.")
        return "Error: No chapter text provided."
           
    return f"""
    Analyze the following chapter and provide the following details in JSON format:

    1. **Characters**: List all major and minor characters that appear in this chapter. Include their traits, motivations, last known location, and their most recent actions.
    2. **Plot Progression**: Summarize the main goal of the chapter, any subplots, significant events, and challenges faced by the characters.
    3. **Locations**: Identify all significant locations in the chapter and describe important events tied to those locations.
    4. **Themes and Tone**: Describe the recurring themes and tone presented in this chapter.
    5. **Objects and Artifacts**: Note any significant objects or artifacts introduced or mentioned, including who possesses them.
    6. **Unresolved Questions and Conflicts**: List any unresolved questions, mysteries, or conflicts that arise or persist in this chapter.
    7. **Notable Quotes**: Provide any important or memorable quotes from the characters.

    Provide the output strictly as valid JSON without any backticks or formatting markers.

    Chapter text:
    {chapter_text}
    """

# Parse the response and update memory.json with relevant sections
def update_memory_with_analysis(analysis, chapter_text):
    """
    Parse the OpenAI API response and update the memory.json with the analysis.
    """
    memory = load_memory()

    try:
        # Log the analysis before parsing it
        logging.debug(f"Received analysis: {analysis}")

        # Clean up the analysis by removing triple backticks and any 'json' markers
        cleaned_analysis = re.sub(r"```(json)?", "", analysis).strip()

        # Log the cleaned analysis for debugging
        logging.debug(f"Cleaned analysis: {cleaned_analysis}")

        # Parse the analysis as JSON (ensure it's valid)
        parsed_data = json.loads(cleaned_analysis)

        # Update characters
        if "Characters" in parsed_data:
            for character in parsed_data["Characters"]:
                memory["characters"].append(character)

        # Update plot progression
        # Update plot progression
        if "Plot_Progression" in parsed_data:
            plot_data = parsed_data["Plot_Progression"]
            memory["plot"]["main_goal"] = plot_data.get("main_goal", "")
            memory["plot"]["subplots"] = plot_data.get("subplots", [])
            memory["plot"]["past_events"].extend(plot_data.get("significant_events", []))
            memory["plot"]["current_challenge"] = plot_data.get("challenges", "")


        # Update locations
        if "Locations" in parsed_data:
            memory["locations"].extend(parsed_data["Locations"])

        # Update themes
        if "Themes_and_Tone" in parsed_data:
            memory["theme"]["theme"] = parsed_data["Themes_and_Tone"].get("recurring_themes", [])
            memory["theme"]["tone"] = parsed_data["Themes_and_Tone"].get("tone", "")

        # Update artifacts
        if "Objects_and_Artifacts" in parsed_data:
            memory["artifacts"].extend(parsed_data["Objects_and_Artifacts"])

        # Update unresolved questions
        if "Unresolved_Questions_and_Conflicts" in parsed_data:
            memory["unresolved_questions"].extend(parsed_data["Unresolved_Questions_and_Conflicts"])

        # Update notable quotes
        if "Notable_Quotes" in parsed_data:
            memory["notable_quotes"].extend(parsed_data["Notable_Quotes"])

        # Add the new chapter and summary to memory
        memory["chapters"].append({
            "chapter_text": chapter_text,
            "summary": analysis
        })

        # Save the updated memory back to the file
        save_memory(memory)

    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {e}")
        logging.error(f"OpenAI analysis was not in valid JSON format: {analysis}")


# Summarize the chapter
def summarize_chapter(chapter_text):
    """
    Summarizes the chapter and updates memory.json with relevant details.
    """
    # Check if the chapter_text is empty or invalid
    if not chapter_text or chapter_text.strip() == "":
        logging.error("Chapter text is missing or empty.")
        return "Error: No chapter text provided."

    # Log the chapter text for debugging
    logging.debug(f"Chapter text passed for analysis: {chapter_text}")

    system_prompt = "You are a literary assistant."
    user_prompt = generate_analysis_prompt(chapter_text)

    # Call OpenAI API to get chapter analysis
    analysis = call_openai_api(system_prompt, user_prompt, max_tokens=5000)

    # Check if the analysis is empty or invalid
    if not analysis or "It seems you've provided" in analysis:
        logging.error("Received invalid analysis from OpenAI API.")
        return "Error: Invalid analysis received."

    # Update memory.json with the analysis
    update_memory_with_analysis(analysis, chapter_text)

    return analysis


# Generate the story seed
def generate_seed():
    """
    Generate a seed for the story (initial prompt to kickstart the story).
    """
    system_prompt = "You are a storyteller."
    user_prompt = "Generate a captivating opening scene for a fantasy adventure story."
    story_seed = call_openai_api(system_prompt, user_prompt, max_tokens=1000)

    # Save the seed to memory as the first chapter
    memory = load_memory()
    memory["chapters"].append({
        "chapter_text": story_seed,
        "summary": "Story seed generated as the opening scene."
    })
    save_memory(memory)
    return story_seed

#generate the next chapter
def generate_next_chapter(previous_chapters):
    """
    Generate the next chapter using OpenAI API based on previous chapters and memory.json data.
    """
    # Load memory.json to access stored characters, plot details, themes, etc.
    memory = load_memory()

    # Extract key elements from memory
    characters = memory.get("characters", [])
    plot = memory.get("plot", {})
    locations = memory.get("locations", [])
    theme = memory.get("theme", {})
    unresolved_questions = memory.get("unresolved_questions", [])
    conflicts = memory.get("conflicts", [])

    # Convert lists and dictionaries to strings for the prompt
    characters_str = json.dumps(characters, indent=2)
    plot_str = json.dumps(plot, indent=2)
    locations_str = json.dumps(locations, indent=2)
    theme_str = json.dumps(theme, indent=2)
    unresolved_questions_str = json.dumps(unresolved_questions, indent=2)
    conflicts_str = json.dumps(conflicts, indent=2)

    # Prepare the system prompt and user prompt
    system_prompt = "You are a storyteller."
    
    user_prompt = f"""
    Here are the previous chapters: {previous_chapters}

    Below is the information that has been developed so far in the story:
    - Characters: {characters_str}
    - Plot: {plot_str}
    - Locations: {locations_str}
    - Themes: {theme_str}
    - Unresolved Questions: {unresolved_questions_str}
    - Conflicts: {conflicts_str}

    Generate the next chapter of the story. Ensure the following:
    1. The chapter should conclude naturally, without cutting off mid-sentence.
    2. If a subplot or conflict is introduced in this chapter, it should either be resolved or set up a clear transition for the next chapter.
    3. The chapter should end with a sense of closure or suspense, setting up the next chapter if necessary.
    4. Ensure that the chapter is of appropriate length, and aim to complete it at a natural narrative break, such as the end of a scene or important dialogue.    """
    
    # Call OpenAI API to generate the next chapter based on the memory and previous chapters
    next_chapter_text = call_openai_api(system_prompt, user_prompt, max_tokens=2000)

    return next_chapter_text


# Flask routes
@app.route('/')
def index():
    """Show the list of chapters."""
    memory = load_memory()
    chapters = memory.get("chapters", [])
    return render_template('index.html', chapters=chapters)

@app.route('/chapter/<int:chapter_index>')
def show_chapter(chapter_index):
    """Show a specific chapter based on the chapter index."""
    memory = load_memory()
    chapters = memory.get("chapters", [])

    # Ensure the chapter_index is valid
    if chapter_index < 0 or chapter_index >= len(chapters):
        return redirect(url_for('index'))

    # Get the current chapter
    chapter = chapters[chapter_index]

    # Get next and previous chapter indices
    next_chapter_index = chapter_index + 1 if chapter_index < len(chapters) - 1 else None
    prev_chapter_index = chapter_index - 1 if chapter_index > 0 else None

    # Check if the current chapter is the latest one
    is_latest_chapter = chapter_index == len(chapters) - 1

    return render_template('chapter.html', 
                           chapter=chapter, 
                           chapter_index=chapter_index, 
                           next_chapter_index=next_chapter_index, 
                           prev_chapter_index=prev_chapter_index,
                           total_chapters=len(chapters),
                           is_latest_chapter=is_latest_chapter)

@app.route('/next-chapter', methods=['POST'])
def next_chapter():
    """Generate the next chapter based on previous chapters."""
    # Retrieve the previous chapters from memory
    memory = load_memory()
    previous_chapters = "\n".join([chapter["chapter_text"] for chapter in memory["chapters"]])

    # Generate the next chapter text
    next_chapter_text = generate_next_chapter(previous_chapters)

    # Check if this chapter is already present in the memory (to avoid duplication)
    if not any(chapter["chapter_text"] == next_chapter_text for chapter in memory["chapters"]):
        # Summarize the new chapter and update memory.json (summary and other analysis are handled here)
        summarize_chapter(next_chapter_text)

    memory = load_memory()

    # Redirect to the newly created chapter
    new_chapter_index = len(memory["chapters"]) - 1

    # Log the new chapter index for debugging
    logging.debug(f"Redirecting to new chapter index: {new_chapter_index}")

    return redirect(url_for('show_chapter', chapter_index=new_chapter_index))


@app.route('/generate-seed', methods=['POST'])
def generate_story_seed():
    """
    Generate the story seed (opening chapter) for the novel.
    """
    # Load the memory from memory.json
    memory = load_memory()

    story_seed = generate_seed()

    summary = summarize_chapter(story_seed)

    if len(memory["chapters"]) == 0:
        memory["chapters"].append({
            "chapter_text": story_seed,
            "summary": summary
        })

    save_memory(memory)

    # Redirect to the first chapter (chapter index 0)
    return redirect(url_for('show_chapter', chapter_index=0))

@app.route('/clear-memory', methods=['POST'])
def clear_memory():
    """Clear the memory.json file and reset it."""
    initialize_memory()
    return render_template('index.html', previous_chapters="", next_chapter="")

if __name__ == '__main__':
    # Initialize the memory.json file with the correct structure
    initialize_memory()

    # Run the Flask app
    app.run(debug=True)