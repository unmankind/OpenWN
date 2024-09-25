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
        "Characters": [],
        "PlotProgression": {
            "mainGoal": "",
            "current_challenge": [],
            "subplots": [],
            "significantEvents": [],
            "challenges": []
        },
        "Locations": [],
        "ThemesAndTone": {
            "themes": [],
            "tone": "",
            "recurring_motifs": []
        },
        "chapters": [],
        "ObjectsAndArtifacts": [],
        "UnresolvedQuestionsAndConflicts": [],
        "conflicts": [],
        "dialogue_styles": {},
        "NotableQuotes": []
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
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(memory, f, ensure_ascii=False, indent=4)

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
    2. **PlotProgression**: Summarize the main goal of the chapter, any subplots, significant events, and challenges faced by the characters.
    3. **Locations**: Identify all significant locations in the chapter and describe important events tied to those locations.
    4. **ThemesAndTone**: Describe the recurring themes and tone presented in this chapter.
    5. **ObjectsAndArtifacts**: Note any significant objects or artifacts introduced or mentioned, including who possesses them.
    6. **UnresolvedQuestionsAndConflicts**: List any unresolved questions, mysteries, or conflicts that arise or persist in this chapter.
    7. **NotableQuotes**: Provide any important or memorable quotes from the characters.

    Provide the output strictly as valid JSON without any backticks or formatting markers.

    Chapter text:
    {chapter_text}
    """

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

        # Update characters (handling both "Major" and "Minor")
        if "Characters" in parsed_data:
            memory["Characters"].extend(parsed_data["Characters"])

        # Update plot progression (handle keys consistently)
        if "PlotProgression" in parsed_data:
            plot_data = parsed_data["PlotProgression"]
            memory["PlotProgression"]["main_goal"] = plot_data.get("main_goal", "")
            memory["PlotProgression"]["subplots"] = plot_data.get("subplots", [])
            memory["PlotProgression"]["significantEvents"].extend(plot_data.get("significantEvents", []))
            memory["PlotProgression"]["challenges"] = plot_data.get("challenges", [])

        # Update locations
        if "Locations" in parsed_data:
            memory["Locations"].extend(parsed_data["Locations"])

        # Update themes and tone
        if "ThemesAndTone" in parsed_data:
            memory["ThemesAndTone"]["themes"] = parsed_data["ThemesAndTone"].get("themes", [])
            memory["ThemesAndTone"]["tone"] = parsed_data["ThemesAndTone"].get("tone", "")

        # Update artifacts
        if "ObjectsAndArtifacts" in parsed_data:
            memory["ObjectsAndArtifacts"].extend(parsed_data["ObjectsAndArtifacts"])

        # Update unresolved questions and conflicts
        if "UnresolvedQuestionsAndConflicts" in parsed_data:
            memory["UnresolvedQuestionsAndConflicts"].extend(parsed_data["UnresolvedQuestionsAndConflicts"])

        # Update notable quotes
        if "NotableQuotes" in parsed_data:
            memory["NotableQuotes"].extend(parsed_data["NotableQuotes"])

        # Add the new chapter and summary to memory
        memory["chapters"].append({
            "chapter_text": chapter_text,
            "summary": parsed_data.get("PlotProgression", {}).get("significantEvents", "No summary available")
        })

        # Save the updated memory back to the file
        save_memory(memory)

        # Log memory after the update
        logging.debug(f"Updated Memory After Analysis: {json.dumps(memory, indent=2)}")

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

    # Optionally, extract just the summary from the analysis JSON
    try:
        parsed_data = json.loads(analysis)
        summary = parsed_data.get("plot_progression", {}).get("main_goal", "No summary available")
    except json.JSONDecodeError:
        summary = "No summary available"

    return summary



# Generate the story seed
def generate_seed():
    """
    Generate a seed for the story (initial prompt to kickstart the story).
    """
    system_prompt = "You are a storyteller."
    user_prompt = "Generate a captivating opening scene for a fantasy adventure story."
    story_seed = call_openai_api(system_prompt, user_prompt, max_tokens=1000)

    return story_seed

# Generate the next chapter
def generate_next_chapter(previous_chapters):
    """
    Generate the next chapter using OpenAI API based on previous chapters and memory.json data.
    """
    # Load memory.json to access stored characters, plot details, themes, etc.
    memory = load_memory()

    # Extract key elements from memory
    characters = memory.get("Characters", [])
    plot = memory.get("PlotProgression", {})
    locations = memory.get("Locations", [])
    theme = memory.get("ThemesAndTone", {})
    unresolved_questions = memory.get("UnresolvedQuestionsAndConflicts", [])
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

    # Adjust the chapter index (subtract 1) to match the internal memory list index
    chapter_index -= 1

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

    #Summarize the next_chapter_text
    summary = summarize_chapter(next_chapter_text)

    # Append the new chapter and summary to memory
    memory["chapters"].append({
        "chapter_text": next_chapter_text,
        "summary": summary
        })

    # Log the generated summary for debugging
    logging.debug(f"Summary: {summary}")

    # Calculate the new chapter index (add 1 for 1-based indexing)
    new_chapter_index = len(memory["chapters"])  # len gives you the total number of chapters


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

    # Generate the story seed (first chapter)
    story_seed = generate_seed()

    # Log the generated story seed for debugging
    logging.debug(f"Story Seed: {story_seed}")

    # Summarize and analyze the story seed (this will update memory.json automatically)
    summary = summarize_chapter(story_seed)

    # Log the generated summary for debugging
    logging.debug(f"Summary: {summary}")

    # Ensure that the chapter was added to memory correctly during summary analysis
    memory = load_memory()  # Reload the memory to confirm it's updated
    
    # Log the updated memory to confirm it's been updated correctly
    logging.debug(f"Updated Memory: {json.dumps(memory, indent=2)}")

    # Redirect to the first chapter (chapter index 1)
    return redirect(url_for('show_chapter', chapter_index=1))




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
