# OpenWN - AI Web Novel Generator

OpenWN is an AI-powered web application that generates a fantasy adventure novel one chapter at a time. Users can generate the next chapter of the story based on a generated story skeleton, with support for dynamic plot progression, characters, and themes.

## Features
- Generate an initial story skeleton (setting, earth-shattering event, protagonist details).
- Create subsequent chapters that build upon the established story.
- Manage chapters, plot progression, and characters.
- Easily extendable and customizable.

## Getting Started

### Cloning the Repository

To get started, clone this repository to your local machine:

```bash
git clone https://github.com/unmankind/OpenWN.git
cd OpenWN

Setting Up Your Virtual Environment
You should set up a Python virtual environment to isolate the project's dependencies. This can be done using the following commands:

# Create the virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate

Installing the Requirements
Once your virtual environment is activated, install the required packages:
pip install -r requirements.txt

Adding Your OpenAI API Key
The app uses OpenAI's API to generate story content. You'll need to add your OpenAI API key to a .env file.

Create a .env file in the root of the project.
touch .env
nano .env
Add your OpenAI API key like so:
OPENAI_API_KEY=your_openai_api_key

Make sure to replace your_openai_api_key with your actual API key.

Running the Flask Application
With everything set up, you can now start the Flask application:

flask run


By default, Flask will run the app at http://127.0.0.1:5000/. Open this URL in your browser to view the app.

Additional Information
To clear memory and reset the story, use the "Clear Memory" button provided in the UI.
To generate the next chapter, use the navigation buttons at the bottom of each chapter page.
License
This project is licensed under the MIT License. See the LICENSE file for more details.

