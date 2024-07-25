# Glaucoma_Detection_ML
Detecting Glaucoma with Machine Learning Techniques and CNN, and Visualizing Models in a Streamlit Web Application

## Getting started

# Why Virtual Environment 
First, we create a virtual environment. A virtual environment is an isolated Python environment that allows you to install packages and dependencies specific to a project without affecting the global Python installation. It helps manage project-specific dependencies efficiently.
We might need a virtual environment for each of our project as each project might need a particular version of package or library, and using the same environment for multiple projects may lead to some features missing out on specific projects.

# Creating Virtual Environment
-> Create a project folder
-> Open the command prompt in that directory
-> Run the following command:
    python -m venv env
-> After creating a virtual environment, run the following command to activate it:
    env\scripts\activate
-> Now, our virtual environment is active. After you complete using the environment and want to deactivate it, just run the "deactivate" prompt to do the job.

# Installing required packages
-> All the required packages have been specified in the requirement.txt file.
-> Copy the requirements.txt file into your project folder.
-> After activating the virtual environment, run the following command to install required packages:
    pip install -r requirement.txt
-> All required packages will be installed.
-> to get the requirements from an existing environment, run the following command there:
    pip freeze > requirements.txt


# Importing all python files
Copy all the files into your project folder

# Running the project
-> First, open the project folder using VS Code and activate the environment using the following command in the terminal:
    env\scripts\activate
-> Once the env is active, run the following command to get the streamlit app going:
    streamlit run app.py
-> To stop running the app, just click "ctrl+C" in the terminal.


# To get to our deployed streamlit app,
[Just click here...!](https://glaucomadetection.streamlit.app)
