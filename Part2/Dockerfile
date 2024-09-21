# Use the official Python image with version 3.11.7
FROM python:3.11.7

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy the requirements.txt file into the container
COPY requirements.txt .

# Step 4: Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the rest of the application code into the container
COPY . .

# Step 6: Ensure the OpenAI key is passed securely using Docker secrets
# For security, do not copy openai.key directly into the container.
# Instead, read it as a secret during runtime
RUN mkdir -p /app/secrets

# Step 7: Expose the port that Streamlit will run on
EXPOSE 8501

# Step 8: Command to run your Streamlit app (pass the OpenAI key as an environment variable)
CMD ["streamlit", "run", "app.py"]


