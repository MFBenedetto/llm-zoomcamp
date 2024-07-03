# Start from the base image ollama/ollama
FROM ollama/ollama

# Copy files from the local directory ollama_files to /root/.ollama in the container
COPY ./ollama_files /root/.ollama

# Optional: Expose ports if your application listens on a specific port
EXPOSE 11434

# Optional: Define the command to run the application
# CMD ["python", "/root/.ollama/app.py"]