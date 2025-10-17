# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 8010

# Define environment variable
ENV PORT 8010

# Run uvicorn when the container launches
# Use 0.0.0.0 to make it accessible from outside the container
CMD ["uvicorn", "orchestrator_main:app", "--host", "0.0.0.0", "--port", "8010"]