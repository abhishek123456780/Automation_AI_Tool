# Use the official Python image as the base image
FROM python:3.8

# Set the working directory within the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .


# Install the required packages
RUN pip install -r requirements.txt
RUN pip install Flask
RUN pip install gdown  # Install gdown for downloading from Google Drive

# Download the model from Google Drive
# Replace '1XgJhPr5aqGuWZQfoCOF57JKf_NCzucC9' with the actual file ID
RUN gdown --id 1XgJhPr5aqGuWZQfoCOF57JKf_NCzucC9 -O model.zip \
    && unzip model.zip \
    && rm model.zip
    
# Copy the models directory into the container
COPY models /app/models

# Copy the rest of the application files into the container
COPY . .

# Set the environment variable for the Python buffering
ENV PYTHONUNBUFFERED 1


CMD [ "python", "-m" , "app", "app", "--host=0.0.0.0:5011"]

