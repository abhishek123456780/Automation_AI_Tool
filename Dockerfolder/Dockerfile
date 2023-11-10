# Use the official Python image as the base image
FROM python:3.8

# Set the working directory within the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install -r requirements.txt
RUN pip install gunicorn
RUN pip install Flask
RUN pip install gdown

# Copy the models directory into the container
# Download the model from Google Drive using gdown
RUN set -eux \
    && gdown --id 1XgJhPr5aqGuWZQfoCOF57JKf_NCzucC9 -O models.zip \
    && unzip models.zip \
    && rm models.zip \
    && ls -la \
    && echo "Download and extraction successful"


# Copy the rest of the application files into the container
COPY . .

# Set the environment variable for the Python buffering
ENV PYTHONUNBUFFERED 1

# Run your Python script using Gunicorn
# CMD ["gunicorn", "-b", "0.0.0.0:5011", "fast_AI_API:app"]
CMD [ "python", "-m" , "app", "app", "--host=0.0.0.0:5011"]

