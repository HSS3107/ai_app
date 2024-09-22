FROM python:3.11-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    dos2unix \
    gnupg \
    libssl-dev \
    libexpat1-dev \
    gettext \
    wget \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /code/

# Copy and install Python dependencies
COPY requirements.txt /code/
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Install uWSGI
RUN pip install uwsgi

# Copy project files
COPY . /code/

# Convert line endings
RUN dos2unix /code/docker/prod/entrypoint.sh /code/manage.py

# Set permissions
RUN chmod 755 /code/docker/prod/entrypoint.sh

# Expose ports
EXPOSE 2222 80

# Entry point
ENTRYPOINT ["/code/docker/prod/entrypoint.sh"]

# Command to run
CMD ["uwsgi", "--ini", "/code/docker/prod/uwsgi.ini"]