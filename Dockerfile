# Use official Python image
FROM python:3.10

# Set work directory
WORKDIR /app

# Copy and Install Python dependencies
COPY . /app

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port (Flask)
EXPOSE 5000

# Command to start the app
CMD ["python", "app.py"]