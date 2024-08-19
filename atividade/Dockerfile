# Use an official Python image as the base
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app/

#Copiar o modelo para dentro do container
COPY modelo_clf.sav /app/modelo_clf.sav

# Expose the port
EXPOSE 3000

# Run the command to start the development server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000"]