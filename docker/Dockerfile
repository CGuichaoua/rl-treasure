# Base image with Python and GPU support (if needed)
FROM python:3.12-slim

# Set working directory
WORKDIR /app

COPY docker/requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install -r requirements.txt

# Copy project files
COPY treasure_hunt /app/treasure_hunt

# Expose a port (if needed for rendering or serving results)
EXPOSE 5000

# Default command for the container
ENTRYPOINT ["python", "-m", "treasure_hunt.main"]
CMD ["--environment", "base", "--agent", "oblivious", "--timesteps", "1000"]