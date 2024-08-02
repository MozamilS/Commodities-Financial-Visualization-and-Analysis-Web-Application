# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the application code into the container at /app
COPY . /app

# Copy the .env file into the container at /app
COPY .env /app/.env

# Install the required Python packages
#RUN pip install pytz base64 requests streamlit_autorefresh fredapi scipy datetime pandas_market_calendars streamlit boto3 st-files-connection "s3fs>=0.5.2" seaborn plotly_express pandas numpy matplotlib python-dotenv statsmodels


RUN pip install pytz requests
RUN pip install streamlit_autorefresh fredapi scipy
RUN pip install datetime pandas_market_calendars streamlit
RUN pip install boto3 st-files-connection "s3fs>=0.5.2"
RUN pip install seaborn plotly_express pandas numpy
RUN pip install matplotlib python-dotenv statsmodels kaleido

# Expose port 8501 to the outside world
EXPOSE 8501

# Specify the command to run the Streamlit application
ENTRYPOINT ["streamlit", "run"]
CMD ["streamlit_app.py"]