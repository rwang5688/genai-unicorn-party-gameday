FROM python:3.11

WORKDIR /home

# Install deps
ADD requirements.txt /home/
RUN pip3 install -r requirements.txt

# Add files
ADD bedrock_app_st.py /home/

# Expose port
EXPOSE 80

# Start App
CMD [ "streamlit","run" ,"bedrock_app_st.py","--server.port","80"]