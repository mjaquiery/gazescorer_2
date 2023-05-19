FROM python:3.9

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN mkdir -p /app
WORKDIR /app
COPY . .
#RUN conda init bash && chmod +x /root/.bashrc && /root/.bashrc
#RUN conda env create -f Example/BRM_GS.yml
#CMD conda activate BRM_GS
#
RUN apt-get update && \
		apt-get install -y \
			ffmpeg libsm6 libxext6 && \
		apt-get autoremove && \
		apt-get autoclean

RUN pip install pandas matplotlib scipy imutils plotnine scikit-learn keras tensorflow
RUN pip install ffmpeg
RUN pip install ffmpeg-python opencv-python
RUN pip install cmake
RUN pip install dlib
CMD python
