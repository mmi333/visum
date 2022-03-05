Quick project for a hackathon. 

Give it a paragraph from a blog post and it will (try) to make you a (weird but interesting) 256x256 image fitting the paragraph. 


1. Clone the repo
2. pip install -r requirements.txt
3. python3 server.py
4. Give it a paragraph and wait a minute (GPU) or 20+ minutes (CPU) 

Note: During the first run it will take more time because it will be downloading the models for the first time


The front-end is designed with minimalism in mind, which is another way of saying it's empty because I can't do front-end.

The back-end is simply Flask.

snrspeaks/t5-one-line-summary from huggingface is used for summarizing the paragraph into a line. 

glide_text2im is used to convert the line into an image.

Another Note: It may very well not work on your machine because of one of many little details (CUDA version, etc.)

while I admit it should at least be dockerized, I don't know how to do that yet so to run it you may have to do some hacks.

