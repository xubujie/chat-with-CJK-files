# chat-with-CJK-files
Chat with CJK(Chinese, Japanese, Korea) Docs.


# How to use 

1. clone the repo, and create an new python env, then `pip install -r requirements.txt`
1. `cp .env.example .env`, set up your OPENAI_API_KEY in .env 
2. Put your docs into docs/ directory
3. run `python ingest.py`
4. run `python app.py` and start chating


# Change Language 

If you want to chat with Japanese or Korean doc, please change the default prompt in app.py

