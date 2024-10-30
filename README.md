# rag

## Deploy
Download Model
```sh
python src/inference.py
```

Deploy API Server
```sh
beam deploy src/api.py:yicoder_api --name yicoder-inference-server
```

Run Chatbot
```sh
python src/chat.py
```

