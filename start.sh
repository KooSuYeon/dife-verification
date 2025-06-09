#!/bin/bash

echo -e "$AI_TLS_CRT" > /tmp/server.crt
echo -e "$AI_TLS_KEY" > /tmp/server.key

uvicorn verify:app --host 0.0.0.0 --port 8000 \
    --ssl-certfile /tmp/server.crt \
    --ssl-keyfile /tmp/server.key
