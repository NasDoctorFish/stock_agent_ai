#!/bin/bash

#read -p "Which directory?: " directory
LOCAL_DIR="/Users/jju/Documents/CODING/Home_Sweet_Home_Server"
REMOTE="jwon@192.168.0.1:/home/jwon/Documents"

echo "입력한 폴더: $LOCAL_DIR"
echo "🔄 Sending folder..."

scp -r "$LOCAL_DIR" "$REMOTE"

echo "✅ Folder sent!"
