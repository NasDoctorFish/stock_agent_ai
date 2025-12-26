#!/bin/bash

read -p "Which file?: " filename
LOCAL_FILE="/Users/jju/Documents/Stock_ML/$filename"
REMOTE="jwon@192.168.0.16:/home/jwon/post_office"

echo "입력한 파일: $LOCAL_FILE"
echo "🔄 Sending file..."

scp "$LOCAL_FILE" "$REMOTE"

echo "✅ File sent!"