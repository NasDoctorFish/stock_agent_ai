#!/bin/bash

read -p "Which directory?: " directory
LOCAL_DIR="/Users/jju/Documents/Stock_ML/$directory"
REMOTE="jwon@192.168.0.16:/home/jwon/post_office"

echo "입력한 폴더: $LOCAL_DIR"
echo "🔄 Sending folder..."

scp -r "$LOCAL_DIR" "$REMOTE"

echo "✅ Folder sent!"
