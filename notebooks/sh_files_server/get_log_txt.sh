#!/bin/bash

# 일단 log.txt에 올려진 기록을 자동으로 가져와주는 코드 굳이 파일 이름 X

# read -p "Which file or directory on server?: " target
REMOTE_PATH="jwon@192.168.0.16:/home/jwon/post_office/log.txt"
LOCAL_PATH="/Users/jju/Documents/Stock_ML"

echo "서버에서 가져올 대상: $REMOTE_PATH"
echo "저장될 로컬 경로: $LOCAL_PATH"
echo "🔄 Downloading..."

scp -r "$REMOTE_PATH" "$LOCAL_PATH"

echo "✅ Download complete!"
