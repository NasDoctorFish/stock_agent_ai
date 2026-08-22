#!/bin/bash

# Fill in with: tailscale status (on either device)
TAILSCALE_IP="100.79.94.117"
REMOTE_USER="jwon"

echo "🔄 Connecting to $REMOTE_USER@$TAILSCALE_IP via Tailscale SSH..."

ssh "$REMOTE_USER@$TAILSCALE_IP"
