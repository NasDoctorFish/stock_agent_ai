#!/bin/bash
# Simple Wi-Fi connect script for Debian (wpa_supplicant + dhclient)

IFACE="wlan0"          # Wi-Fi interface name
SSID="YOUR_WIFI_NAME"  # Wi-Fi SSID
PASSWORD="YOUR_WIFI_PASSWORD"

CONF="/etc/wpa_supplicant/wpa_supplicant-${IFACE}.conf"

echo "[1/5] Checking interface..."
ip link show $IFACE > /dev/null 2>&1 || {
  echo "❌ Interface $IFACE not found"
  exit 1
}

echo "[2/5] Creating wpa_supplicant config..."
sudo bash -c "cat > $CONF <<EOF
ctrl_interface=DIR=/run/wpa_supplicant GROUP=netdev
update_config=1
country=SG

network={
    ssid=\"${SSID}\"
    psk=\"${PASSWORD}\"
}
EOF"

chmod 600 $CONF

echo "[3/5] Bringing interface up..."
sudo ip link set $IFACE up

echo "[4/5] Starting wpa_supplicant..."
sudo pkill wpa_supplicant
sudo wpa_supplicant -B -i $IFACE -c $CONF

sleep 5

echo "[5/5] Getting IP via DHCP..."
sudo dhclient $IFACE

echo "✅ Done. Current IP:"
ip a show $IFACE

