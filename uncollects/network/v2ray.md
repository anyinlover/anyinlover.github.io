# V2Ray

## What

## Why

## How

```json
{
  "log": {
    "loglevel": "warning"
  },
  "inbounds": [
    {
      "port": 443,
      "protocol": "trojan",
      "settings": {
        "clients": [
          {
            "password": "trojan-gu"
          }
        ]
      },
      "streamSettings": {
        "network": "tcp",
        "security": "tls",
        "tlsSettings": {
          "certificates": [
            {
              "certificateFile": "/etc/letsencrypt/live/seetheworld.run.place/fullchain.pem",
              "keyFile": "/etc/letsencrypt/live/seetheworld.run.place/privkey.pem"
            }
          ]
        }
      }
    }
  ],
  "outbounds": [
    {
      "protocol": "freedom",
      "settings": {}
    }
  ]
}
```

```json
{
  "log": {
    "loglevel": "warning",
    "access": "/home/linuxbrew/.linuxbrew/var/log/v2ray/access.log",
    "error": "/home/linuxbrew/.linuxbrew/var/log/v2ray/error.log"
  },
  "inbounds": [
    {
      "port": 1080,
      "protocol": "socks",
      "settings": {
        "auth": "noauth"
      }
    }
  ],
  "outbounds": [
    {
      "protocol": "freedom",
      "tag": "direct"
    },
    {
      "protocol": "blackhole",
      "tag": "blocked"
    },
    {
      "protocol": "trojan",
      "tag": "proxy",
      "settings": {
        "servers": [
          {
            "address": "seetheworld.run.place",
            "port": 443,
            "password": "trojan-gu"
          }
        ]
      },
      "streamSettings": {
        "network": "tcp",
        "security": "tls",
        "tlsSettings": {
          "allowInsecure": false,
          "serverName": "seetheworld.run.place"
        }
      }
    }
  ],
  "dns": {
    "servers": [
      {
        "address": "223.5.5.5",    // Alibaba DNS (China)
        "port": 53,
        "domains": [
          "geosite:cn"            // Use for Chinese domains
        ]
      },
      {
        "address": "8.8.8.8",      // Google Public DNS (Global)
        "port": 53,
        "domains": [
          "geosite:geolocation-!cn" // Use for non-Chinese domains
        ]
      },
      "1.1.1.1",                   // Fallback DNS for international domains
      "localhost"                  // Use system resolver as a last resort
    ]
  },
  "routing": {
    "domainStrategy": "IPOnDemand",
    "rules": [
      {
        "type": "field",
        "outboundTag": "direct",
        "domain": [
          "geosite:cn"
        ]
      },
      {
        "type": "field",
        "outboundTag": "direct",
        "ip": [
          "geoip:cn"
        ]
      },
      {
        "type": "field",
        "outboundTag": "blocked",
        "domain": [
          "geosite:category-ads"
        ]
      },
      {
        "type": "field",
        "outboundTag": "proxy",
        "domain": [
          "geosite:geolocation-!cn"
        ]
      },
      {
        "type": "field",
        "outboundTag": "proxy",
        "ip": [
          "geoip:!cn"
        ]
      }
    ]
  }
}

```
