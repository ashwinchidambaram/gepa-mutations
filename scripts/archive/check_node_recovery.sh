#!/bin/bash
# Check if downed GPU cluster nodes (bourbaki, ansatz, deepseek) have recovered.
# Sends a Telegram alert if any node is back up. Silent if all still down.
# Designed to run unattended from crontab every 15 minutes.

TELEGRAM_TOKEN="8537062091:AAELThmGGbEKHCCzP0T2vfKjc0IWVwRBxSU"
TELEGRAM_CHAT="5914916935"
NODES=("bourbaki" "ansatz" "deepseek")

send_telegram() {
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
        -H "Content-Type: application/json" \
        -d "{\"chat_id\": \"${TELEGRAM_CHAT}\", \"text\": \"$1\", \"parse_mode\": \"HTML\"}" \
        > /dev/null
}

recovered=()
still_down=()

for node in "${NODES[@]}"; do
    if ping -c1 -W2 "$node" &>/dev/null; then
        recovered+=("$node")
    else
        still_down+=("$node")
    fi
done

# Only send a message if at least one node is back
if [ ${#recovered[@]} -gt 0 ]; then
    slurm_info=$(sinfo -N -l 2>&1 | grep -E "bourbaki|ansatz|deepseek" | awk '{printf "%s: %s\\n", $1, $5}')

    recovered_str=$(printf "%s, " "${recovered[@]}")
    recovered_str="${recovered_str%, }"
    down_str=$(printf "%s, " "${still_down[@]}")
    down_str="${down_str%, }"

    msg="🟢 <b>Node Recovery Alert</b>

<b>Back online:</b> ${recovered_str}"

    if [ -n "$down_str" ]; then
        msg="${msg}
<b>Still down:</b> ${down_str}"
    fi

    msg="${msg}

<b>SLURM state:</b>
<code>${slurm_info}</code>"

    send_telegram "$msg"
    echo "[$(date)] ALERT sent — recovered: ${recovered_str}"
else
    echo "[$(date)] All nodes still down (bourbaki, ansatz, deepseek)"
fi
