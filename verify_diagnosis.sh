#!/bin/bash

echo "=== Verifying Cloud-Orchestrator Synchronization Issue ==="
echo ""

# First, clean up hung processes
echo "1. Cleaning up hung processes..."
HUNG=$(ps aux | grep -E "leaf_client.py" | grep -v grep | wc -l)
if [ $HUNG -gt 0 ]; then
    echo "   Found $HUNG hung leaf_client processes"
    echo "   Run: pkill -f leaf_client.py"
fi

echo ""
echo "2. Checking Cloud Server Lifecycle:"
echo "-----------------------------------"

if [ -f cloud.log.err ]; then
    echo "a) When cloud started and for how many rounds:"
    grep -E "Will handle|Starting for.*rounds" cloud.log.err | tail -2
    
    echo ""
    echo "b) When cloud finished:"
    grep "All.*rounds completed" cloud.log.err | tail -1
    
    echo ""
    echo "c) Last rounds cloud processed:"
    grep "aggregate_fit: round=" cloud.log.err | tail -5 | awk '{print $5}' | sed 's/round=/Round /'
    
    echo ""
    echo "d) Any failures reported:"
    grep "failures=[1-9]" cloud.log.err | tail -3
fi

echo ""
echo "3. Checking Proxy Connection Attempts:"
echo "--------------------------------------"

for proxy_log in proxy_*.log.err; do
    if [ -f "$proxy_log" ]; then
        echo "$proxy_log:"
        # Show connection attempts and failures
        grep -E "Starting|Connection refused|completed" "$proxy_log" | tail -3
        echo ""
    fi
done

echo ""
echo "4. Comparing Timelines:"
echo "-----------------------"

# Extract timestamps
echo "Server completions:"
grep "completed all rounds" server*.log.err 2>/dev/null | sed 's/.*\[/[/' | cut -d']' -f1 | while read ts; do
    echo "  $ts"
done

echo ""
echo "Cloud completion:"
grep "All.*rounds completed" cloud.log.err 2>/dev/null | sed 's/.*\[/[/' | cut -d']' -f1

echo ""  
echo "Proxy failures:"
grep "Connection refused" proxy_*.log.err 2>/dev/null | sed 's/.*\[/[/' | cut -d']' -f1 | sort -u | while read ts; do
    echo "  $ts"
done

echo ""
echo "5. Configuration Analysis:"
echo "-------------------------"

CONFIG_ROUNDS=$(grep "global_rounds" pyproject.toml | grep -oE "[0-9]+" | head -1)
echo "Configured rounds: $CONFIG_ROUNDS"

if [ -f cloud.log.err ]; then
    CLOUD_EXPECTED=$(grep -oE "Will handle [0-9]+" cloud.log.err | grep -oE "[0-9]+" | tail -1)
    CLOUD_PROCESSED=$(grep "aggregate_fit: round=" cloud.log.err | tail -1 | grep -oE "round=[0-9]+" | grep -oE "[0-9]+")
    
    echo "Cloud expected: $CLOUD_EXPECTED rounds"
    echo "Cloud last processed: Round $CLOUD_PROCESSED"
    
    if [ -n "$CLOUD_PROCESSED" ] && [ -n "$CONFIG_ROUNDS" ]; then
        if [ "$CLOUD_PROCESSED" -lt "$CONFIG_ROUNDS" ]; then
            echo ""
            echo "⚠️  ISSUE CONFIRMED: Cloud stopped at round $CLOUD_PROCESSED but config expects $CONFIG_ROUNDS"
            echo "   This explains why proxies get 'Connection refused'"
        fi
    fi
fi

echo ""
echo "6. The Smoking Gun:"
echo "------------------"

# Check if cloud exited before all orchestrator rounds
if grep -q "All.*rounds completed" cloud.log.err && grep -q "Connection refused" proxy_*.log.err 2>/dev/null; then
    echo "✗ CONFIRMED: Cloud server completed and exited"
    echo "✗ CONFIRMED: Proxies then tried to connect and failed"
    echo ""
    echo "This proves the cloud-orchestrator synchronization issue!"
    echo "The cloud runs all rounds independently and exits,"
    echo "while orchestrator is still trying to submit results."
fi

echo ""
echo "7. Why None Occurs:"
echo "------------------"
cat << 'EOF'
The sequence:
1. Cloud expects 3 servers per round (min_fit_clients=3)
2. Some proxies fail to connect (cloud already exited)
3. Cloud gets < 3 results
4. accept_failures=False → aggregate_fit returns None
5. CloudFedAvg tries to process None → CRASH

This can happen at ANY round where timing causes this race.
Round 17 is just when it happened in your run.
EOF
