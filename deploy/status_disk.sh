#!/bin/bash
echo "=== df ==="
df -h / | tail -1
echo "=== home bags ==="
ls -lh /home/jetbot/*.bag* 2>/dev/null
echo "=== maps ==="
ls -lh /home/jetbot/maps | head
echo "=== current sidecars ==="
BAG=$(cat /home/jetbot/current_survey_bag 2>/dev/null || true)
if [ -n "$BAG" ]; then
  echo "$BAG"
  ls -lh "$BAG"* 2>/dev/null || echo no_current_output
else
  echo no_current_survey_bag
fi
