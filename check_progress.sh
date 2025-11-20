#!/bin/bash
# Script to check generation progress
# Usage: ./check_progress.sh

echo "============================================"
echo "LLM Response Generation Progress"
echo "============================================"
echo ""

# DailyDialog
dd_file="data/llm_outputs/dailydialog_gpt35_responses.jsonl"
dd_total=999

if [ -f "$dd_file" ]; then
    dd_count=$(wc -l < "$dd_file" | tr -d ' ')
    dd_percent=$(( dd_count * 100 / dd_total ))
    echo "📊 DailyDialog:"
    echo "   Progress: $dd_count / $dd_total samples ($dd_percent%)"
    echo "   File: $dd_file"
    
    if [ $dd_count -lt $dd_total ]; then
        echo "   Status: ⚠️  Incomplete"
        echo "   To resume: ./generate_llm_batch.sh dailydialog 100 processed_sample"
    else
        echo "   Status: ✅ Complete"
    fi
else
    echo "📊 DailyDialog:"
    echo "   Progress: 0 / $dd_total samples (0%)"
    echo "   Status: ❌ Not started"
fi

echo ""

# EmpatheticDialogues
ed_file="data/llm_outputs/empathetic_dialogues_gpt35_responses.jsonl"
ed_total=1644

if [ -f "$ed_file" ]; then
    ed_count=$(wc -l < "$ed_file" | tr -d ' ')
    ed_percent=$(( ed_count * 100 / ed_total ))
    echo "📊 EmpatheticDialogues:"
    echo "   Progress: $ed_count / $ed_total samples ($ed_percent%)"
    echo "   File: $ed_file"
    
    if [ $ed_count -lt $ed_total ]; then
        echo "   Status: ⚠️  Incomplete"
        echo "   To resume: ./generate_llm_batch.sh empathetic_dialogues 100 processed_sample"
    else
        echo "   Status: ✅ Complete"
    fi
else
    echo "📊 EmpatheticDialogues:"
    echo "   Progress: 0 / $ed_total samples (0%)"
    echo "   Status: ❌ Not started"
fi

echo ""
echo "============================================"

# Summary files
echo "📋 Summary Files:"
for summary in data/llm_outputs/*_summary.json; do
    if [ -f "$summary" ]; then
        echo "   - $(basename "$summary")"
    fi
done

echo "============================================"

