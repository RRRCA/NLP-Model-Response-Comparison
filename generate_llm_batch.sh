#!/bin/bash
# Script for automatically generating LLM responses in batches
# Usage: ./generate_llm_batch.sh <dataset> <batch_size> <data_source>
# Example: ./generate_llm_batch.sh dailydialog 100 processed_sample

set -e  # Exit immediately on error

DATASET=$1
BATCH_SIZE=${2:-100}  # Default batch size: 100
DATA_SOURCE=${3:-processed_sample}  # Default data source: processed_sample
AUTO_RESUME=${4:-false}  # Auto-resume mode: true/false (default: false)

# Check parameters
if [ -z "$DATASET" ]; then
    echo "Usage: $0 <dataset> [batch_size] [data_source] [auto_resume]"
    echo "  dataset: dailydialog | empathetic_dialogues | both"
    echo "  batch_size: Number of samples per batch (default: 100)"
    echo "  data_source: processed_sample | full_processed_sample (default: processed_sample)"
    echo "  auto_resume: true | false - Auto-resume without asking (default: false)"
    echo ""
    echo "Examples:"
    echo "  $0 dailydialog 100 processed_sample"
    echo "  $0 empathetic_dialogues 50 processed_sample"
    echo "  $0 dailydialog 100 processed_sample true  # Auto-resume if interrupted"
    exit 1
fi

# Function to get dataset size
get_dataset_size() {
    local dataset=$1
    if [ "$dataset" = "dailydialog" ]; then
        echo 999
    elif [ "$dataset" = "empathetic_dialogues" ]; then
        echo 1644
    else
        echo 0
    fi
}

# Function to process a single dataset
process_dataset() {
    local dataset=$1
    local total=$(get_dataset_size "$dataset")
    
    if [ "$total" -eq 0 ]; then
        echo "❌ Unknown dataset: $dataset"
        return 1
    fi
    
    # Check if output file exists and count existing samples
    local output_file="data/llm_outputs/${dataset}_gpt35_responses.jsonl"
    local existing_count=0
    
    if [ -f "$output_file" ]; then
        existing_count=$(wc -l < "$output_file" | tr -d ' ')
        echo ""
        echo "⚠️  Found existing output file with $existing_count samples"
        
        if [ "$AUTO_RESUME" = "true" ]; then
            echo "✅ Auto-resume enabled: continuing from sample $existing_count..."
        else
            echo "Do you want to:"
            echo "  1) Resume from sample $existing_count (recommended)"
            echo "  2) Start over (will backup existing file)"
            echo "  3) Cancel"
            read -p "Enter choice [1-3]: " choice
            
            case $choice in
                1)
                    echo "✅ Resuming from sample $existing_count..."
                    ;;
                2)
                    local backup_file="${output_file}.backup_$(date +%Y%m%d_%H%M%S)"
                    mv "$output_file" "$backup_file"
                    echo "✅ Backed up to: $backup_file"
                    echo "✅ Starting fresh..."
                    existing_count=0
                    ;;
                3)
                    echo "❌ Cancelled by user"
                    return 0
                    ;;
                *)
                    echo "❌ Invalid choice, cancelling..."
                    return 1
                    ;;
            esac
        fi
        echo ""
    fi
    
    echo "============================================"
    echo "Processing: $dataset"
    echo "Total samples: $total"
    echo "Starting from: $existing_count"
    echo "Remaining: $((total - existing_count))"
    echo "Batch size: $BATCH_SIZE"
    echo "Data source: $DATA_SOURCE"
    echo "============================================"
    
    local offset=$existing_count
    local batch_num=$(( (existing_count / BATCH_SIZE) + 1 ))
    
    while [ $offset -lt $total ]; do
        local remaining=$((total - offset))
        local current_batch=$BATCH_SIZE
        
        if [ $remaining -lt $BATCH_SIZE ]; then
            current_batch=$remaining
        fi
        
        echo ""
        echo "📊 Batch $batch_num: Samples $offset - $((offset + current_batch - 1))"
        echo "----------------------------------------"
        
        if [ $offset -eq 0 ]; then
            # First batch: overwrite mode (only when starting fresh)
            python generate_llm_responses.py \
                --dataset "$dataset" \
                --num-samples $current_batch \
                --offset $offset \
                --data-source "$DATA_SOURCE"
        else
            # Subsequent batches or resuming: append mode
            python generate_llm_responses.py \
                --dataset "$dataset" \
                --num-samples $current_batch \
                --offset $offset \
                --append \
                --data-source "$DATA_SOURCE"
        fi
        
        if [ $? -ne 0 ]; then
            echo ""
            echo "❌ Error occurred during batch $batch_num"
            echo "💾 Progress saved: $offset samples completed"
            echo "🔄 To resume, run the script again and choose option 1"
            return 1
        fi
        
        offset=$((offset + current_batch))
        batch_num=$((batch_num + 1))
        
        echo "✅ Batch completed! Processed $offset / $total samples ($(( offset * 100 / total ))%)"
        echo "💾 Progress saved to: $output_file"
        
        # Brief pause to avoid API rate limiting
        if [ $offset -lt $total ]; then
            echo "⏸️  Pausing for 2 seconds..."
            echo "   (Press Ctrl+C to stop. Progress is saved and can be resumed)"
            sleep 2
        fi
    done
    
    echo ""
    echo "🎉 Dataset $dataset completed! Total processed: $total samples"
    echo ""
}

# Main logic
if [ "$DATASET" = "both" ]; then
    echo "Processing both datasets..."
    process_dataset "dailydialog"
    process_dataset "empathetic_dialogues"
else
    process_dataset "$DATASET"
fi

echo "============================================"
echo "✨ All batches completed!"
echo "Output directory: data/llm_outputs/"
echo "============================================"

