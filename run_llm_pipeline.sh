#!/bin/bash
# Quick start script for LLM response generation pipeline
# Author: Junyao Liu

echo "=========================================="
echo "LLM Response Generation Pipeline"
echo "=========================================="
echo ""

# Check if API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY is not set!"
    echo ""
    echo "Please set your OpenAI API key:"
    echo "  export OPENAI_API_KEY='your-api-key-here'"
    echo ""
    exit 1
fi

echo "✅ OpenAI API key is set"
echo ""

# Check if required packages are installed
echo "Checking dependencies..."
python3 -c "import openai" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ openai package not found. Installing dependencies..."
    pip install -r requirements_llm.txt
fi

python3 -c "import bert_score" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  bert_score package not found. Installing..."
    pip install bert-score
fi

echo "✅ Dependencies are installed"
echo ""

# Default parameters
NUM_SAMPLES=${1:-20}
DATASET=${2:-both}

echo "=========================================="
echo "Step 1: Generate LLM Responses"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Number of samples: $NUM_SAMPLES"
echo ""

python3 generate_llm_responses.py \
    --dataset $DATASET \
    --num-samples $NUM_SAMPLES \
    --model gpt-3.5-turbo \
    --temperature 0.7 \
    --max-tokens 150

if [ $? -ne 0 ]; then
    echo "❌ Error generating responses!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 2: Compute Metrics"
echo "=========================================="
echo ""

# Compute metrics for DailyDialog if applicable
if [ "$DATASET" = "dailydialog" ] || [ "$DATASET" = "both" ]; then
    echo "Computing metrics for DailyDialog..."
    python3 compute_llm_metrics.py \
        --input data/llm_outputs/dailydialog_gpt35_responses.jsonl \
        --output data/llm_outputs/dailydialog_metrics.json \
        --bertscore-model microsoft/deberta-xlarge-mnli
    
    if [ $? -ne 0 ]; then
        echo "⚠️  Warning: Failed to compute metrics for DailyDialog"
    fi
    echo ""
fi

# Compute metrics for EmpatheticDialogues if applicable
if [ "$DATASET" = "empathetic_dialogues" ] || [ "$DATASET" = "both" ]; then
    echo "Computing metrics for EmpatheticDialogues..."
    python3 compute_llm_metrics.py \
        --input data/llm_outputs/empathetic_dialogues_gpt35_responses.jsonl \
        --output data/llm_outputs/empathetic_dialogues_metrics.json \
        --bertscore-model microsoft/deberta-xlarge-mnli
    
    if [ $? -ne 0 ]; then
        echo "⚠️  Warning: Failed to compute metrics for EmpatheticDialogues"
    fi
    echo ""
fi

echo "=========================================="
echo "✅ Pipeline Complete!"
echo "=========================================="
echo ""
echo "Output files saved to: data/llm_outputs/"
echo ""
echo "Generated files:"
ls -lh data/llm_outputs/
echo ""
echo "=========================================="
echo "Next Steps:"
echo "1. Review the metrics in data/llm_outputs/*_metrics.json"
echo "2. Analyze responses in data/llm_outputs/*_responses.jsonl"
echo "3. Perform qualitative error analysis"
echo "4. Compare with small transformer model results"
echo "=========================================="

