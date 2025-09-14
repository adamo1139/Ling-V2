#!/bin/bash
set -e

# Script to deploy Parquet support to Megatron preprocessing
# This script copies our modified preprocess_data.py to the Megatron installation

echo "=== Megatron Parquet Support Deployment Script ==="
echo

# Configuration
MEGATRON_TOOLS_DIR="Megatron-LM-core_v0.13.0/tools"
BACKUP_SUFFIX=".backup.$(date +%Y%m%d_%H%M%S)"
SOURCE_FILE="megatron_modified_preprocess_data_parquet.py"
TARGET_FILE="preprocess_data.py"

# Check if we're in the right directory
if [ ! -f "$SOURCE_FILE" ]; then
    echo "ERROR: Source file $SOURCE_FILE not found!"
    echo "Please run this script from the Ling-V2 directory"
    exit 1
fi

# Check if Megatron directory exists
if [ ! -d "$MEGATRON_TOOLS_DIR" ]; then
    echo "ERROR: Megatron directory $MEGATRON_TOOLS_DIR not found!"
    echo "Please ensure Megatron-LM-core_v0.13.0 is properly installed"
    exit 1
fi

TARGET_PATH="$MEGATRON_TOOLS_DIR/$TARGET_FILE"
BACKUP_PATH="$TARGET_PATH$BACKUP_SUFFIX"

echo "Source file: $SOURCE_FILE"
echo "Target file: $TARGET_PATH"
echo "Backup file: $BACKUP_PATH"
echo

# Check if target file exists
if [ -f "$TARGET_PATH" ]; then
    echo "Found existing $TARGET_FILE, creating backup..."
    cp "$TARGET_PATH" "$BACKUP_PATH"
    echo "✓ Backup created: $BACKUP_PATH"
else
    echo "No existing $TARGET_FILE found, no backup needed"
fi

# Copy our modified file
echo "Deploying Parquet support..."
cp "$SOURCE_FILE" "$TARGET_PATH"
echo "✓ Parquet support deployed successfully!"

# Verify the deployment
echo
echo "Verifying deployment..."
if [ -f "$TARGET_PATH" ]; then
    # Check if our modifications are present
    if grep -q "datasets_available" "$TARGET_PATH"; then
        echo "✓ Parquet support modifications detected in $TARGET_FILE"
    else
        echo "⚠ WARNING: Parquet support modifications not detected"
        echo "  The file was copied but may not contain the expected changes"
    fi
    
    # Check file permissions
    if [ -r "$TARGET_PATH" ] && [ -w "$TARGET_PATH" ]; then
        echo "✓ File permissions are correct"
    else
        echo "⚠ WARNING: File permissions may be incorrect"
    fi
else
    echo "✗ ERROR: Deployment verification failed"
    exit 1
fi

echo
echo "=== Deployment Complete ==="
echo
echo "Megatron preprocessing now supports Parquet files!"
echo
echo "Usage example:"
echo "  python $MEGATRON_TOOLS_DIR/preprocess_data.py --input your_dataset.parquet --output-prefix processed_data ..."
echo
echo "To restore the original file:"
echo "  cp $BACKUP_PATH $TARGET_PATH"
echo
echo "Note: You may need to install additional dependencies:"
echo "  pip install datasets pyarrow"
