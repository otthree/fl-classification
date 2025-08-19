#!/bin/bash

# sync_data.sh - Script to synchronize data to a remote machine
# Usage: ./sync_data.sh [machine_name]
# If machine_name is not provided

# Set default machine name
DEFAULT_MACHINE="<MACHINE_NAME>"
MACHINE=${1:-$DEFAULT_MACHINE}
DOMAIN="<DOMAIN>"
REMOTE_HOST="<USERNAME>@$MACHINE$DOMAIN"
SOURCE_BASE="<SOURCE_BASE_PATH>"
DEST_BASE="<DEST_BASE_PATH>"
SSH_KEY="<PRIVATE_KEY_PATH>"

# Create secure directory for SSH control socket in home directory
SSH_CONTROL_DIR="$HOME/.ssh/controlmasters"
mkdir -p "$SSH_CONTROL_DIR"
chmod 700 "$SSH_CONTROL_DIR"  # Ensure only user can access this directory
SSH_CONTROL_PATH="$SSH_CONTROL_DIR/ssh-control-$MACHINE-$$"

# Setup SSH control socket to reuse connection
echo "Setting up SSH connection to $REMOTE_HOST..."
ssh -i "$SSH_KEY" -o ControlMaster=yes -o ControlPath="$SSH_CONTROL_PATH" -o ControlPersist=600 "$REMOTE_HOST" "mkdir -p $DEST_BASE/data/ADNI"
if [ $? -ne 0 ]; then
    echo "Error establishing SSH connection!"
    exit 1
fi

echo "Syncing data to $REMOTE_HOST..."

# Sync ICBM152
echo "Syncing ICBM152 directory..."
rsync -Pav -e "ssh -i $SSH_KEY -o ControlPath=$SSH_CONTROL_PATH" "$SOURCE_BASE/data/ICBM152" "$REMOTE_HOST:$DEST_BASE/data/"
if [ $? -ne 0 ]; then
    echo "Error syncing ICBM152 directory!"
    ssh -i "$SSH_KEY" -o ControlPath="$SSH_CONTROL_PATH" -O exit "$REMOTE_HOST" 2>/dev/null
    exit 1
fi

# Sync MRI zip file
echo "Syncing MRI zip file..."
rsync -Pav -e "ssh -i $SSH_KEY -o ControlPath=$SSH_CONTROL_PATH" "$SOURCE_BASE/data/ADNI/ALL_1.5T_bl_ScaledProcessed_MRI_611images_step3_skull_stripping.zip" "$REMOTE_HOST:$DEST_BASE/data/ADNI/"
if [ $? -ne 0 ]; then
    echo "Error syncing MRI zip file!"
    ssh -i "$SSH_KEY" -o ControlPath="$SSH_CONTROL_PATH" -O exit "$REMOTE_HOST" 2>/dev/null
    exit 1
fi

# Sync CSV files
echo "Syncing CSV files..."
# Use find to get the list of CSV files, then rsync them if they exist
CSV_FILES=$(find "$SOURCE_BASE/data/ADNI" -maxdepth 1 -name "*.csv" 2>/dev/null)
if [ -n "$CSV_FILES" ]; then
    rsync -Pav -e "ssh -i $SSH_KEY -o ControlPath=$SSH_CONTROL_PATH" $CSV_FILES "$REMOTE_HOST:$DEST_BASE/data/ADNI/"
    if [ $? -ne 0 ]; then
        echo "Error syncing CSV files!"
        ssh -i "$SSH_KEY" -o ControlPath="$SSH_CONTROL_PATH" -O exit "$REMOTE_HOST" 2>/dev/null
        exit 1
    fi
else
    echo "No CSV files found in $SOURCE_BASE/data/ADNI - skipping CSV sync"
fi

# Close SSH control connection
ssh -i "$SSH_KEY" -o ControlPath="$SSH_CONTROL_PATH" -O exit "$REMOTE_HOST" 2>/dev/null

echo "Data synchronization completed successfully!"
