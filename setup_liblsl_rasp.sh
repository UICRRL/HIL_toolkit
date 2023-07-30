#!/bin/bash

# Remember the current directory
current_dir=$(pwd)

# Step 1: Git clone the repository in the parent folder
cd ..
git clone https://github.com/sccn/liblsl

# Step 2: Run the script in the repository
cd liblsl
sudo ./standalone_compilation_linux.sh

# Step 3: Add liblsl.so path to PYLSL_LIB
liblsl_path=$(realpath liblsl.so)
echo "export PYLSL_LIB=$liblsl_path" >> ~/.bashrc

# Load the new environment variable for the current session
source ~/.bashrc

# Go back to the original folder
cd "$current_dir"

echo "Setup complete. The liblsl.so file has been added to PYLSL_LIB in bash."
