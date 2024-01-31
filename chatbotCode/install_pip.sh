#!/bin/bash

# Define the packages you want to install
packages=("pyautogen[retrievechat]" "openai")

# Iterate through the packages
for package in "${packages[@]}"; do
    # Check if the package is already installed
    if pip show "$package" > /dev/null 2>&1; then
        echo "$package is already installed."
    else
        # Install the package
        pip install "$package"
        if [ $? -eq 0 ]; then
            echo "Successfully installed $package."
        else
            echo "Failed to install $package."
        fi
    fi
done
