#!/usr/bin/env bash

cleanup() {
    # Remove any artifacts from the previous build
    if [ -d target ]; then
        rm -r target
    fi
}

download_models() {
    mkdir -p target
    mkdir -p target/model
    mkdir -p target/nltk_data

    python -m nltk.downloader all -d ./target/nltk_data
    export NLTK_DATA=./target/nltk_data/
}

download_models