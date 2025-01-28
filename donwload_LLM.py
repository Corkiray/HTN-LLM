#!/usr/bin/env python
# coding: utf-8

from huggingface_hub import snapshot_download
import json as js

KEYS_FILE = '/mnt/homeGPU/ipuerta/HTN-LLM/keys.json'
with open(KEYS_FILE) as f:
    keys_data = js.load(f)

token = keys_data['hf_token']


model = "neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w4a16"

snapshot_download(repo_id=model, local_dir='/mnt/homeGPU/ipuerta/Models/' + model, 
                  cache_dir='/mnt/homeGPU/ipuerta/Models/mycache', token=token)
