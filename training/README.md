# Pensez Training Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Training Configuration](#training-configuration)
4. [Data Preparation](#data-preparation)
5. [Training](#training)

## Introduction
This guide provides instructions for training models using the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

## Installation
To install the necessary tools, follow the [installation instructions](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#installation) provided by LLaMA-Factory.

## Training Configuration
The training configuration is specified in the `pensez/assets/models/fr_full_sft.yaml` file. This file contains settings for model training, such as hyperparameters and paths to data files. Ensure this file is correctly set up before starting training.

## Data Preparation
Data is specified in the `data/dataset_info.json` file. This file should include:
```json
{
  "pensez": {
    "hf_hub_url": "HoangHa/Pensez-v0.1-formatted",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }
  }
}
```
Ensure your data is formatted correctly and accessible.

## Training
To start training, use the following command:

```bash
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/fr_full_sft.yaml
```

This command initiates the training process using the specified configuration file.