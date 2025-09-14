# IMPersona: Enabling Individual Impersonation for LLMs

Website: [https://impersona-website.vercel.app](https://impersona-website.vercel.app)

## Overview
![performance_teaser](https://github.com/user-attachments/assets/14e1f72f-9f89-4e3d-b4a0-fd73d4a6cecf)

This repository contains the code to replicate the experiments in the paper "IMPersona: Enabling Individual Impersonation for LLMs" to create custom IMPersonas for LLMs. We support data downloaded from iMessage and Facebook Messenger only currently.

### Inference Flags:
* ICL (Sample chats in prompt)
* Memory (Baseline memory: recommended for basic testing)
* Hierarchical Memory (Best performance: but much slower inference)

### APIs Supported 
* OpenAI (Prompting), Anthropic (Prompting), Together (Prompting, Finetuning)
* Local Custom Models (Prompting, Finetuning)

### Models Recommended:
* Training-Free: Claude-3.5-Sonnet
* Training: Llama-3.1-8B-Instruct

## How to Use

### Part 1: Downloading the Data

1. Clone and navigate to the repository.
   ```bash
   git clone https://github.com/princeton-nlp/p-impersona
   cd p-impersona
   ```

2. Create a virtual environment and install dependencies.
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Download the imessage-exporter. If using Homebrew on Mac, run:
   ```bash
   brew install imessage-exporter
   ```
   
   If using cargo:
   ```bash
   cargo install imessage-exporter
   ```
   
   You may need to install xcode developer tools first:
   ```bash
   xcode-select --install
   ```
   
   You may otherwise install it manually:
   ```bash
   git clone https://github.com/ReagentX/imessage-exporter
   cd imessage-exporter
   cargo run --release
   ```

4. Export your iMessage data from your Mac and/or iPhone.
   
   ### Mac Export (Basic)
   ```bash
   # Export from your Mac
   imessage-exporter --format txt --export-path imessage_export_mac
   ```

   ### iPhone Export
   If your Mac is missing many conversations, follow these steps to export from your iPhone:

   1. **Backup your iPhone**
      - Connect your iPhone to your Mac via USB and unlock your phone
      - Click "Trust this computer" if prompted
      - Open Finder and navigate to your iPhone under Locations
      - Make sure "Encrypt local backups" is turned OFF
      - Click "Back Up Now" and wait for the backup to complete (10-30 minutes)
      - Once complete, you can eject and disconnect your iPhone

   2. **Find your backup location**
      ```bash
      # List backups (most recent will be last)
      ls -latr /Users/$(id -un)/Library/Application\ Support/MobileSync/Backup/
      ```
      - Note the most recent backup folder (e.g., `00004540-000107C20F90003H`)

   3. **Export messages from your backup**
      ```bash
      # Replace <backup_path_here> with your backup folder path
      imessage-exporter --format txt --export-path imessage_export_ios --db-path /Users/$(id -un)/Library/Application\ Support/MobileSync/Backup/<backup_path_here>
      ```

   4. **Merge data** (if you exported from both sources)
      ```bash
      python merge_imessage_exports.py
      ```
      This will save the combined data in `imessage_export_mac_and_ios`

   ### Facebook Messenger Export
   If you use Facebook Messenger, you can export your Messenger data too.

   1. **Download data from Facebook Messenger**
      - Go to [Messenger for web](https://www.messenger.com/)
      - Click on your profile picture in the corner
      - Click on "Privacy and Safety"
      - Click on "End-to-end encrypted chats"
      - Click on "Message storage"
      - Click on "Download secure storage data"
      - Click on "Download file" (this may take a few minutes to start downloading)
      - Move the downloaded file to this repo's root directory and name is `messenger_export.zip`
   2. **Unzip the file**
      ```bash
      unzip messenger_export.zip -d messenger_export
      ```
   3. **Convert Messenger data to iMessage data**
      ```bash
      python convert_messenger_data.py --me_participant <your_name_here>
      ```
      - This will automatically move the converted data to `data/imessage_export`, so you don't need to do anything else after running the script.
      - You can remove the `messenger_export` folder after running the script.

   ### Move the final export to the required location
      ```bash
      # Choose either the mac-only or the merged export
      mv imessage_export_mac_and_ios data/imessage_export
      # OR
      mv imessage_export_mac data/imessage_export
      ```

5. **[Highly Recommended]** The above export utilizes phone numbers to identify users. To use names instead, you will need to export your contacts as a vcf file. Some users have reported issues with their mac and phone contacts not syncing: use whichever contact app contains the updated information.
   
   - **On your Mac:** Go to `Contacts (Mac App) -> Select All (Cmd + A) -> File -> Export -> Export vCard`. 
   - **On your phone:** Go to `Contacts (Phone App) -> Lists -> Long Press All Contacts -> Export -> AirDrop to Mac`.
   - Rename and move the file to `data/contacts.vcf`.

### Part 2: Processing the Data

1. Run the following command to process the data files for training. The terminal will prompt you to input any contacts that were left out of the vcf file, so pay attention to the output.
   ```bash
   python process_imessage.py
   ```

2. Create a `.env` file in the root directory. Add your API key for OpenAI, named `OPENAI_API_KEY`, as well as other APIs that you plan to use.
   ```
   # .env
   OPENAI_API_KEY=<your_api_key>
   ANTHROPIC_API_KEY=<your_api_key>
   TOGETHER_API_KEY=<your_api_key>
   ```

### Part 3: Training

Training is only necessary if you want to create finetuned IMPersonas. If you wish to only interact with prompting based IMPersonas, **you may skip this step.** If you have the resources to do so, we recommend training locally and on the full dataset for best results. TODO: better configuration for low-memory settings.

#### If Training Locally/On Personal Cluster

Option 1: Use Hugging Face
1. We provide a script `IMPersona/train.py` that will train a finetuned IMPersona locally. You may need to install additional dependencies. To finetune Llama-3.1-8B-Instruct, on the full dataset, run the following command:
   ```bash
   python IMPersona/train.py \
     --model_name meta-llama/Llama-3.1-8B-Instruct \
     --dataset_path ./data/impersona_imessage_0buffer_BFull.json \
     --output_dir ./output \
     --num_epochs 3 \
     --learning_rate 1e-4 \
     --use_lora \
     --lora_r 8 \
     --lora_alpha 32 \
     --lora_dropout 0.05 \
     --format_template llama
   ```
   - Note: The default effective batch size is 8. If you need to reduce the batch size, you may increase the gradient accumulation steps to compensate.

Option 2: Use LLaMA-Factory (Recommended for low-memory settings)
1. Follow the instructions [here](https://github.com/Lightning-AI/llama-factory) to install LLaMA-Factory.
2. Use LLaMA-Factory scripts for lora training and inference (training sets already in proper format)

#### If Using [Together API](https://docs.together.ai/docs/fine-tuning-overview)

1. In the command line, run the following with your API key:
   ```bash
   export TOGETHER_API_KEY=<your_api_key>
   ```

2. Run the following commands to check + upload the dataset to Together. Keep track of the file ids generated: you will need them to submit fine-tuning jobs.
   ```bash
   together files check data/<your_name_here>_impersona_imessage_0buffer_BFull_together_format.jsonl
   ```

   If you cannot find the file id, run the following to see a list of files:
   ```bash
   together files list
   ```

3. Submit fine-tuning job(s) to the Together API.
   ```bash
   together fine-tuning create \
     --training-file <file_id> \
     --model meta-llama/Meta-Llama-3.1-8B-Instruct-Reference \
     --lora \
     --suffix <your_name_here>-BFull \
     --n-epochs 3 \
     --batch-size 8 \
     --learning-rate 0.0001 
   ```

   - Check the status of your fine-tuning jobs with the following command.
     ```bash
     together fine-tuning list
     ```

4. After the job has finished, save/keep track of the value in model output name. This is how you will call your models.

5. If you need to see the model output name again, you can see them with the following command.
   ```bash
   together fine-tuning list
   ```

### Part 4: Creating Memory Banks

1. Run the following command to create a memory bank for your IMPersona. This is necessary for the memory inference setting. This section will require an OpenAI API key.
   ```bash
   python process_memory.py
   ```
   
   Note: You may encounter errors in this step in the terminal. If you do, just rerun the script.

2. [Optional] To visualize the memories created, feel free to run the following web UI:
   ```bash
   python memory_visualizer.py
   ```

### Part 5: Inference

1. The `run_impersona_chat.py` script allows you to converse with your IMPersona in a chat-room esque UI. Simply run the following command, fill in the parameters, and start chatting!
   ```bash
   python run_impersona_chat.py
   ```

   Alternatively, to chat with your IMPersona in the terminal, you may use `run_impersona.py`. For example, to run Claude with icl and memory, run the following command:
   ```bash
   python run_impersona.py --model_name claude-3-5-sonnet-20241022 --impersonation_name <your_name_here> --memory --icl
   ```

2. [Optional] For those operating the Human or Not IMPersona task, you may use the `run_impersona_web.py` script to interface with the impersona. **The training pipeline is optimized for this usage.** The script takes in a full conversation in the format given by the web interface https://impersona-web.vercel.app/ and will output a response with `<|>` serving as message delimiter. It takes in the same arguments as `run_impersona.py`. For example, to run Llama-BFull + BM, run the following command:
   ```bash
   # llama-BFull + BM
   python run_impersona_web.py --model_name <finetuned_model_name_here> --memory --impersonation_name <your_name_here>
   ```


