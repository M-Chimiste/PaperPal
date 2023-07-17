# PaperPal
Tool for helping me sort papers off my own research interests.  I've tried to make it relatively accessible and hackable so that anyone can adapt it for their own uses.

Please note that this repository is currently a minimum "viable" product.  It may have additional features over time.

## Key Features
- Vicuna 13B model integration for summarization and recommendation.
- Automated papers with code dowloads (I can use the client, but it's slower and more annoying than just processing the json).
- Saving outputs as csv or excel (I'm having issues with sqlite for right now but should have them solved soon). 
- Automated emails sent after completion (presently only planning integration with gmail).

## Requirements
Before we get started it's important to note a few things about my current set up and PaperPal in general.

- You're probably going to need a pretty good machine for this.  I'm not going to explicitly support CPUs at this time, but I might try to support M1/2 Macs since I've got one.  If you want to add CPU support or MPS support please feel free and I'll totally merge that.
  - My current system specs:
  - AMD Ryzen 7900X
  - 128GB GDDR5 RAM
  - 2 x RTX 3090s with NVLink
- The default settings inside my arguments are selfish.  I've set it up for the ease of my own use and I highly recommend you configure them for your own useage as needed.  
- I'm presently only supporting sending emails through Gmail.  You can very easily modify my communication.py file to add in normal SMTP support or other API calls.  Gmail just seemed easiest since I don't have an outlook server.  To configure an application password with Gmail check out [these instructions](https://support.google.com/mail/answer/185833?hl=en).  You need to pass these as an argument or add them into a credentials json.  You can see a sample of the credentials in the config folder.

## Use
- You'll want to clone the repo and install the requirements (I'd recommend a virtual environment for this).  I'm currently using 1.13 for my version of pytorch with CUDA 11.7, but you should be able to get away with pytorch 2.0 and a different version of CUDA.  I cannot vouch for older version of pytorch.
- I recommend downloading your desired model of LLM inference first.  You can check out [huggingface](https://huggingface.co/models?search=vicuna) for some models as I'm not planning to distribute any models until either StableLM releases a 15B model or Red Pyjama releases a fully permissible LLaMA.  The application is presently set up the best for Vicuna 1.1 in re to the prompts being provided.  I've done testing on both the 7B and 13B vicuna models and they should work.  For the 7B model, you may need to clean up the stop tokens in outputs, but that should be simple enough.  You probably can use other models or versions of Vicuna.  I'm using AutoTokenizer and AutoModelForCausalLM, but you'll want to check out and modify the prompts in the inference.py file.  Also, if you're unhappy with outputs, I really recommend playing around with the prompts.  Also do let me know if you get interesting results with a specific prompt.
- You will want to update the research_intrests.txt file for your own interests.  The format is exactly as the example.  I'm listing them as a numbered list, but you should be able to change that however you want.  Just realize this will change how your prompt is generated so do keep that in mind.  Under the hood I'm just creating them as a single string with new lines at the end of each interest.
- Running the script is pretty easy.  In the project directory you can run ```python paperpal/paperpal.py --start_date "2023-04-24"``` and that should launch the program and have it pull data from that date.  You will want to make sure you look at all the arguments to ensure you like the default arguments or pass / change the arguments for your own purpose.

## ToDo
- Configuration file for passing arguments versus just argparse (allow option to do either).
- Dump data into a database / Opensearch for follow on analysis.

## Completed
- Code to download papers with code data and process them.
- Code to send emails using Gmail.
- Generate prompt templates and serving code for Vicuna or other LLMs.
- Write script to run entire pipeline end-to-end.
- Generate a requirements file.
- Write some basic documentation.
