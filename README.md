# PaperPal
Tool for helping me sort papers off my own research interests.

Please note that this repository is currently a work in progress.

## Key Features
- Vicuna 13B model integration for summarization and recommendation.
- Automated papers with code dowloads (I can use the client, but it's slower and more annoying than just processing the json).
- Ingestion of recommendations into sqlite3 database for historical tracking (I may convert this to an OpenSearch instance at a later date).
- Automated emails sent after completion (presently only planning integration with gmail).

## ToDo
- Generate a requirements file.
- Generate prompt templates and serving code for Vicuna or other LLMs.
- Write script to run entire pipeline end-to-end.
- Write some basic documentation.

## Completed
- Code to download papers with code data and process them.
- Code to generate a database and code to insert data into it.
- Code to send emails using Gmail.
