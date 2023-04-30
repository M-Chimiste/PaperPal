import os
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

class GmailCommunication:

    def __init__(self, sender_address=None, app_password=None, creds_path=None):
        self.sender_address = sender_address
        self.app_password = app_password
        if creds_path:
            try:
                with open(creds_path, 'r') as f:
                        self.json_creds = json.load(f)
            except Exception as e:
                 raise Exception(f"Unable to read credentials path with error: {e}")

        if not self.app_password:
            self.app_password = os.getenv['app_password']
            if not self.app_password and creds_path:
                self.app_password = self.json_creds.get('app_password', None)
                if not self.app_password:
                    raise Exception("No application password found.  Please pass an application password or check your json for an app_password key.")
                
        if not self.sender_address and creds_path:
            self.sender_address = self.json_creds.get('sender_address', None)
            if not self.sender_address:
                raise Exception("No sender address found")
             
                
                