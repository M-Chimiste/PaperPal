import os
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import datetime

class GmailCommunication:

    def __init__(self, sender_address=None, receiver_address=None, app_password=None, creds_path=None):
        self.sender_address = sender_address
        self.app_password = app_password
        self.receiver_address = receiver_address
        self.email_message = None
        
        if creds_path:
            try:
                with open(creds_path, 'r') as f:
                        self.json_creds = json.load(f)
            except Exception as e:
                 raise Exception(f"Unable to read credentials path with error: {str(e)}")

        if not self.app_password:
            self.app_password = os.getenv('app_password')
            if not self.app_password:
                if not creds_path:
                    raise Exception("No application password found.  Please pass an application password or check your json for an app_password key.")
                self.app_password = self.json_creds.get('app_password', None)
                if not self.app_password:
                    raise Exception("No application password found.  Please pass an application password or check your json for an app_password key.")
                
        if not self.sender_address:
            self.sender_address = os.getenv('sender_address')
            if not self.sender_address:
                if not creds_path:
                    raise Exception("No sender address found.  Please pass a sender address or check your json for sender_address key.")
                self.sender_address = self.json_creds.get('sender_address', None)
                if not self.sender_address:
                    raise Exception("No sender address found.  Please pass a sender address or check your json for sender_address key.")


    def compose_message(self, content):
        """Method to compose a MIMEMultipart Message

        Args:
            content (str): string email content for generating an email from.
        """
        sender_address = self.sender_address
        receiver_address = self.receiver_address
        today = datetime.datetime.today()
        today = today.strftime('%B %d, %Y')
        
        if not receiver_address:  # we send the email to ourselves if we aren't sending it to someone else.
            receiver_address = sender_address
        
        message = MIMEMultipart()
        message["From"] = sender_address
        message["To"] = receiver_address
        message['Subject'] = f"PaperPal Paper Assessment for {today}"
        message.attach(MIMEText(content, 'plain'))
        self.email_message = message
    

    def send_email(self):

        sender_address = self.sender_address
        receiver_address = self.receiver_address
        
        if not receiver_address:
            receiver_address = sender_address
        
        app_password = self.app_password
        message = self.email_message
        
        try:
            session = smtplib.SMTP('smtp.gmail.com', 587)
            session.starttls()
            session.login(sender_address, app_password)
            message_text  = message.as_string()
            session.sendmail(sender_address, receiver_address, message_text)
            session.quit()
        except Exception as e:
            raise Exception(f"Unable to send email with exception {str(e)}")
                