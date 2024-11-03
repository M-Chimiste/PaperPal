# Copyright 2023 M Chimiste

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def construct_email_body(content,
                         start_date,
                         end_date,
                         urls_and_titles):
    if start_date == end_date:
        date_range = start_date
    else:
        date_range = f"{start_date} - {end_date}"
    body = f"""Greetings!  Here is your digest for the time period {date_range}:

{content}

References:
{urls_and_titles}
"""
    return body


class GmailCommunication:

    def __init__(self, sender_address=None, receiver_address=None, app_password=None):
        self.sender_address = sender_address
        self.app_password = app_password
        self.receiver_address = receiver_address
        self.email_message = None
        
        if not self.app_password:
            self.app_password = os.getenv('GMAIL_APP_PASSWORD', None)
            if not self.app_password:
                raise Exception("No application password found. Please set the GMAIL_APP_PASSWORD environment variable.")
                
        if not self.sender_address:
            self.sender_address = os.getenv('GMAIL_SENDER_ADDRESS', None)
            if not self.sender_address:
                raise Exception("No sender address found. Please set the GMAIL_SENDER_ADDRESS environment variable.")


    def compose_message(self, content, start_date, end_date):
        """Method to compose a MIMEMultipart Message

        Args:
            content (str): string email content for generating an email from.
        """
        sender_address = self.sender_address
        receiver_address = self.receiver_address

        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
            start_date = start_date.strftime("%B %d, %Y")
        
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            end_date = end_date.strftime("%B %d, %Y")

        if isinstance(receiver_address, list):
            receiver_address = ', '.join(receiver_address)
        if start_date == end_date:
            date_range = start_date
        else:
            date_range = f"{start_date} to {end_date}"
        
        if not receiver_address:  # we send the email to ourselves if we aren't sending it to someone else.
            receiver_address = sender_address
        
        message = MIMEMultipart()
        message["From"] = sender_address
        message["To"] = receiver_address
        message['Subject'] = f"PaperPal Paper Assessment for {date_range}"
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
                