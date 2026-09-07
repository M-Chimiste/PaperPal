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

import os
import base64
import copy
import socket
import smtplib
import struct
from datetime import datetime
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import google.auth.transport.requests
import markdown2
import requests
from google.oauth2.credentials import Credentials
from ..data_access import SettingsRepository


SMTP_GMAIL_HOST = os.getenv("SMTP_GMAIL_HOST", "smtp.gmail.com")
SMTP_GMAIL_PORT = int(os.getenv("SMTP_GMAIL_PORT", "587"))
SMTP_DNS_FALLBACK_ENABLED = os.getenv("SMTP_DNS_FALLBACK_ENABLED", "true").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
SMTP_DNS_NAMESERVERS = [
    server.strip()
    for server in os.getenv(
        "SMTP_DNS_NAMESERVERS",
        "tcp://8.8.8.8,tcp://8.8.4.4,8.8.8.8,8.8.4.4",
    ).split(",")
    if server.strip()
]
SMTP_DNS_TIMEOUT_SEC = float(os.getenv("SMTP_DNS_TIMEOUT_SEC", "5"))
SMTP_CONNECT_TIMEOUT_SEC = float(os.getenv("SMTP_CONNECT_TIMEOUT_SEC", "20"))
GMAIL_API_SEND_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"
GMAIL_API_TIMEOUT_SEC = float(os.getenv("GMAIL_API_TIMEOUT_SEC", "30"))
GMAIL_TOKEN_FILE = os.getenv("GMAIL_TOKEN_FILE", "gmail_token.json")
GMAIL_SEND_SCOPES = ["https://www.googleapis.com/auth/gmail.send"]


def _build_dns_a_query(hostname: str, transaction_id: int) -> bytes:
    labels = hostname.rstrip(".").split(".")
    qname = b"".join(len(label).to_bytes(1, "big") + label.encode("idna") for label in labels) + b"\x00"
    header = struct.pack("!HHHHHH", transaction_id, 0x0100, 1, 0, 0, 0)
    return header + qname + struct.pack("!HH", 1, 1)


def _skip_dns_name(packet: bytes, offset: int) -> int:
    while True:
        if offset >= len(packet):
            raise ValueError("DNS packet ended before name parsing completed.")
        length = packet[offset]
        if length & 0xC0 == 0xC0:
            return offset + 2
        if length == 0:
            return offset + 1
        offset += 1 + length


def _resolve_hostname_via_nameserver(hostname: str, nameserver: str, timeout: float) -> str:
    transaction_id = int.from_bytes(os.urandom(2), "big")
    query = _build_dns_a_query(hostname, transaction_id)

    transport = "udp"
    server_address = nameserver
    if nameserver.startswith("tcp://"):
        transport = "tcp"
        server_address = nameserver[len("tcp://"):]
    elif nameserver.startswith("udp://"):
        server_address = nameserver[len("udp://"):]

    if transport == "tcp":
        with socket.create_connection((server_address, 53), timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall(struct.pack("!H", len(query)) + query)
            length_prefix = sock.recv(2)
            if len(length_prefix) != 2:
                raise socket.gaierror(f"Incomplete TCP DNS response header from {nameserver} for {hostname}")
            expected_length = struct.unpack("!H", length_prefix)[0]
            response = b""
            while len(response) < expected_length:
                chunk = sock.recv(expected_length - len(response))
                if not chunk:
                    raise socket.gaierror(f"Truncated TCP DNS response from {nameserver} for {hostname}")
                response += chunk
    else:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(timeout)
            sock.sendto(query, (server_address, 53))
            response, _ = sock.recvfrom(512)

    if len(response) < 12:
        raise socket.gaierror(f"Incomplete DNS response from {nameserver} while resolving {hostname}")

    response_id, flags, qdcount, ancount, _, _ = struct.unpack("!HHHHHH", response[:12])
    if response_id != transaction_id:
        raise socket.gaierror(f"Mismatched DNS transaction ID from {nameserver} while resolving {hostname}")

    rcode = flags & 0x000F
    if rcode != 0:
        raise socket.gaierror(f"DNS server {nameserver} returned rcode={rcode} for {hostname}")

    offset = 12
    for _ in range(qdcount):
        offset = _skip_dns_name(response, offset)
        offset += 4

    for _ in range(ancount):
        offset = _skip_dns_name(response, offset)
        if offset + 10 > len(response):
            raise socket.gaierror(f"Truncated DNS answer from {nameserver} for {hostname}")
        record_type, record_class, _, data_length = struct.unpack("!HHIH", response[offset:offset + 10])
        offset += 10
        record_data = response[offset:offset + data_length]
        offset += data_length
        if record_type == 1 and record_class == 1 and data_length == 4:
            return socket.inet_ntoa(record_data)

    raise socket.gaierror(f"No A record returned by {nameserver} for {hostname}")


def _resolve_hostname_with_fallback(hostname: str, verbose: bool = False) -> tuple[str, str]:
    last_error: Exception | None = None
    for nameserver in SMTP_DNS_NAMESERVERS:
        try:
            resolved_ip = _resolve_hostname_via_nameserver(hostname, nameserver, SMTP_DNS_TIMEOUT_SEC)
            if verbose:
                print(f"Resolved {hostname} via fallback DNS {nameserver} -> {resolved_ip}")
            return resolved_ip, nameserver
        except Exception as exc:  # noqa: BLE001 - we want to keep trying fallback resolvers
            last_error = exc
            if verbose:
                print(f"Fallback DNS query failed against {nameserver}: {exc}")
    if last_error is None:
        raise socket.gaierror(f"No fallback DNS servers configured for {hostname}")
    raise socket.gaierror(
        f"Unable to resolve {hostname} via fallback DNS servers {SMTP_DNS_NAMESERVERS}: {last_error}"
    )


class _FallbackResolvingSMTP(smtplib.SMTP):
    """SMTP client that retries hostname resolution against fallback DNS servers on gaierror."""

    def __init__(self, *args, fallback_verbose: bool = False, **kwargs):
        self._fallback_verbose = fallback_verbose
        super().__init__(*args, **kwargs)

    def _get_socket(self, host, port, timeout):
        try:
            return super()._get_socket(host, port, timeout)
        except socket.gaierror:
            if not SMTP_DNS_FALLBACK_ENABLED:
                raise
            resolved_ip, nameserver = _resolve_hostname_with_fallback(host, verbose=self._fallback_verbose)
            if self._fallback_verbose:
                print(f"Retrying SMTP connection to {host}:{port} using resolved IP {resolved_ip} from {nameserver}")
            return socket.create_connection((resolved_ip, port), timeout, self.source_address)


def create_gmail_smtp_session(verbose: bool = False) -> smtplib.SMTP:
    session = _FallbackResolvingSMTP(
        SMTP_GMAIL_HOST,
        SMTP_GMAIL_PORT,
        timeout=SMTP_CONNECT_TIMEOUT_SEC,
        fallback_verbose=verbose,
    )
    if verbose:
        session.set_debuglevel(1)
    return session


def _get_gmail_token_path() -> Path:
    return Path(GMAIL_TOKEN_FILE).expanduser().resolve()


def load_gmail_api_credentials() -> Credentials:
    token_path = _get_gmail_token_path()
    if not token_path.exists():
        raise FileNotFoundError(
            f"Gmail API token file not found at {token_path}. "
            "Run scripts/send_test_email.py --send --authorize to create it."
        )

    creds = Credentials.from_authorized_user_file(str(token_path), GMAIL_SEND_SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(google.auth.transport.requests.Request())
        token_path.write_text(creds.to_json())

    if not creds.valid:
        raise RuntimeError(
            f"Gmail API token at {token_path} is not valid. "
            "Re-run scripts/send_test_email.py --send --authorize."
        )

    return creds


def construct_email_body(content,
                         start_date,
                         end_date,
                         urls_and_titles):
    """
    Construct the body of an email.

    Args:
        content (str): The content of the email
        start_date (str): The start date of the newsletter
        end_date (str): The end date of the newsletter
        urls_and_titles (str): The URLs and titles of the references
    Returns:
        str: The body of the email
    """
    if start_date == end_date:
        date_range = start_date
    else:
        date_range = f"{start_date} - {end_date}"
    body = f"""## Theseus Insight Newsletter for {date_range}:

{content}

## References:
{urls_and_titles}
"""
    return body


class GmailCommunication:
    """
    Class for sending emails via Gmail.
    """
    def __init__(self,
                sender_address=None, 
                receiver_address=None, 
                app_password=None, 
                verbose=False):
        self.sender_address = sender_address
        self.app_password = app_password
        self.receiver_address = receiver_address
        self.email_message = None
        self.verbose = verbose
        
        if not self.app_password:
            self.app_password = os.getenv('GMAIL_APP_PASSWORD', None)
            if not self.app_password:
                raise Exception("No application password found. Please set the GMAIL_APP_PASSWORD environment variable.")
        if not self.sender_address:
            self.sender_address = os.getenv('GMAIL_SENDER_ADDRESS', None)
            if not self.sender_address:
                raise Exception("No sender address found. Please set the GMAIL_SENDER_ADDRESS environment variable.")

    def _build_api_delivery_message(self, message: MIMEMultipart, recipients: list[str] | None) -> MIMEMultipart:
        api_message = copy.deepcopy(message)
        if recipients:
            api_message["Bcc"] = ", ".join(recipients)
        return api_message

    def _send_via_smtp(self, message: MIMEMultipart, recipients: list[str] | None):
        sender_address = self.sender_address
        app_password = self.app_password

        session = create_gmail_smtp_session(verbose=self.verbose)
        if self.verbose:
            print("Connected to SMTP server")

        session.starttls()
        if self.verbose:
            print("Started TLS")

        session.login(sender_address, app_password)
        if self.verbose:
            print("Logged in successfully")

        message_text = message.as_string()
        self._smtp_submission_started = True
        if recipients:
            refused = session.sendmail(sender_address, [sender_address] + recipients, message_text)
            if refused:
                raise RuntimeError("Partial SMTP delivery; review recipients before retrying")
            if self.verbose:
                print("Successfully sent email to all BCC recipients via SMTP")
        else:
            session.sendmail(sender_address, sender_address, message_text)
            if self.verbose:
                print("Successfully sent email to sender via SMTP")

        try:
            session.quit()
        except Exception:
            session.close()
        if self.verbose:
            print("SMTP session closed")

    def _send_via_gmail_api(self, message: MIMEMultipart, recipients: list[str] | None):
        creds = load_gmail_api_credentials()
        api_message = self._build_api_delivery_message(message, recipients)
        raw_message = base64.urlsafe_b64encode(api_message.as_bytes()).decode("utf-8")
        headers = {
            "Authorization": f"Bearer {creds.token}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            GMAIL_API_SEND_URL,
            headers=headers,
            json={"raw": raw_message},
            timeout=GMAIL_API_TIMEOUT_SEC,
        )
        response.raise_for_status()
        if self.verbose:
            response_json = response.json()
            print(f"Successfully sent email via Gmail API (message id: {response_json.get('id', 'unknown')})")

    def _send_with_fallback(self, message: MIMEMultipart, recipients: list[str] | None):
        smtp_error = None
        self._smtp_submission_started = False
        try:
            self._send_via_smtp(message, recipients)
            return
        except Exception as exc:  # noqa: BLE001
            if self._smtp_submission_started:
                raise RuntimeError("SMTP submission outcome uncertain; automatic fallback suppressed to prevent duplicate delivery") from exc
            smtp_error = exc
            if self.verbose:
                print(f"SMTP delivery failed, trying Gmail API fallback: {exc}")

        try:
            self._send_via_gmail_api(message, recipients)
            return
        except Exception as api_exc:  # noqa: BLE001
            raise Exception(
                f"SMTP send failed: {smtp_error}; Gmail API send failed: {api_exc}"
            ) from api_exc


    def compose_message(self, content, start_date, end_date):
        """
        Compose an email message.
        
        Args:
            content (str): The content of the email
            start_date (str): The start date of the newsletter
            end_date (str): The end date of the newsletter
        """
        sender_address = self.sender_address
        receiver_address = self.receiver_address
        # If receiver_address is not provided, fetch from DB
        if not receiver_address:
            recipients = SettingsRepository.get_email_recipients()
            if recipients:
                receiver_address = recipients
            else:
                receiver_address = [sender_address]
        elif isinstance(receiver_address, str):
            receiver_address = [addr.strip() for addr in receiver_address.split(',')]
        # Remove sender's address from BCC list if it's there
        receiver_address = [addr for addr in receiver_address if addr != sender_address]
        
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
            start_date = start_date.strftime("%B %d, %Y")
        
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            end_date = end_date.strftime("%B %d, %Y")

        if start_date == end_date:
            date_range = start_date
        else:
            date_range = f"{start_date} to {end_date}"
        
        message = MIMEMultipart('alternative')
        message["From"] = sender_address
        message["To"] = sender_address  # Set To as the sender address
        # Remove the BCC header - it will be handled during sendmail
        if "Bcc" in message:
            del message["Bcc"]
        message['Subject'] = f"Theseus Insight Paper Newsletter for {date_range}"
        
        # Create both plain text and HTML versions
        text_part = MIMEText(content, 'plain')
        
        # Enhanced HTML generation with proper CSS styling
        html_content = self._create_enhanced_html(content)
        html_part = MIMEText(html_content, 'html')
        
        # Attach both versions - the email client will use the last attached version it can handle
        message.attach(text_part)
        message.attach(html_part)
        
        self.email_message = message
        # Store the receiver list separately
        self.receiver_list = receiver_address

    def _create_enhanced_html(self, content):
        from .rendering import render_newsletter_html
        return render_newsletter_html(content)


    def send_email(self):
        """
        Send an email.
        """
        sender_address = self.sender_address
        message = self.email_message

        try:
            if self.verbose:
                print(f"Attempting to send email from: {sender_address}")
                print(f"Recipients list: {self.receiver_list}")
                print(f"Message headers: {dict(message.items())}")
            self._send_with_fallback(message, self.receiver_list)

        except Exception as e:
            error_msg = f"Unable to send email with exception {str(e)}"
            print(f"Error details: {error_msg}")
            # Try to send error notification without raising
            try:
                self.send_error_notification(error_msg)
            except:
                pass
            raise Exception(error_msg)
                

    def send_error_notification(self, error_msg: str):
        """
        Send an error notification email to the sender.
        
        Args:
            error_msg (str): The error message to send
        """
        try:
            print(f"Attempting to send error notification to {self.sender_address}")
            
            message = MIMEMultipart('alternative')
            message["From"] = self.sender_address
            message["To"] = self.sender_address
            message['Subject'] = "Theseus Insight Error Notification"
            
            error_content = f"""## Theseus Insight Error Report

An error occurred during the Theseus Insight execution:

{error_msg}
"""
            
            text_part = MIMEText(error_content, 'plain')
            html_content = markdown2.markdown(error_content)
            html_part = MIMEText(html_content, 'html')
            
            message.attach(text_part)
            message.attach(html_part)
            self._send_with_fallback(message, None)
            if self.verbose:
                print("Error notification sent successfully")
            
        except Exception as e:
            if self.verbose:
                print(f"Failed to send error notification: {str(e)}")
