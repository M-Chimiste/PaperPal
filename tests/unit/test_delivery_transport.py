from types import SimpleNamespace
import pytest


def test_uncertain_smtp_submission_does_not_fallback():
    from theseus_insight.communication.communication import GmailCommunication
    sent = []
    instance = SimpleNamespace(verbose=False)
    def smtp(*args):
        instance._smtp_submission_started = True
        raise TimeoutError('lost response after submission')
    instance._send_via_smtp = smtp
    instance._send_via_gmail_api = lambda *args: sent.append('gmail')
    with pytest.raises(RuntimeError, match='uncertain'):
        GmailCommunication._send_with_fallback(instance, None, [])
    assert sent == []


def test_pre_submission_failure_can_fallback():
    from theseus_insight.communication.communication import GmailCommunication
    sent = []
    instance = SimpleNamespace(verbose=False)
    def smtp(*args):
        raise ConnectionError('connection refused')
    instance._send_via_smtp = smtp
    instance._send_via_gmail_api = lambda *args: sent.append('gmail')
    GmailCommunication._send_with_fallback(instance, None, [])
    assert sent == ['gmail']
