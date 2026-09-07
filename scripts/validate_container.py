"""Isolated Docker startup/restart check. No host ports, accounts, or user data."""
import argparse
import subprocess
import time
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default='theseus-insight:validation')
    args = parser.parse_args()
    suffix = uuid.uuid4().hex[:10]
    network, database, application = [f'theseus-check-{kind}-{suffix}' for kind in ('net', 'db', 'app')]
    def docker(*parts, check=True):
        return subprocess.run(['docker', *parts], check=check, capture_output=True, text=True)
    def ready():
        code = "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3)"
        for _ in range(90):
            if docker('exec', application, 'python', '-c', code, check=False).returncode == 0:
                return
            time.sleep(1)
        logs = docker('logs', '--tail', '60', application, check=False)
        print(logs.stdout + logs.stderr)
        raise RuntimeError('Container never became ready')
    try:
        docker('network', 'create', '--internal', network)
        docker('run', '-d', '--rm', '--name', database, '--network', network,
               '--tmpfs', '/var/lib/postgresql/data', '-e', 'POSTGRES_USER=theseus',
               '-e', 'POSTGRES_PASSWORD=validation-only', '-e', 'POSTGRES_DB=theseus_validation_test',
               'pgvector/pgvector:pg14')
        for _ in range(30):
            if docker('exec', database, 'pg_isready', '-U', 'theseus', check=False).returncode == 0:
                break
            time.sleep(1)
        else:
            raise RuntimeError('Disposable PostgreSQL did not start')
        docker('run', '-d', '--rm', '--name', application, '--network', network,
               '-e', f'DATABASE_URL=postgresql://theseus:validation-only@{database}:5432/theseus_validation_test',
               '-e', 'APP_SECRET_KEY=container-validation-secret', '-e', 'APP_AUTH_TOKEN=container-validation-token',
               '-e', 'HF_HUB_OFFLINE=1', args.image)
        ready()
        dependency_check = docker('exec', application, 'python', '-m', 'pip', 'check', check=False)
        if dependency_check.returncode:
            raise RuntimeError(dependency_check.stdout + dependency_check.stderr)
        docker('exec', application, 'python', '-c',
               'import torch, torchvision, torchaudio; '
               'from docling.document_converter import DocumentConverter; '
               'from theseus_insight.theseus_insight import TheseusInsight')
        probe = """
import base64, json, urllib.request, urllib.error
base='http://127.0.0.1:8000'
try:
    urllib.request.urlopen(base+'/api/papers')
    raise AssertionError('Unauthenticated API request succeeded')
except urllib.error.HTTPError as exc:
    assert exc.code == 401
headers={'Authorization':'Basic '+base64.b64encode(b'theseus:container-validation-token').decode(), 'Content-Type':'application/json'}
request=urllib.request.Request(base+'/api/papers', headers=headers)
assert urllib.request.urlopen(request).status == 200
request=urllib.request.Request(base+'/api/settings/credentials', data=json.dumps({'OPENAI_API_KEY':'validation-only-key'}).encode(), headers=headers, method='PUT')
assert urllib.request.urlopen(request).status == 200
"""
        docker('exec', application, 'python', '-c', probe)
        docker('restart', application)
        ready()
        verify = """
import base64, json, urllib.request
headers={'Authorization':'Basic '+base64.b64encode(b'theseus:container-validation-token').decode()}
request=urllib.request.Request('http://127.0.0.1:8000/api/settings/credentials', headers=headers)
data=json.load(urllib.request.urlopen(request))
assert data['OPENAI_API_KEY'] == {'configured': True, 'value': ''}
"""
        docker('exec', application, 'python', '-c', verify)
        print('Container dependency, migration, readiness, authentication, credential persistence and restart checks passed.')
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr or exc.stdout) from exc
    finally:
        docker('rm', '-f', application, database, check=False)
        docker('network', 'rm', network, check=False)


if __name__ == '__main__':
    main()
