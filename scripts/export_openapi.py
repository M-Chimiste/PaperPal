"""Export the API contract without starting workers or requiring a database."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    from theseus_insight.main import app
    path = Path('theseus-ui/openapi.json')
    schema = app.openapi()
    if args.check:
        if json.loads(path.read_text()) != schema:
            raise SystemExit('API contract drift: run make generate-api and review the diff')
    else:
        path.write_text(json.dumps(schema, indent=2, sort_keys=True)+'\n')


if __name__ == '__main__':
    main()
