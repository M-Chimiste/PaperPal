"""Explicit, transactional migration of untagged credential values.

Run with --format plaintext or --format xor; use --key for mixed databases.
Only key names are printed. Already migrated values are verified, not rewritten.
"""
import argparse
from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=["plaintext", "xor"], required=True)
    parser.add_argument("--key", action="append")
    parser.add_argument("--apply", action="store_true", help="Without this flag, validate only")
    args = parser.parse_args()
    load_dotenv()
    from theseus_insight.api.dependencies import CREDENTIAL_KEYS
    from theseus_insight.security import migrate_legacy, PREFIX
    from theseus_insight.db import get_connection
    keys = args.key or [key for key in CREDENTIAL_KEYS if key != "OLLAMA_URL"]
    if any(key not in CREDENTIAL_KEYS or key == "OLLAMA_URL" for key in keys):
        parser.error("Only secret credential keys can be migrated")
    with get_connection() as conn:
        with conn.cursor() as cur:
            for key in keys:
                cur.execute("SELECT value FROM settings WHERE key = %s FOR UPDATE", (key,))
                row = cur.fetchone()
                if not row or not row["value"]:
                    continue
                original = row["value"]
                replacement = migrate_legacy(original, args.format)
                if not original.startswith(PREFIX):
                    if args.apply:
                        cur.execute("UPDATE settings SET value = %s WHERE key = %s", (replacement, key))
                    print(f"{key}: {'migrated' if args.apply else 'ready to migrate'}")
            if args.apply:
                conn.commit()
            else:
                conn.rollback()


if __name__ == "__main__":
    main()
