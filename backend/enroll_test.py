#!/usr/bin/env python3
"""
Simple helper to POST an image file to the /enroll endpoint.
Usage:
  python enroll_test.py --image "C:\path\to\alice.jpg" --name Alice

This avoids PowerShell/curl quoting issues.
"""
import argparse
import requests
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to image file")
    p.add_argument("--name", required=True, help="User name to enroll")
    p.add_argument("--url", default="http://127.0.0.1:5000/enroll", help="Enroll endpoint URL")
    args = p.parse_args()

    try:
        with open(args.image, "rb") as f:
            files = {"image": (args.image, f, "image/jpeg")}
            data = {"name": args.name}
            print(f"Posting {args.image} to {args.url} as {args.name}...")
            r = requests.post(args.url, files=files, data=data, timeout=30)
            print("Status:", r.status_code)
            try:
                print(r.json())
            except Exception:
                print(r.text)
    except FileNotFoundError:
        print("Error: image file not found:", args.image, file=sys.stderr)
        sys.exit(2)
    except requests.RequestException as e:
        print("Request failed:", e, file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
