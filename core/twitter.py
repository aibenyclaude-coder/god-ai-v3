#!/usr/bin/env python3
"""Twitter API連携モジュール - tweepy使用"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# .envファイルを読み込み
load_dotenv(Path(__file__).parent / ".env")

# Twitter API認証情報
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY", "")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET", "")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN", "")
TWITTER_ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET", "")


def is_configured() -> bool:
    """Twitter APIキーが設定済みか確認"""
    return all([
        TWITTER_API_KEY,
        TWITTER_API_SECRET,
        TWITTER_ACCESS_TOKEN,
        TWITTER_ACCESS_TOKEN_SECRET
    ])


def get_client():
    """Twitter API v2クライアントを取得"""
    if not is_configured():
        raise ValueError("Twitter API credentials not configured")

    try:
        import tweepy
    except ImportError:
        raise ImportError("tweepy is not installed. Run: pip install tweepy")

    client = tweepy.Client(
        consumer_key=TWITTER_API_KEY,
        consumer_secret=TWITTER_API_SECRET,
        access_token=TWITTER_ACCESS_TOKEN,
        access_token_secret=TWITTER_ACCESS_TOKEN_SECRET
    )
    return client


def post_tweet(text: str, media: list[str] | None = None) -> dict:
    """
    Post a tweet with an appended call-to-action linking to the Coconala service page.
    Optionally upload media (images).

    Args:
        text: Tweet body (280 chars recommended limit)
        media: A list of file paths to images to be uploaded with the tweet.

    Returns:
        dict: Post result
            - success: bool
            - tweet_id: str (on success)
            - url: str (on success)
            - error: str (on failure)
    """
    import time
    import logging
    import re  # Import the regular expression module

    log = logging.getLogger("god.twitter")

    if not text or not text.strip():
        return {"success": False, "error": "Tweet body is empty"}

    if not is_configured():
        return {
            "success": False,
            "error": "Twitter API not configured. Set the following in .env:\n"
                     "TWITTER_API_KEY\n"
                     "TWITTER_API_SECRET\n"
                     "TWITTER_ACCESS_TOKEN\n"
                     "TWITTER_ACCESS_TOKEN_SECRET"
        }

    # Append call-to-action with Coconala link if not already present and if it seems like a service offering
    coconala_url = "https://coconala.com/services/4072452"
    cta_suffix = f"\n\nLP made by AI: {coconala_url}"

    # Define keywords that suggest a service offering
    service_keywords = ["service", "offer", "consulting", "design", "development", "writing", "translation", "support", "coaching", "tutoring", "freelance"]

    # Check if the text contains any service keywords (case-insensitive)
    is_service_offering = any(re.search(r'\b' + keyword + r'\b', text, re.IGNORECASE) for keyword in service_keywords)

    # Append CTA only if it's a service offering and the URL is not already in the text
    if is_service_offering and coconala_url not in text:
        # Check if adding CTA exceeds the limit only if no media is present or if it's short
        if len(text) + len(cta_suffix) > 280 and (not media or len(text) <= 280 - len(cta_suffix)):
            # If text is already long, and we have media, we might not be able to add CTA.
            # Prioritize posting the content and media.
            pass
        else:
            text = text.rstrip() + cta_suffix

    try:
        client = get_client()

        # Handle media upload if provided
        media_ids = None
        if media:
            # Ensure media is a list of file paths and each file exists
            valid_media_paths = []
            for media_path in media:
                if Path(media_path).is_file():
                    valid_media_paths.append(media_path)
                else:
                    log.warning("Media file not found, skipping: %s", media_path)

            if valid_media_paths:
                media_upload_response = client.upload_media(valid_media_paths, media_category="tweet_image")
                media_ids = media_upload_response.data["media_id"]
            else:
                log.warning("No valid media files provided for upload.")


        # Retry with exponential backoff for transient errors
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try:
                response = client.create_tweet(text=text, media_ids=media_ids)
                tweet_id = response.data["id"]
                return {
                    "success": True,
                    "tweet_id": tweet_id,
                    "url": f"https://twitter.com/i/web/status/{tweet_id}"
                }
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                # Non-transient errors: fail immediately without retry
                if any(keyword in error_str for keyword in [
                    "unauthorized", "forbidden", "authentication",
                    "invalid", "401", "403", "not found", "404"
                ]):
                    log.error("Non-transient Twitter API error, not retrying: %s", e)
                    return {"success": False, "error": f"Tweet post error: {e}"}

                # Transient errors (rate limit, server error): retry with backoff
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    log.warning(
                        "Transient Twitter API error (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1, max_retries, wait_time, e
                    )
                    time.sleep(wait_time)
                else:
                    log.error(
                        "Twitter API error after %d attempts: %s",
                        max_retries, e
                    )

        return {"success": False, "error": f"Tweet post error after {max_retries} attempts: {last_error}"}
    except ImportError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"Tweet post error: {e}"}


def get_setup_instructions() -> str:
    """Twitter API設定手順を返す"""
    return """Twitter API設定手順:

1. Twitter Developer Portal (https://developer.twitter.com/) にアクセス
2. アプリを作成し、API Keys を取得
3. core/.env に以下を追加:

TWITTER_API_KEY=your_api_key
TWITTER_API_SECRET=your_api_secret
TWITTER_ACCESS_TOKEN=your_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret

4. tweepy をインストール: pip install tweepy
"""


# テスト用
if __name__ == "__main__":
    print("Twitter API Configuration Check")
    print("-" * 40)
    if is_configured():
        print("Status: Configured")
        print("Ready to post tweets!")
    else:
        print("Status: Not configured")
        print(get_setup_instructions())
