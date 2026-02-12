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


def post_tweet(text: str) -> dict:
    """
    Post a tweet with an appended call-to-action linking to the Coconala service page.

    Args:
        text: Tweet body (280 chars recommended limit)

    Returns:
        dict: Post result
            - success: bool
            - tweet_id: str (on success)
            - error: str (on failure)
    """
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

    # Append call-to-action with Coconala link if not already present
    coconala_url = "https://coconala.com/services/4072452"
    if coconala_url not in text:
        cta_suffix = f"\n\nLP made by AI: {coconala_url}"
        # Respect Twitter's 280 char limit
        if len(text) + len(cta_suffix) <= 280:
            text = text.rstrip() + cta_suffix

    try:
        client = get_client()
        response = client.create_tweet(text=text)
        tweet_id = response.data["id"]
        return {
            "success": True,
            "tweet_id": tweet_id,
            "url": f"https://twitter.com/i/web/status/{tweet_id}"
        }
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
