#!/usr/bin/env python3
"""Twitter API連携モジュール - tweepy使用
ツイート投稿、自動ツイート生成、スケジューラーを管理する。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from config import GOALS_PATH, STATE_PATH, log

# .envファイルを読み込み
load_dotenv(Path(__file__).parent / ".env")

# Twitter API認証情報
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY", "")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET", "")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN", "")
TWITTER_ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET", "")

TWEET_INTERVAL = 6 * 3600  # 6時間ごと
MAX_TWEET_HISTORY = 20  # 保存する履歴数


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


def get_tweet_history() -> list[str]:
    """state.jsonからtweet_historyのテキスト一覧を取得する。"""
    try:
        if not STATE_PATH.exists():
            return []
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        history = state.get("tweet_history", [])
        return [h.get("text", "") for h in history if isinstance(h, dict)]
    except Exception as e:
        log.warning(f"tweet_history読み込み失敗: {e}")
        return []


def add_to_tweet_history(text: str):
    """state.jsonのtweet_historyに投稿を追加（最大MAX_TWEET_HISTORY件）。"""
    try:
        state = {}
        if STATE_PATH.exists():
            state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        history = state.get("tweet_history", [])
        history.append({
            "text": text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        # 直近MAX_TWEET_HISTORY件のみ保持
        state["tweet_history"] = history[-MAX_TWEET_HISTORY:]
        STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        log.warning(f"tweet_history保存失敗: {e}")


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

    # Check for duplicate tweets
    tweet_history = get_tweet_history()
    if text in tweet_history:
        return {"success": False, "error": "Duplicate tweet detected. This tweet has already been posted."}

    # Append call-to-action with Coconala link if not already present and if it seems like a service offering
    coconala_url = "https://coconala.com/services/4072452"
    cta_suffix = f"\n\nLP made by AI: {coconala_url}"

    service_keywords = ["service", "offer", "consulting", "design", "development", "writing", "translation", "support", "coaching", "tutoring", "freelance"]
    is_service_offering = any(re.search(r'\b' + keyword + r'\b', text, re.IGNORECASE) for keyword in service_keywords)

    if is_service_offering and coconala_url not in text:
        if len(text) + len(cta_suffix) > 280 and (not media or len(text) <= 280 - len(cta_suffix)):
            pass
        else:
            text = text.rstrip() + cta_suffix

    try:
        client = get_client()

        media_ids = None
        if media:
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
                add_to_tweet_history(text)
                return {
                    "success": True,
                    "tweet_id": tweet_id,
                    "url": f"https://twitter.com/i/web/status/{tweet_id}"
                }
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in [
                    "unauthorized", "forbidden", "authentication",
                    "invalid", "401", "403", "not found", "404"
                ]):
                    log.error("Non-transient Twitter API error, not retrying: %s", e)
                    return {"success": False, "error": f"Tweet post error: {e}"}

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


async def generate_tweet(client) -> str:
    """Geminiにツイート文を生成させる。goals.mdのX Strategyを読んでトーンや内容を決める。

    Args:
        client: httpx.AsyncClient (未使用だが将来のAPI呼び出し用)

    Returns:
        生成されたツイートテキスト。失敗時は空文字列。
    """
    from brain import think

    goals_text = ""
    try:
        if GOALS_PATH.exists():
            goals_text = GOALS_PATH.read_text(encoding="utf-8")
    except Exception as e:
        log.warning(f"generate_tweet: goals.md読み込み失敗: {e}")

    # 既存ツイート履歴を取得して重複を避ける
    history = get_tweet_history()
    history_text = "\n".join(history[-5:]) if history else "(なし)"

    prompt = (
        "あなたはGod AI（@GODAI_Beny）。以下の戦略に従ってX（Twitter）投稿文を1つ生成しろ。\n\n"
        f"{goals_text}\n\n"
        f"【直近の投稿（重複禁止）】\n{history_text}\n\n"
        "【ルール】\n"
        "- 140文字以内の日本語ツイート1つだけ出力\n"
        "- ハッシュタグは1-2個まで\n"
        "- 実用的で具体的な内容\n"
        "- 押し売りしない自然なトーン\n"
        "- AI活用Tips、LP制作の価値、ビジネス効率化のいずれか\n"
        "- 直近の投稿と似た内容は避ける\n"
        "- ツイート本文のみ出力。説明や前置き不要\n"
    )

    try:
        tweet_text, brain_name = await think(prompt, heavy=False)
        tweet_text = tweet_text.strip().strip('"').strip("'")
        if not tweet_text or len(tweet_text) > 280:
            log.warning(f"generate_tweet: 生成テキスト不正 (len={len(tweet_text) if tweet_text else 0})")
            return ""
        return tweet_text
    except Exception as e:
        log.error(f"generate_tweet エラー: {e}", exc_info=True)
        return ""


async def auto_tweet(client) -> bool:
    """Tweet generation -> duplicate check -> posting -> journal recording.
    Includes a check for minimum engagement metrics before posting.

    Args:
        client: httpx.AsyncClient (for Telegram sending)

    Returns:
        True if posting was successful.
    """
    from god import tg_send
    from memory import load_conversations, save_conversations, append_journal

    if not is_configured():
        log.info("auto_tweet: Twitter not configured, skipping.")
        return False

    # Fetch engagement metrics and check threshold
    engagement_threshold = 5  # Configurable threshold: minimum average likes+retweets+replies
    tweet_history_with_metrics = get_tweet_history_with_metrics() # Assumes this function returns tweets with engagement data
    
    if tweet_history_with_metrics:
        recent_tweets = tweet_history_with_metrics[-10:] # Look at the last 10 tweets for average engagement
        if recent_tweets:
            total_engagement = sum(t.get('likes', 0) + t.get('retweets', 0) + t.get('replies', 0) for t in recent_tweets)
            average_engagement = total_engagement / len(recent_tweets)

            if average_engagement < engagement_threshold:
                log.info(f"auto_tweet: Average engagement ({average_engagement:.2f}) is below threshold ({engagement_threshold}). Skipping tweet.")
                return False
        else:
            log.info("auto_tweet: No recent tweet metrics available, proceeding with tweet generation.")
    else:
        log.info("auto_tweet: No tweet history with metrics, proceeding with tweet generation.")

    tweet_text = await generate_tweet(client)
    if not tweet_text or not tweet_text.strip():
        log.warning("auto_tweet: Generated tweet text is empty or whitespace, skipping.")
        return False

    # Duplicate check (already checked in generate_tweet, but final confirmation)
    tweet_history_texts = get_tweet_history()
    if tweet_text in tweet_history_texts:
        log.warning("auto_tweet: Generated tweet is a duplicate, skipping.")
        return False

    # Retry mechanism for posting tweets
    max_retries = 3
    last_error = None
    for attempt in range(max_retries):
        result = post_tweet(tweet_text)
        if result["success"]:
            log.info(f"auto_tweet: Successfully posted {result['url']}")
            await tg_send(client, f"[Auto-tweet Posted]\n{tweet_text}\n{result['url']}")
            try:
                conversations = load_conversations()
                conversations.append({
                    "time": datetime.now(timezone.utc).isoformat(),
                    "from": "system",
                    "text": f"Auto-tweet: {tweet_text[:100]}"
                })
                save_conversations(conversations)
            except Exception as e:
                log.warning(f"auto_tweet: Failed to save conversation history: {e}")
            return True
        else:
            last_error = result["error"]
            log.warning(f"auto_tweet: Post attempt {attempt+1}/{max_retries} failed: {last_error}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                log.info(f"auto_tweet: Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)

    log.error(f"auto_tweet: Posting failed after {max_retries} attempts. Last error: {last_error}")
    return False


# Helper function to get tweet history with engagement metrics.
# This function needs to be implemented to fetch and store engagement data.
# For now, it returns an empty list if not implemented.
def get_tweet_history_with_metrics() -> list[dict]:
    """
    Retrieves tweet history including engagement metrics (likes, retweets, replies).
    This is a placeholder and needs to be implemented to fetch actual metrics.
    Example return: [{"text": "...", "likes": 10, "retweets": 5, "replies": 2, "timestamp": "..."}, ...]
    """
    try:
        if not STATE_PATH.exists():
            return []
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        history = state.get("tweet_history_with_metrics", [])
        return history if isinstance(history, list) else []
    except Exception as e:
        log.warning(f"tweet_history_with_metrics読み込み失敗: {e}")
        return []

# NOTE: The `add_to_tweet_history` function should ideally be modified to also store engagement metrics
# when a tweet is successfully posted. This would involve fetching these metrics from the Twitter API
# after posting and updating the state. For this specific improvement, we are focusing on the
# `auto_tweet` function and assuming `get_tweet_history_with_metrics` can provide the data.


async def tweet_scheduler_loop(client):
    """Initial immediate tweet posting, then a scheduler that calls auto_tweet() at 6-hour intervals.

    Args:
        client: httpx.AsyncClient (for Telegram sending)
    """
    log.info(f"tweet_scheduler_loop started (interval: {TWEET_INTERVAL} seconds)")

    # Initial immediate tweet posting
    log.info("tweet_scheduler_loop: Posting the first tweet immediately.")
    try:
        await auto_tweet(client)
    except Exception as e:
        log.error(f"tweet_scheduler_loop: Error during initial tweet post: {e}", exc_info=True)

    # Periodic posting loop
    while True:
        log.info(f"tweet_scheduler_loop: Sleeping for {TWEET_INTERVAL} seconds until next tweet.")
        await asyncio.sleep(TWEET_INTERVAL)
        try:
            # Generate tweet and check for duplicates before posting
            tweet_text = await generate_tweet(client)
            if not tweet_text or not tweet_text.strip():
                log.warning("tweet_scheduler_loop: tweet generation failed, skipping.")
                continue

            tweet_history = get_tweet_history()
            if tweet_text in tweet_history:
                log.warning("tweet_scheduler_loop: Generated tweet is a duplicate, skipping.")
                continue

            result = post_tweet(tweet_text)
            if result["success"]:
                log.info(f"tweet_scheduler_loop: Tweet posted successfully {result['url']}")
                await tg_send(client, f"[Auto-tweet Posted]\n{tweet_text}\n{result['url']}")
                try:
                    conversations = load_conversations()
                    conversations.append({
                        "time": datetime.now(timezone.utc).isoformat(),
                        "from": "system",
                        "text": f"Auto-tweet: {tweet_text[:100]}"
                    })
                    save_conversations(conversations)
                except Exception as e:
                    log.warning(f"tweet_scheduler_loop: Failed to save conversation history: {e}")
            else:
                log.error(f"tweet_scheduler_loop: Tweet posting failed {result['error']}")
        except Exception as e:
            log.error(f"tweet_scheduler_loop: An error occurred: {e}", exc_info=True)


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
