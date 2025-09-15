import os
import json
import re
import pytz
import csv
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv
from atproto import Client
import sys

# Ensure proper encoding for stdout
sys.stdout.reconfigure(encoding='utf-8')

# Load Credentials
load_dotenv("dotenv_path")
username, password = os.getenv("bsky_username"), os.getenv("bsky_password")
if not username or not password:
    raise ValueError("Please set the bsky_username and bsky_password environment variables.")

# Connect to Bluesky API
client = Client()
client.login(username, password)
target_user_id = 'did:plc:psv3dr2zp7plgqa6jtaz6nzy' #DID of the tracked account - found under https://atproto-browser.vercel.app/


# Determine Post Type (Original, Reply, Quote, Repost)
def determine_post_type(post, target_user_id):
    try:
        author_did = post.author.did
        is_self_authored = (author_did == target_user_id)
        record = post.record
        embed = getattr(record, "embed", None)

        if not is_self_authored:
            return "Repost"
        if getattr(record, "reply", None):
            return "Reply"
        if embed:
            quoted_record = getattr(embed, 'record', None)
            if quoted_record and hasattr(quoted_record, 'uri'):
                return "Quote"
        return "Original"

    except AttributeError as e:
        print(f"AttributeError: {e} - Post structure may be incomplete.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None


#  Check for Media in Post
def has_media(post):
    def safeget(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    try:
        record = safeget(post, "record")
        if not record:
            return (False, None)
        embed = safeget(record, "embed")
        if not embed:
            return (False, None)
        if safeget(embed, "images"):
            return (True, "images")
        if safeget(embed, "video"):
            return (True, "video")
        ext = safeget(embed, "external")
        if ext and safeget(ext, "thumb"):
            return (True, "external_thumbnail")
        media = safeget(embed, "media")
        if media:
            if safeget(media, "images"):
                return (True, "images")
            mediaType = safeget(media, "mediaType")
            if mediaType and mediaType.startswith("image/"):
                return (True, "image_mediaType")
        t = safeget(embed, "$type") or safeget(embed, "_type")
        if t:
            t_lower = t.lower()
            if "image" in t_lower:
                return (True, "image_type")
            if "video" in t_lower:
                return (True, "video_type")
    except Exception as e:
        print(f"Error checking for media: {e}")
    return (False, None)


# Extract URLs and Hashtags from Post Text
def extract_urls(text):
    url_pattern = re.compile(r'(?:http://|https://|www\.)\S+|botfrens.com\S+|bsky\.app/profile/\w+|m.youtube.com/watch?\S+')
    return url_pattern.findall(text)

def extract_hashtags(text):
    return [f"#{tag}" for tag in re.findall(r'#(\w+)', text)]


# Mention Detection
class Post:
    def __init__(self, content):
        self.content = content
        self.has_mentions = self.check_mentions()
        self.mention_count = self.count_mentions()

    def check_mentions(self):
        return bool(re.search(r'(@\w+)', self.content))

    def count_mentions(self):
        return len(re.findall(r'(@\w+)', self.content))


# Count Items with Pagination (likes, reposts, etc.) so it doesn't break on large datasets
def count_all_items(fetch_fn, params, item_key):
    total = 0
    cursor = None

    while True:
        if cursor:
            params['cursor'] = cursor
        response = fetch_fn(params)
        items = getattr(response, item_key, [])
        total += len(items)
        cursor = getattr(response, 'cursor', None)
        if not cursor:
            break

    return total


#Anonymise the post uri to remove user information (did:plc:...)
def anonymise_uri(uri):
    # Use regular expression to extract the last part of the URI after the last '/'
    match = re.search(r'\/([^\/]+)$', uri)
    if match:
        return match.group(1)  # This will return the post ID
    return None 


#Split time of post creation into weekday and time of day ("morning", "afternoon", "evening", "night") for later analysis (are afternoon posts more popular? etc.)
def bucket_time(iso_timestamp):
    try:
        dt = datetime.fromisoformat(iso_timestamp)
        # Convert to local time if necessary (e.g., 'America/New_York', 'UTC')
        local_tz = pytz.timezone('UTC')  # Change timezone as needed
        dt = dt.astimezone(local_tz)
        
        hour = dt.hour
        weekday = dt.strftime("%A")  # Get the weekday as a full name (e.g., "Monday")
        
        # Determine the time of day
        if 6 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 18:
            time_of_day = "afternoon"
        elif 18 <= hour < 24:
            time_of_day = "evening"
        else:
            time_of_day = "night"
        
        # Combine weekday and time of day for anonymization
        return f"{weekday}_{time_of_day}"
    except Exception as e:
        print(f"Error in bucket_time: {e}")
        return "unknown_time"


# Functions to anonymise Mentions and URLs
def anonymise_mentions_and_urls(text):
    # Replace mentions with @mention
    text = re.sub(r'@\S+', '@mention', text)
    
    # Define the URL patterns for classification
    url_pattern = re.compile(r'(?:http://|https://|www\.)\S+|botfrens.com\S+|bsky\.app/profile/\w+|m.youtube.com/watch?\S+')
    
    def replace_url(url):
        # Check if it's a self link (museum's website or shop)
        if 'botfrens.com' in url:
            return 'self_link'
        # Check if it's a platform link (Bluesky or similar)
        elif 'bsky.app/profile/' in url:
            return 'platform_link'
        # Otherwise, treat it as an external link
        else:
            return 'external_link'
    
    # Replace all URLs with appropriate placeholder
    text = re.sub(url_pattern, lambda match: replace_url(match.group(0)), text)

    return text


# Get engagement_data
def get_engagement_data(client, target_user_id, posts_back=100):
    execution_timestamp = datetime.now().isoformat()
    posts_with_media = 0
    posts_with_urls = 0
    posts_with_hashtags = 0

    try:
        user_profile = client.app.bsky.actor.get_profile({"actor": target_user_id})
        profile_data = {
            "execution_timestamp": execution_timestamp,
            "follower_count": user_profile.followers_count,
            "following_count": user_profile.follows_count,
            "posts_count": getattr(user_profile, 'posts_count', 0),
            "display_name": user_profile.display_name,
            "description": user_profile.description,
        }

        # Fetch exactly 100 posts per query
        response = client.app.bsky.feed.get_author_feed({
            "actor": target_user_id,
            "limit": 100
        })
        user_posts = response.feed  # Get exactly 100 posts per query

        post_counts, hashtag_counts = defaultdict(int), defaultdict(int)
        total_likes, total_reposts, total_replies = 0, 0, 0
        posts_data = []

        for post_data in user_posts:
            post = post_data.post
            post_text = getattr(post.record, 'text', "").replace('\n', ' ').replace('\r', '')

            # Anonymise mentions and URLs in the post text
            post_text_anonymised = anonymise_mentions_and_urls(post_text)
            
            post_type = determine_post_type(post, target_user_id)
            post_counts[post_type] += 1

            hashtags = extract_hashtags(post_text)
            for tag in hashtags:
                hashtag_counts[tag] += 1
                hashtag_counts[tag.lower()] += 1

            urls = extract_urls(post_text)
            post_instance = Post(post_text)
            has_media_flag, media_details = has_media(post)

            if has_media_flag:
                posts_with_media += 1
            if urls:
                posts_with_urls += 1
            if hashtags:
                posts_with_hashtags += 1

            likes = count_all_items(client.app.bsky.feed.get_likes, {'uri': post.uri, 'cid': post.cid, 'limit': 100}, 'likes')
            reposts = count_all_items(client.app.bsky.feed.get_reposted_by, {'uri': post.uri, 'cid': post.cid, 'limit': 100}, 'reposted_by')

            try:
                thread = client.app.bsky.feed.get_post_thread({'uri': post.uri, 'depth': 1})
                replies = len(thread.thread.replies or [])
            except:
                replies = 0

            engagement_score = likes + reposts + replies
            # Engagement per 1,000 followers calculation with increased precision
            engagement_per_1k = round(engagement_score / max(profile_data["follower_count"], 1) * 1000, 4)

            total_likes += likes
            total_reposts += reposts
            total_replies += replies

            # Anonymise URLs by replacing them with categories like 'self_link', 'platform_link', 'external_link'
            anonymised_urls = [anonymise_mentions_and_urls(url) for url in urls]

            post_dict = {
                "execution_timestamp": execution_timestamp,
                "created_at": post.record.created_at,
                "time_bucket": bucket_time(post.record.created_at),
                "cid": str(post.cid),
                "uri": anonymise_uri(post.uri),
                "text": post_text_anonymised,
                "post_type": post_type,
                "hashtags": ", ".join(hashtags),
                "hashtags_normalised": ", ".join([tag.lower() for tag in hashtags]),
                "urls": ", ".join(anonymised_urls), 
                "media": media_details,
                "has_mentions": post_instance.has_mentions,
                "mention_count": post_instance.mention_count,
                "likes_count": likes,
                "reposts_count": reposts,
                "replies_count": replies,
                "engagement_score": engagement_score,
                "engagement_per_1k_followers": engagement_per_1k
            }

            posts_data.append(post_dict)

        return {
            "execution_timestamp": execution_timestamp,
            "user_profile": profile_data,
            "posts": posts_data,
            "total_posts": len(posts_data),
            "post_type_distribution": dict(post_counts),
            "hashtag_counts": dict(sorted(hashtag_counts.items(), key=lambda x: x[1], reverse=True)),
            "total_likes": total_likes,
            "total_reposts": total_reposts,
            "total_replies": total_replies,
            "avg_engagement_per_post": (total_likes + total_reposts + total_replies) / len(posts_data) if posts_data else 0,
            "posts_with_media": posts_with_media,
            "posts_with_urls": posts_with_urls,
            "posts_with_hashtags": posts_with_hashtags,
            "top_hashtags": dict(list(hashtag_counts.items())[:5])
        }

    except Exception as e:
        return {"error": str(e), "execution_timestamp": execution_timestamp}


# Save to JSON and CSV
if __name__ == "__main__":
    results = get_engagement_data(client, target_user_id)
    existing_data = []

    # Save to JSON
    try:
        with open('rijksmuseumart.json', 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            if not isinstance(existing_data, list):
                existing_data = []
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []

    existing_data.append(results)

    with open('rijksmuseumart.json', 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)

    print("Data saved to 'rijksmuseumart.json'")

    # Append to CSV
    csv_account_file = 'rijksmuseumart_account.csv'
    csv_posts_file = 'rijksmuseumart_posts.csv'

    account_headers = ["execution_timestamp", "follower_count", "following_count", "posts_count", "display_name", "description"]
    posts_headers = ["execution_timestamp", "created_at", "time_bucket", "cid", "uri", "text", "post_type", "hashtags", "hashtags_normalised", "urls", "media", "has_mentions", "mention_count", "likes_count", "reposts_count", "replies_count", "engagement_score", "engagement_per_1k_followers"]

    try:
        # Clean description field to remove line breaks
        user_profile_cleaned = results['user_profile'].copy()
        if 'description' in user_profile_cleaned and isinstance(user_profile_cleaned['description'], str):
            user_profile_cleaned['description'] = user_profile_cleaned['description'].replace('\n', ' ').replace('\r', ' ').strip()

        # Write Account Data
        account_file_exists = os.path.isfile(csv_account_file)
        with open(csv_account_file, 'a', newline='', encoding='utf-8') as account_csv:
            account_writer = csv.DictWriter(account_csv, fieldnames=account_headers, quoting=csv.QUOTE_ALL)
            if not account_file_exists:
                account_writer.writeheader()
            account_writer.writerow(user_profile_cleaned)
        print("Data appended to 'rijksmuseumart_account.csv'")

        # Write Posts Data
        posts_file_exists = os.path.isfile(csv_posts_file)
        with open(csv_posts_file, 'a', newline='', encoding='utf-8') as posts_csv:
            posts_writer = csv.DictWriter(posts_csv, fieldnames=posts_headers, quoting=csv.QUOTE_ALL)
            if not posts_file_exists:
                posts_writer.writeheader()
            for post in results['posts']:
                posts_writer.writerow(post)
        print("Data appended to 'rijksmuseumart_posts.csv'")

    except Exception as e:
        print(f"Failed to write CSV files: {e.__class__.__name__}: {e}")


"""
License for use of this code as foundo on the AT Protocol SDK website (https://atproto.blue/en/latest/licence.html):
MIT License
Copyright (c) 2024 Ilya
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""