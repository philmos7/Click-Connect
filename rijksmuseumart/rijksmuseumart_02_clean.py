import pandas as pd
from atproto import Client
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv("dotenv_path")
username = os.getenv("bsky_username")
password = os.getenv("bsky_password")
target_did = 'did:plc:psv3dr2zp7plgqa6jtaz6nzy'

# Login to Bluesky API
client = Client()
client.login(username, password)

# Utility functions
def build_uri(post_id: str, did: str) -> str:
    return f"at://{did}/app.bsky.feed.post/{post_id}"

def extract_post_id(uri: str) -> str:
    return uri.split("/")[-1] if uri else None

def detect_reply_type(uri_post_id: str):
    uri = build_uri(uri_post_id, target_did)
    try:
        thread = client.app.bsky.feed.get_post_thread({'uri': uri, 'depth': 10})
        thread_view = thread.thread

        if not thread_view.parent:
            return pd.Series(["Not a reply", None, None])

        parent = thread_view.parent.post
        root = thread_view
        while root.parent:
            root = root.parent
        root = root.post

        if parent.author.did == target_did:
            reply_type = "Reply to self"
        elif root.author.did == target_did:
            reply_type = "Reply in own thread"
        else:
            reply_type = "Reply to other"

        return pd.Series([reply_type, extract_post_id(parent.uri), extract_post_id(root.uri)])

    except Exception as e:
        print(f"[Error] {uri}: {e}")
        return pd.Series([None, None, None])

# Load post data
posts_df = pd.read_csv("/path/to/data/rijksmuseumart_posts.csv")

# Convert 'created_at' to datetime and drop invalid entries
posts_df['created_at'] = pd.to_datetime(posts_df['created_at'], errors='coerce')
posts_df = posts_df.dropna(subset=['created_at'])

# Sort so the latest post per URI is last
posts_df = posts_df.sort_values(by='created_at')

# Step 1: Keep only the latest post per URI
latest_df = posts_df.groupby('uri', as_index=False).last()

# Step 2: Save top and worst 10 by engagement (excluding reposts)
non_reposts_df = latest_df[latest_df['post_type'] != 'Repost']

top10_df = non_reposts_df.sort_values(by='engagement_score', ascending=False).head(10)
top10_df.to_csv("/path/to/data/rijksmuseumart_top10.csv", index=False)

worst10_df = non_reposts_df.sort_values(by='engagement_score', ascending=True).head(10)
worst10_df.to_csv("/path/to/data/rijksmuseumart_worst10.csv", index=False)

# Step 3: Engagement growth for URIs with multiple entries
duplicated_uris = posts_df['uri'].value_counts()[lambda x: x > 1].index
growth_df = posts_df[posts_df['uri'].isin(duplicated_uris)]

growth_records = []

for uri, group in growth_df.groupby('uri'):
    group_sorted = group.sort_values(by='created_at')
    first = group_sorted.iloc[0]
    last = group_sorted.iloc[-1]

    record = {
        "uri": uri,
        "first_timestamp": first['created_at'],
        "last_timestamp": last['created_at'],
        "likes_growth_pct": ((last['likes_count'] - first['likes_count']) / first['likes_count'] * 100) if first['likes_count'] > 0 else None,
        "reposts_growth_pct": ((last['reposts_count'] - first['reposts_count']) / first['reposts_count'] * 100) if first['reposts_count'] > 0 else None,
        "replies_growth_pct": ((last['replies_count'] - first['replies_count']) / first['replies_count'] * 100) if first['replies_count'] > 0 else None,
        "engagement_score_growth_pct": ((last['engagement_score'] - first['engagement_score']) / first['engagement_score'] * 100) if first['engagement_score'] > 0 else None
    }

    growth_records.append(record)

engagement_growth_df = pd.DataFrame(growth_records)
engagement_growth_df.to_csv("/path/to/data/rijksmuseumart_engagement_growth.csv", index=False)

# Step 4: Enrich latest posts with reply type info
latest_df[['reply_type', 'parent_post_id', 'root_post_id']] = latest_df.apply(
    lambda row: detect_reply_type(extract_post_id(row['uri'])) if row['post_type'] == 'Reply' else pd.Series([None, None, None]),
    axis=1
)

# Final Output: Save enriched latest dataset
latest_df.to_csv("/path/to/data/rijksmuseumart_posts_latest_enriched.csv", index=False)

# Summary Output
print(f"Original post count: {len(posts_df)}")
print(f"Unique URIs after deduplication: {len(latest_df)}")
print(f"URIs with multiple entries (for growth): {len(duplicated_uris)}")
print("Saved files:\n - rijksmuseumart_posts_latest_enriched.csv\n - rijksmuseumart_top10.csv\n - rijksmuseumart_worst10.csv\n - rijksmuseumart_engagement_growth.csv")
