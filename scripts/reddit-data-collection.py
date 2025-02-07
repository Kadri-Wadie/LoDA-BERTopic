# Install required libraries
!pip install asyncpraw nest_asyncio tqdm pandas

# Import necessary libraries
import nest_asyncio
import asyncio
import asyncpraw
import pandas as pd
from tqdm.notebook import tqdm

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Define the async function to fetch Reddit data
async def fetch_reddit_data():
    """
    Fetches Reddit posts and comments from the 'climatechange' subreddit.
    Saves the data to CSV files while ensuring ethical considerations (e.g., anonymization).
    """
    # Initialize Reddit API client
    reddit = asyncpraw.Reddit(
        client_id='kZNzW0OkYD5p037GjTC8Lg',  # Replace with your actual client ID
        client_secret='nqwbO5hvyxd3-wHGbFjsBPDU0u6JCQ',  # Replace with your actual client secret
        user_agent='clim_change'  # Replace with a descriptive user agent
    )

    # Define the subreddit and target number of posts/comments
    subreddit = await reddit.subreddit('climatechange')  # Replace with the desired subreddit
    posts_data = []
    comments_data = []
    target_num_posts = 20  # Adjust as needed
    target_num_comments = 50  # Adjust as needed

    # Fetch posts and comments
    async for post in subreddit.hot(limit=target_num_posts):
        # Extract post metadata
        post_id = post.id
        post_title = post.title
        post_body = post.selftext
        post_author = post.author.name if post.author else 'Anonymous'  # Anonymize author
        post_date = post.created_utc
        post_upvotes = post.score
        post_downvotes = post.downs
        num_comments = post.num_comments

        # Append post data to list
        posts_data.append({
            'Post ID': post_id,
            'Title': post_title,
            'Body': post_body,
            'Author': post_author,  # Anonymized
            'Date': pd.to_datetime(post_date, unit='s'),
            'Upvotes': post_upvotes,
            'Downvotes': post_downvotes,
            'Number of Comments': num_comments,
        })

        # Fetch comments and replies
        submission = await reddit.submission(id=post_id)
        await submission.comments.replace_more(limit=None)

        for comment in submission.comments:
            if isinstance(comment, asyncpraw.models.Comment):
                comment_id = comment.id
                parent_id = comment.parent_id
                comment_body = comment.body
                comment_author = comment.author.name if comment.author else 'Anonymous'  # Anonymize author
                comment_date = comment.created_utc
                comment_upvotes = comment.score
                comment_downvotes = comment.downs

                # Append comment data to list
                comments_data.append({
                    'Post ID': post_id,
                    'Comment ID': comment_id,
                    'Parent ID': parent_id,
                    'Comment': comment_body,
                    'Author': comment_author,  # Anonymized
                    'Date': pd.to_datetime(comment_date, unit='s'),
                    'Upvotes': comment_upvotes,
                    'Downvotes': comment_downvotes
                })

                # Fetch replies to comments
                await comment.replies.replace_more(limit=None)
                for reply in comment.replies:
                    if isinstance(reply, asyncpraw.models.Comment):
                        reply_id = reply.id
                        reply_parent_id = reply.parent_id
                        reply_body = reply.body
                        reply_author = reply.author.name if reply.author else 'Anonymous'  # Anonymize author
                        reply_date = reply.created_utc
                        reply_upvotes = reply.score
                        reply_downvotes = reply.downs

                        # Append reply data to list
                        comments_data.append({
                            'Post ID': post_id,
                            'Comment ID': reply_id,
                            'Parent ID': reply_parent_id,
                            'Comment': reply_body,
                            'Author': reply_author,  # Anonymized
                            'Date': pd.to_datetime(reply_date, unit='s'),
                            'Upvotes': reply_upvotes,
                            'Downvotes': reply_downvotes
                        })

    # Convert lists to DataFrames
    posts_df = pd.DataFrame(posts_data)
    comments_df = pd.DataFrame(comments_data)

    # Save to CSV files
    posts_df.to_csv('reddit_posts.csv', index=False)
    comments_df.to_csv('reddit_comments.csv', index=False)

    # Close the Reddit client
    await reddit.close()

# Run the async function
asyncio.run(fetch_reddit_data())