# Reddit Data Collection

This subsection contains a Python script to collect Reddit posts and comments from the `climatechange` subreddit.

Steps to Prepare for Reddit API Access
1. Create a Reddit Account
If you don’t already have a Reddit account, go to Reddit and sign up. This account will be used to create a Reddit application and access the API 410.

2. Create a Reddit Application
Log in to your Reddit account and navigate to the Reddit Developer Portal.

Click on the "Create Application" or "Create Another App" button.

Fill out the form with the following details:

Name: Choose a unique name for your application (e.g., "ClimateChangeDataCollector").

App Type: Select "Script" (this is the appropriate type for local scripts or bots).

Description: Provide a brief description of your application (optional).

About URL: Leave this blank (not required for scripts).

Redirect URI: Leave this blank (not required for scripts).

Click "Create App" to complete the process.

3. Obtain client_id and client_secret
After creating the application, you will be redirected to a page summarizing your app’s details.

Locate the client_id: This is a string found under your app’s name (e.g., kZNzW0OkYD5p037GjTC8Lg).

Locate the client_secret: This is a longer string labeled as "secret" (e.g., nqwbO5hvyxd3-wHGbFjsBPDU0u6JCQ).

Save both the client_id and client_secret securely, as they are required for API authentication.

4. Define the user_agent
The user_agent is a unique identifier that helps Reddit track the source of API requests. It should follow the format:
<platform>:<app ID>:<version string> (by u/<Reddit username>)

For example:
python:ClimateChangeDataCollector:v1.0 (by u/YourUsername)
Replace YourUsername with your Reddit username.

5. Choose the Subreddit
Decide which subreddit you want to collect data from. For example, if you’re interested in climate change discussions, you might choose r/climatechange.

Ensure the subreddit name is correctly specified in your code (e.g., subreddit = await reddit.subreddit('climatechange')).

## Requirements
- Python 3.8+
- Libraries: `asyncpraw`, `nest_asyncio`, `pandas`, `tqdm`

## Setup
1. Install the required libraries:
   ```bash
   pip install asyncpraw nest_asyncio tqdm pandas
   
2. Replace the client_id, client_secret, and user_agent in the script with your Reddit API credentials.

Usage
Run the script:
python reddit_crawler.py

Output
reddit_posts.csv: Contains metadata for Reddit posts.

reddit_comments.csv: Contains metadata for comments and replies.

Ethical Considerations
Authors are anonymized to protect privacy.

Only publicly available data is collected.
