import praw
import os
import time
import json
from typing import List, Dict
from datetime import datetime
import streamlit as st
from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from .llm_service import llm_service

class RedditService:
    """Service for interacting with Reddit API using PRAW"""
    
    def __init__(self):
        """Initialize Reddit client with credentials"""
        try:
            # Try to get credentials from Streamlit secrets first, then environment variables
            client_id = None
            client_secret = None
            user_agent = None
            
            try:
                if hasattr(st, 'secrets'):
                    reddit_secrets = st.secrets.get('reddit', {})
                    if reddit_secrets:
                        client_id = reddit_secrets.get('client_id')
                        client_secret = reddit_secrets.get('client_secret')
                        user_agent = reddit_secrets.get('user_agent')
            except Exception:
                # If secrets access fails, continue to environment variables
                pass
            
            # Fallback to environment variables
            client_id = client_id or os.getenv('REDDIT_CLIENT_ID')
            client_secret = client_secret or os.getenv('REDDIT_CLIENT_SECRET')
            user_agent = user_agent or os.getenv('REDDIT_USER_AGENT', 'Reddrop:v1.0')
            
            if not all([client_id, client_secret]):
                raise ValueError("Reddit API credentials not found. Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET.")
            
            # Include username and password for user-level authentication
            username = os.getenv('REDDIT_USERNAME')
            password = os.getenv('REDDIT_PASSWORD')

            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
                username=username,
                password=password
            )

            # Test the connection
            self.reddit.user.me()
            
        except Exception as e:
            st.warning("Reddit API not configured. Using demo mode.")
            self.reddit = None


    def discover_relevant_threads(self, content: str, status_callback=None, time_filter: str = 'week', subreddit_limit: int = 30, threads_limit: int = 20) -> List[Dict]:
        """
        Discover relevant Reddit threads using the simplified process:
        1. Analyze content using LLM to extract topics and other aspects.
        2. Extract subreddits using the output of content analysis.
        3. Generate search queries using LLM.
        4. Run the queries against the subreddits found.
        5. Collect basic scores for each thread.
        6. Display the basic scores in a table format.

        Args:
            content: User's input topic or paragraph
            status_callback: Optional function to call for status updates

        Returns:
            List of relevant threads with basic scores in table format
        """
        try:
            # Step 1: Analyze content using LLM
            if status_callback:
                status_callback("Step 1: Analyzing content...")

            content_analysis = self._analyze_content(content)

            # Step 2: Extract subreddits using content analysis
            if status_callback:
                status_callback("Step 2: Extracting subreddits...")

            discovered_subreddits = self._find_relevant_subreddits(content_analysis, limit=subreddit_limit)

            if not discovered_subreddits:
                st.warning("Could not find relevant subreddits. Using demo mode.")
                return 

            subreddit_names = [sub['name'] for sub in discovered_subreddits]

            # Step 3: Generate search queries using LLM
            if status_callback:
                status_callback("Step 3: Generating search queries...")

            search_queries = self._generate_search_queries(content_analysis)

            # Step 4: Run queries against subreddits
            if status_callback:
                status_callback(f"Step 4: Searching threads in {len(subreddit_names)} subreddits...")

            candidate_threads = self._search_threads_directly(
                subreddits_to_search=subreddit_names,
                search_queries=search_queries,
                content_analysis=content_analysis,
                status_callback=status_callback,
                time_filter=time_filter,
                threads_limit=threads_limit
            )

            if not candidate_threads:
                return []
            

            # Step 5: Evaluate threads and subreddits for alignment with content
            if status_callback:
                status_callback("Step 5: Evaluating threads and subreddits for alignment...")

            candidate_threads = self._llm_filter_and_rank_threads(
                content=content,
                content_analysis=content_analysis,
                threads=candidate_threads
            )

            return candidate_threads

        except Exception as e:
            st.error(f"Error in discover_relevant_threads: {str(e)}")
            return []

    def _analyze_content(self, content: str) -> Dict:
        """
        Use LLM to analyze content and extract structured information
        """
        try:
            # System message for content analysis
            system_message = """You are an expert content analyzer that extracts structured information from text. 
            You must return valid JSON format only, no additional text or explanations.
            When you extract domain/field and relevent subreddits make sure to also analyze ubreddits like sideprojects, indiehackers, startups, etc. that are subreddits related to entrepreneurship and personal projects."""
            
            # Human message with the prompt
            human_message = f"""
            Analyze this content and extract:
            1. Main topics (3-5 keywords)
            2. Intent (question, discussion, advice, sharing, etc.)
            3. Domain/field (tech, business, personal, etc.)
            4. Relevant subreddit suggestions
            5. Content type (technical, casual, professional, etc.)
            
            Content: "{content}"
            
            Return as JSON format:
            {{
                "topics": ["topic1", "topic2", "topic3"],
                "intent": "question/discussion/advice/sharing",
                "domain": "technology/business/personal/etc",
                "suggested_subreddits": ["subreddit1", "subreddit2"],
                "content_type": "technical/casual/professional",
                "semantic_keywords": ["keyword1", "keyword2", "keyword3"]
            }}
            """
            
            # Use the LLM service
            response = llm_service.generate_response_sync(system_message, human_message)
            
            import json
            analysis = json.loads(response)
            return analysis
            
        except Exception as e:
            st.error(f"Error analyzing content: {str(e)}")
            return {
                "topics": [],
                "intent": "discussion",
                "domain": "general",
                "suggested_subreddits": [],
                "content_type": "casual",
                "semantic_keywords": []
            }

    def _find_relevant_subreddits(self, content_analysis: Dict, limit: int=None) -> List[Dict]:
        """
        Find subreddits using both LLM suggestions and PRAW search
        """
        all_subreddits = []
        
        # 1. Use LLM suggested subreddits
        for sub_name in content_analysis.get("suggested_subreddits", []):
            try:
                subreddit = self.reddit.subreddit(sub_name.removeprefix('r/'))
                if subreddit.subscribers > 1000:  # Filter small subreddits
                    all_subreddits.append({
                        'name': subreddit.display_name,
                        'title': subreddit.title,
                        'subscribers': subreddit.subscribers,
                        'description': getattr(subreddit, 'public_description', '')[:200],
                        'source': 'llm_suggested'
                    })
            except Exception as e:
                st.warning(f"Error fetching subreddit {sub_name}: {str(e)}")
                continue
                
        # 2. Search using extracted topics
        for topic in content_analysis.get("topics", []):
            try:
                search_results = list(self.reddit.subreddits.search(topic, limit=10))
                for subreddit in search_results:
                    if subreddit.subscribers > 5000:
                        all_subreddits.append({
                            'name': subreddit.display_name,
                            'title': subreddit.title,
                            'subscribers': subreddit.subscribers,
                            'description': getattr(subreddit, 'public_description', '')[:200],
                            'source': 'search'
                        })
            except Exception:
                continue
        
        # Remove duplicates and sort
        seen = set()
        unique_subreddits = []
        for sub in all_subreddits:
            if sub['name'] not in seen:
                seen.add(sub['name'])
                unique_subreddits.append(sub)
        
        return unique_subreddits[:limit]

    def _generate_search_queries(self, content_analysis: Dict) -> List[str]:
        """
        Generate search keywords for subreddit discovery using LLM
        """
        try:
            system_message = """You are a Reddit search expert."""

            topics = content_analysis.get('topics', [])
            domain = content_analysis.get('domain', 'general')
            sharing_goal = content_analysis.get('sharing_goal', 'discussion')

            human_message = f"""
            Generate effective search queries for finding relevant subreddits.
            Use boolean operators (AND, OR, NOT) and grouping (parentheses) to create advanced search queries.
            Return only valid JSON with search query strings.
            Based on the following analysis:
            Topics: {topics}
            Domain: {domain}
            Sharing Goal: {sharing_goal}

            Generate advanced search queries using boolean operators and grouping, here are some examples:
            ["topic1 AND topic2", "topic1 OR topic3", "topic1 AND (topic2 OR topic3)", "topic1 NOT topic4", "(topic1 OR topic2) AND topic3"]

            Return as JSON format:
            ["query1", "query2", "query3", "query4", "query5"]
            """

            response = llm_service.generate_response_sync(system_message, human_message)
            pattern = r'```json\n(.*?)```'
            matches = re.findall(pattern, response, re.DOTALL)

            search_queries = json.loads(matches[0])
            return search_queries

        except Exception as e:
            st.error(f"Error generating search queries: {str(e)}")

    def _search_threads_directly(
        self, 
        subreddits_to_search: List[str],
        search_queries: List[str], 
        content_analysis: Dict,
        time_filter: str = 'week',
        threads_limit: int = 20,
        status_callback=None
    ) -> List[Dict]:
        """
        Search for threads directly within specified subreddits using search queries
        """
        all_threads = []
        
        # Remove duplicates from subreddit list
        unique_subreddits = list(set(subreddits_to_search))
        
        for sub_name in unique_subreddits:
            try:
                if status_callback:
                    status_callback(f"Searching in r/{sub_name}...")
                
                subreddit = self.reddit.subreddit(sub_name)
                
                # Search within the subreddit using queries
                for query in search_queries:
                    try:
                        search_results = subreddit.search(query, time_filter=time_filter, limit=threads_limit)
                        
                        for submission in search_results:
                            if submission.stickied or submission.distinguished:
                                continue
                            
                            all_threads.append(self._create_thread_data_from_submission(submission, query))
                    except Exception:
                        continue
                        
            except Exception as e:
                if status_callback:
                    status_callback(f"Could not search in r/{sub_name}")
                continue
        
        # Remove duplicates based on thread ID
        seen_ids = set()
        unique_threads = []
        for thread in all_threads:
            if thread['id'] not in seen_ids:
                seen_ids.add(thread['id'])
                unique_threads.append(thread)
        all_threads = unique_threads
        if status_callback:
            status_callback(f"Found {len(all_threads)} unique threads across {len(unique_subreddits)} subreddits.")
        # Process and deduplicate threads
        return all_threads

    def _create_thread_data_from_submission(self, submission, search_query: str = "") -> Dict:
        """Create thread data dictionary from PRAW submission, including user comment check"""
        user_has_commented = False
        try:
            # Check if the authenticated user has commented on the thread
            submission.comments.replace_more(limit=0)  # Expand all comments
            for comment in submission.comments.list():
                if comment.author and comment.author == self.reddit.user.me():
                    user_has_commented = True
                    break
        except Exception as e:
            st.warning(f"Error checking user comments for thread {submission.id}: {str(e)}")

        return {
            'id': submission.id,
            'title': submission.title,
            'subreddit': submission.subreddit.display_name,
            'url': f"https://reddit.com{submission.permalink}",
            'score': submission.score,
            'num_comments': submission.num_comments,
            'created_utc': submission.created_utc,
            'selftext': submission.selftext,
            'author': str(submission.author) if submission.author else '[deleted]',
            'is_self': submission.is_self,
            'full_text': f"{submission.title} {submission.selftext}",
            'search_query': search_query,
            'user_has_commented': user_has_commented
        }

    def _llm_filter_and_rank_threads(self, content: str, content_analysis: Dict, threads: List[Dict]) -> List[Dict]:
        """
        Use sentence_transformers to filter and rank threads based on content alignment.
        Extract the full content of each thread for evaluation.
        """
        if not threads:
            return []

        try:
            from sentence_transformers import SentenceTransformer, util

            # Initialize the model
            model = SentenceTransformer('all-MiniLM-L6-v2')

            # Prepare thread texts and user content
            original_embedding = model.encode(content, convert_to_tensor=True)
            thread_texts = [f"{thread['title']} {thread['selftext']}" for thread in threads]
            thread_embeddings = model.encode(thread_texts, convert_to_tensor=True)

            # Calculate cosine similarity
            similarity_scores = util.pytorch_cos_sim(original_embedding, thread_embeddings).squeeze().tolist()

            # Apply scores to threads
            for i, thread in enumerate(threads):
                thread['semantic_similarity'] = similarity_scores[i]

            # Sort threads by semantic similarity and final score
            threads.sort(key=lambda t: (
                t.get('semantic_similarity', 0.0),
            ), reverse=True)

            return threads

        except Exception as e:
            st.warning(f"SentenceTransformer filtering failed: {e}. Using original ranking.")
            return threads
    
    def post_comment(self, thread_id: str, comment_text: str) -> bool:
        """
        Post a comment to a Reddit thread
        
        Args:
            thread_id: The ID of the thread to comment on
            comment_text: The text of the comment to post
            
        Returns:
            True if successful, False otherwise
        """

        
        try:
            submission = self.reddit.submission(id=thread_id)
            submission.reply(comment_text)
            return True
        except Exception as e:
            st.error(f"Error posting comment: {str(e)}")
            return False
    
    def _get_threads_from_selected_subreddits(self, selected_subreddits: List[str], search_queries: List[str], status_callback=None) -> List[Dict]:
        """
        Helper method to get threads from user-selected subreddits
        This is used by the two-step UI workflow
        """
        return self._search_threads_directly(
            subreddits_to_search=selected_subreddits,
            search_queries=search_queries,
            content_analysis={},
            time_filter='week',
            threads_limit=8,
            status_callback=status_callback
        )