import praw
import os
import time
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
            
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
            
            # Test the connection
            self.reddit.user.me()
            
        except Exception as e:
            # For demo purposes, we'll create a mock client
            st.warning("Reddit API not configured. Using demo mode.")
            self.reddit = None

    def _demo_discover_relevant_threads(self, content: str, status_callback=None) -> List[Dict]:
        """
        Demo method to simulate thread discovery without Reddit API
        
        Args:
            content: User's input topic or paragraph
            status_callback: Optional function to call for status updates
            
        Returns:
            List of mock threads with structured info and metrics
        """
        if status_callback:
            status_callback("Demo mode: Simulating content analysis...")
            time.sleep(0.5)
            status_callback("Demo mode: Finding mock subreddits...")
            time.sleep(0.5)
            status_callback("Demo mode: Generating sample threads...")
            time.sleep(0.5)
            status_callback("Demo mode: Calculating mock relevance scores...")
            time.sleep(0.3)
        
        # Simulate some mock threads for demo purposes
        mock_threads = [
            {
                'id': f'demo_{i}',
                'title': f"Demo Thread {i} on {content[:20]}...",
                'subreddit': 'demo_subreddit',
                'url': f"https://reddit.com/demo_thread_{i}",
                'score': np.random.randint(1, 100),
                'num_comments': np.random.randint(0, 50),
                'created_utc': datetime.now().timestamp(),
                'selftext': "This is a demo thread content.",
                'author': "demo_user",
                'is_self': True,
                'subreddit_info': {
                    'name': 'demo_subreddit',
                    'title': 'Demo Subreddit',
                    'subscribers': 1000,
                    'description': "This is a demo subreddit description.",
                    'relevance_score': 0.8,
                    'source': 'demo'
                },
                'full_text': f"Demo Thread {i} content related to {content}",
                'semantic_similarity': np.random.uniform(0.3, 0.9),
                'engagement_score': np.random.uniform(1.0, 5.0),
                'final_score': np.random.uniform(1.0, 3.0),
                'rank': i + 1,
                'relevance_tier': 'medium'
            } for i in range(5)
        ]
        
        return self._structure_thread_response(mock_threads)
    
    def discover_relevant_threads(self, content: str, status_callback=None) -> List[Dict]:
        """
        Discover relevant Reddit threads using LLM analysis + PRAW
        
        Args:
            content: User's input topic or paragraph
            status_callback: Optional function to call for status updates
            
        Returns:
            List of relevant threads with structured info and metrics
        """
        if not self.reddit:
            return self._demo_discover_relevant_threads(content, status_callback)
        
        try:
            if status_callback:
                status_callback("Analyzing content with AI to extract topics and intent...")
            content_analysis = self._analyze_content_with_llm(content)
            
            if status_callback:
                topics = content_analysis.get('topics', [])
                status_callback(f"Finding relevant subreddits for topics: {', '.join(topics[:3])}...")
            relevant_subreddits = self._find_relevant_subreddits_smart(content_analysis)
            
            if status_callback:
                sub_names = [sub['name'] for sub in relevant_subreddits[:3]]
                status_callback(f"Collecting threads from r/{', r/'.join(sub_names)}...")
            candidate_threads = self._get_candidate_threads(relevant_subreddits)
            
            if status_callback:
                status_callback(f"Analyzing {len(candidate_threads)} threads for relevance...")
            scored_threads = self._score_threads_semantically(content, candidate_threads, content_analysis)
            
            if status_callback:
                status_callback("Ranking threads by engagement and relevance scores...")
            final_threads = self._rank_and_filter_threads(scored_threads)
            
            if status_callback:
                status_callback(f"Found {len(final_threads)} high-quality matches!")
            structured_threads = self._structure_thread_response(final_threads[:15])
            
            return structured_threads
            
        except Exception as e:
            st.error(f"Error in discover_relevant_threads: {str(e)}")
            return self._demo_discover_relevant_threads(content, status_callback)

    def _analyze_content_with_llm(self, content: str) -> Dict:
        """
        Use LLM to analyze content and extract structured information
        """
        try:
            # System message for content analysis
            system_message = """You are an expert content analyzer that extracts structured information from text. 
            You must return valid JSON format only, no additional text or explanations."""
            
            # Human message with the prompt
            human_message = f"""
            Analyze this content and extract:
            1. Main topics (3-5 keywords)
            2. Intent (question, discussion, advice, sharing, etc.)
            3. Domain/field (tech, business, personal, etc.)
            4. Relevant subreddit suggestions (5-8 subreddits)
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
            # Fallback to basic keyword extraction
            return self._basic_content_analysis(content)

    def _basic_content_analysis(self, content: str) -> Dict:
        """
        Fallback content analysis without LLM
        """
        words = re.findall(r'\b\w+\b', content.lower())
        
        # Simple keyword extraction
        stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by', 'a', 'an'}
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        
        # Determine intent based on question words
        question_words = ['how', 'what', 'where', 'when', 'why', 'which', 'who']
        intent = "question" if any(q in words for q in question_words) else "discussion"
        
        return {
            "topics": keywords[:5],
            "intent": intent,
            "domain": "general",
            "suggested_subreddits": ["AskReddit", "discussion", "advice"],
            "content_type": "casual",
            "semantic_keywords": keywords[:8]
        }

    def _find_relevant_subreddits_smart(self, content_analysis: Dict) -> List[Dict]:
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
                        'relevance_score': 0.9,  # High score for LLM suggestions
                        'source': 'llm_suggested'
                    })
            except Exception as e:
                st.warning(f"Error fetching subreddit {sub_name}: {str(e)}")
                continue
                
        # 2. Search using extracted topics
        for topic in content_analysis.get("topics", [])[:3]:
            try:
                search_results = list(self.reddit.subreddits.search(topic, limit=5))
                for subreddit in search_results:
                    if subreddit.subscribers > 5000:
                        all_subreddits.append({
                            'name': subreddit.display_name,
                            'title': subreddit.title,
                            'subscribers': subreddit.subscribers,
                            'description': getattr(subreddit, 'public_description', '')[:200],
                            'relevance_score': self._calculate_relevance_score(topic, subreddit),
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
        
        unique_subreddits.sort(key=lambda x: (x['relevance_score'], x['subscribers']), reverse=True)
        return unique_subreddits[:8]  # Top 8 subreddits

    def _get_candidate_threads(self, subreddits: List[Dict]) -> List[Dict]:
        """
        Get candidate threads from relevant subreddits
        """
        all_threads = []
        
        for subreddit_info in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_info['name'])
                
                # Get mix of hot and new threads
                hot_threads = list(subreddit.hot(limit=10))
                new_threads = list(subreddit.new(limit=5))
                
                for submission in hot_threads + new_threads:
                    if submission.stickied or submission.score < 3:
                        continue
                    
                    thread_data = {
                        'id': submission.id,
                        'title': submission.title,
                        'subreddit': subreddit_info['name'],
                        'url': submission.url,
                        'score': submission.score,
                        'num_comments': submission.num_comments,
                        'created_utc': submission.created_utc,
                        'selftext': submission.selftext,
                        'author': str(submission.author) if submission.author else '[deleted]',
                        'is_self': submission.is_self,
                        'subreddit_info': subreddit_info,
                        'full_text': f"{submission.title} {submission.selftext}"
                    }
                    
                    all_threads.append(thread_data)
                    
            except Exception as e:
                continue
        
        return all_threads

    def _score_threads_semantically(self, original_content: str, threads: List[Dict], content_analysis: Dict) -> List[Dict]:
        """
        Score threads using semantic similarity
        """
        if not threads:
            return []
        
        # Prepare texts for similarity comparison
        original_text = original_content.lower()
        thread_texts = [thread['full_text'].lower() for thread in threads]
        all_texts = [original_text] + thread_texts
        
        try:
            # Use TF-IDF for semantic similarity
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # Calculate cosine similarity
            similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            
            # Add scores to threads
            for i, thread in enumerate(threads):
                thread['semantic_similarity'] = similarity_scores[i]
                thread['engagement_score'] = self._calculate_engagement_score_from_data(thread)
                thread['final_score'] = self._calculate_final_score(thread, content_analysis)
            
        except Exception as e:
            # Fallback scoring
            for thread in threads:
                thread['semantic_similarity'] = 0.5
                thread['engagement_score'] = self._calculate_engagement_score_from_data(thread)
                thread['final_score'] = thread['engagement_score']
        
        return threads

    def _calculate_engagement_score_from_data(self, thread: Dict) -> float:
        """
        Calculate engagement score from thread data
        """
        score = thread.get('score', 0)
        comments = thread.get('num_comments', 0)
        
        if score == 0:
            return 0.0
        
        # Engagement ratio with bonus for higher absolute numbers
        engagement = (comments / score) * 10 + np.log(score + 1) * 0.5
        return min(engagement, 10.0)  # Cap at 10

    def _calculate_final_score(self, thread: Dict, content_analysis: Dict) -> float:
        """
        Calculate final relevance score combining multiple factors
        """
        semantic_score = thread.get('semantic_similarity', 0.0)
        engagement_score = thread.get('engagement_score', 0.0)
        subreddit_score = thread.get('subreddit_info', {}).get('relevance_score', 0.0)
        
        # Weighted combination
        final_score = (
            semantic_score * 0.4 +          # 40% semantic similarity
            engagement_score * 0.3 +        # 30% engagement
            subreddit_score * 0.2 +         # 20% subreddit relevance
            min(thread.get('score', 0) / 100, 1.0) * 0.1  # 10% popularity
        )
        
        return final_score

    def _rank_and_filter_threads(self, threads: List[Dict]) -> List[Dict]:
        """
        Final ranking and filtering of threads
        """
        # Remove duplicates
        seen_ids = set()
        unique_threads = []
        for thread in threads:
            if thread['id'] not in seen_ids:
                seen_ids.add(thread['id'])
                unique_threads.append(thread)
        
        # Sort by final score
        unique_threads.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        # Add ranking metadata
        for i, thread in enumerate(unique_threads):
            thread['rank'] = i + 1
            thread['relevance_tier'] = 'high' if thread.get('final_score', 0) > 2.0 else 'medium' if thread.get('final_score', 0) > 1.0 else 'low'
        
        return unique_threads

    def _structure_thread_response(self, threads: List[Dict]) -> List[Dict]:
        """
        Structure thread data into info and metrics sections
        
        Args:
            threads: List of thread dictionaries with all data
            
        Returns:
            List of structured threads with info and metrics sections
        """
        structured_threads = []
        
        for thread in threads:
            structured_thread = {
                'info': {
                    'title': thread.get('title', ''),
                    'subreddit': thread.get('subreddit', ''),
                    'url': thread.get('url', ''),
                    'author': thread.get('author', ''),
                    'full_text': thread.get('full_text', '')
                },
                'metrics': {
                    'id': thread.get('id', ''),
                    'score': thread.get('score', 0),
                    'num_comments': thread.get('num_comments', 0),
                    'created_utc': thread.get('created_utc', 0),
                    'semantic_similarity': thread.get('semantic_similarity', 0.0),
                    'engagement_score': thread.get('engagement_score', 0.0),
                    'final_score': thread.get('final_score', 0.0),
                    'rank': thread.get('rank', 0),
                    'relevance_tier': thread.get('relevance_tier', 'low'),
                    'subreddit_subscribers': thread.get('subreddit_info', {}).get('subscribers', 0),
                    'subreddit_relevance_score': thread.get('subreddit_info', {}).get('relevance_score', 0.0),
                    'is_self': thread.get('is_self', False)
                }
            }
            
            structured_threads.append(structured_thread)
        
        return structured_threads
    
    def _calculate_relevance_score(self, topic: str, subreddit) -> float:
        """
        Calculate relevance score between a topic and subreddit
        
        Args:
            topic: The search topic/keyword
            subreddit: PRAW subreddit object
            
        Returns:
            Float relevance score between 0.0 and 1.0
        """
        try:
            topic_lower = topic.lower()
            
            # Check subreddit name similarity
            name_score = 0.0
            if topic_lower in subreddit.display_name.lower():
                name_score = 0.8
            elif any(word in subreddit.display_name.lower() for word in topic_lower.split()):
                name_score = 0.6
            
            # Check subreddit title similarity
            title_score = 0.0
            if hasattr(subreddit, 'title') and subreddit.title:
                title_lower = subreddit.title.lower()
                if topic_lower in title_lower:
                    title_score = 0.7
                elif any(word in title_lower for word in topic_lower.split()):
                    title_score = 0.5
            
            # Check description similarity
            description_score = 0.0
            description = getattr(subreddit, 'public_description', '') or getattr(subreddit, 'description', '')
            if description:
                description_lower = description.lower()
                if topic_lower in description_lower:
                    description_score = 0.6
                elif any(word in description_lower for word in topic_lower.split() if len(word) > 3):
                    description_score = 0.4
            
            # Subscriber count bonus (larger communities get slight boost)
            subscriber_bonus = min(np.log(subreddit.subscribers + 1) / 20, 0.2) if subreddit.subscribers > 0 else 0.0
            
            # Combine scores with weights
            final_score = (
                name_score * 0.4 +          # 40% weight on name match
                title_score * 0.3 +         # 30% weight on title match
                description_score * 0.2 +   # 20% weight on description match
                subscriber_bonus * 0.1      # 10% weight on subscriber bonus
            )
            
            return min(final_score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            # Fallback to basic scoring
            return 0.5