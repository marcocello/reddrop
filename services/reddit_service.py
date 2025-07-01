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
        
        # Simulate some mock threads for demo purposes with enhanced fields
        mock_threads = [
            {
                'id': f'demo_{i}',
                'title': f"Demo Thread {i}: {content[:30]}...",
                'subreddit': f'demo_subreddit_{i%3}',
                'url': f"https://reddit.com/demo_thread_{i}",
                'score': np.random.randint(5, 150),
                'num_comments': np.random.randint(2, 80),
                'created_utc': datetime.now().timestamp(),
                'selftext': f"This is demo content discussing topics related to: {content[:50]}...",
                'author': f"demo_user_{i}",
                'is_self': True,
                'subreddit_info': {
                    'name': f'demo_subreddit_{i%3}',
                    'title': f'Demo Community {i%3}',
                    'subscribers': np.random.randint(5000, 100000),
                    'description': f"Demo community for discussing {content[:20]}...",
                    'relevance_score': np.random.uniform(0.6, 0.95),
                    'source': 'demo'
                },
                'full_text': f"Demo Thread {i}: {content[:30]}... - {content[:100]}...",
                'semantic_similarity': np.random.uniform(0.4, 0.9),
                'engagement_score': np.random.uniform(1.5, 5.0),
                'final_score': np.random.uniform(1.2, 3.5),
                'rank': i + 1,
                'relevance_tier': ['high', 'medium', 'medium', 'low'][i % 4],
                # Enhanced demo fields
                'llm_alignment_score': np.random.uniform(0.6, 0.95),
                'alignment_reasons': [f"Good audience match for demo content", f"Active community discussion"],
                'llm_recommended': i < 3,  # First 3 are recommended
                'keyword_matches': np.random.randint(1, 4),
                'query_pattern_matches': np.random.randint(0, 2)
            } for i in range(8)  # Increased to 8 for better demo
        ]
        
        return self._structure_thread_response(mock_threads)
    
    def discover_relevant_threads(self, content: str, status_callback=None) -> List[Dict]:
        """
        Discover relevant Reddit threads using enhanced LLM analysis + PRAW
        
        Process:
        1. LLM Extraction: Extract topics, tone, target audience, domain, and sharing goal
        2. Reddit Search: Use LLM-generated keyword strings to fetch from broad + niche subreddits
        3. LLM Filtering: Re-rank top threads by alignment with user's sharing goal
        
        Args:
            content: User's input topic or paragraph
            status_callback: Optional function to call for status updates
            
        Returns:
            List of relevant threads with structured info and metrics
        """
        if not self.reddit:
            return self._demo_discover_relevant_threads(content, status_callback)
        
        try:
            # Step 1: Enhanced LLM Extraction
            if status_callback:
                status_callback("Analyzing content with AI to extract topics, tone, and sharing goal...")
            content_analysis = self._analyze_content_with_enhanced_llm(content)
            
            # Step 2: LLM-Enhanced Reddit Search
            if status_callback:
                status_callback("Generating optimized search queries and finding subreddits...")
            search_queries = self._generate_search_keywords_with_llm(content_analysis)
            relevant_subreddits = self._find_subreddits_with_keywords(search_queries, content_analysis)
            
            if status_callback:
                sub_names = [sub['name'] for sub in relevant_subreddits[:5]]
                status_callback(f"Collecting threads from r/{', r/'.join(sub_names)}...")
            candidate_threads = self._get_candidate_threads_enhanced(relevant_subreddits, search_queries)
            
            if status_callback:
                status_callback(f"Found {len(candidate_threads)} candidate threads...")
            
            # Step 3: LLM Filtering and Re-ranking
            if status_callback:
                status_callback("Using AI to filter and rank threads by alignment with your goal...")
            top_candidates = candidate_threads[:30]  # Focus on top 30 for LLM analysis
            llm_ranked_threads = self._llm_filter_and_rank_threads(content, content_analysis, top_candidates)
            
            if status_callback:
                status_callback(f"Found {len(llm_ranked_threads)} high-quality matches!")
            
            # Final structuring
            structured_threads = self._structure_thread_response(llm_ranked_threads[:15])
            
            return structured_threads
            
        except Exception as e:
            st.error(f"Error in discover_relevant_threads: {str(e)}")
            return self._demo_discover_relevant_threads(content, status_callback)

    def _analyze_content_with_enhanced_llm(self, content: str) -> Dict:
        """
        Enhanced LLM analysis to extract comprehensive content information
        including topics, tone, target audience, domain, and sharing goal
        """
        try:
            # System message for enhanced content analysis
            system_message = """You are an expert content strategist and social media analyst. 
            You analyze content to understand the author's intent, audience, and optimal sharing strategy.
            You must return valid JSON format only, no additional text or explanations."""
            
            # Enhanced human message with detailed extraction requirements
            human_message = f"""
            Analyze this content deeply and extract the following information:
            
            1. Core topics and themes (5-7 main keywords/concepts)
            2. Tone and style (professional, casual, technical, personal, humorous, serious, etc.)
            3. Target audience (demographics, expertise level, interests)
            4. Domain/industry (be specific: fintech, gaming, parenting, fitness, etc.)
            5. Sharing goal (what does the user want to achieve: get advice, start discussion, share knowledge, find community, seek validation, etc.)
            6. Content complexity level (beginner, intermediate, expert)
            7. Optimal subreddit categories (broad communities vs niche communities)
            8. Search keywords for Reddit discovery (both broad and specific terms)
            
            Content to analyze: "{content}"
            
            Return as JSON:
            {{
                "topics": ["topic1", "topic2", "topic3", "topic4", "topic5"],
                "tone": "professional/casual/technical/personal/humorous/serious",
                "target_audience": {{
                    "demographics": "description of likely audience",
                    "expertise_level": "beginner/intermediate/expert",
                    "interests": ["interest1", "interest2", "interest3"]
                }},
                "domain": "specific industry or field",
                "sharing_goal": "specific goal description",
                "content_complexity": "beginner/intermediate/expert",
                "subreddit_strategy": {{
                    "broad_communities": ["subreddit1", "subreddit2"],
                    "niche_communities": ["specific_sub1", "specific_sub2"]
                }},
                "search_keywords": {{
                    "broad_terms": ["keyword1", "keyword2"],
                    "specific_terms": ["specific1", "specific2"],
                    "long_tail": ["phrase1", "phrase2"]
                }}
            }}
            """
            
            # Use the LLM service
            response = llm_service.generate_response_sync(system_message, human_message)
            
            import json
            analysis = json.loads(response)
            return analysis
            
        except Exception as e:
            # Enhanced fallback analysis
            return self._enhanced_fallback_analysis(content)

    def _enhanced_fallback_analysis(self, content: str) -> Dict:
        """
        Enhanced fallback content analysis without LLM
        """
        words = re.findall(r'\b\w+\b', content.lower())
        
        # Enhanced keyword extraction
        stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by', 'a', 'an', 'this', 'that', 'it', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        
        # Determine intent and tone
        question_words = ['how', 'what', 'where', 'when', 'why', 'which', 'who']
        sharing_words = ['share', 'show', 'built', 'created', 'made', 'check', 'look']
        advice_words = ['help', 'advice', 'recommend', 'suggest', 'opinion']
        
        sharing_goal = "discussion"
        if any(q in words for q in question_words):
            sharing_goal = "seek advice and answers"
        elif any(s in words for s in sharing_words):
            sharing_goal = "share work and get feedback"
        elif any(a in words for a in advice_words):
            sharing_goal = "get recommendations and opinions"
        
        # Determine tone
        formal_words = ['therefore', 'however', 'furthermore', 'moreover', 'consequently']
        casual_words = ['gonna', 'wanna', 'awesome', 'cool', 'hey', 'guys']
        tone = "casual"
        if any(f in words for f in formal_words):
            tone = "professional"
        elif any(c in words for c in casual_words):
            tone = "casual"
        
        return {
            "topics": keywords[:7],
            "tone": tone,
            "target_audience": {
                "demographics": "general community members",
                "expertise_level": "intermediate",
                "interests": keywords[:3]
            },
            "domain": "general",
            "sharing_goal": sharing_goal,
            "content_complexity": "intermediate",
            "subreddit_strategy": {
                "broad_communities": ["AskReddit", "discussion", "advice"],
                "niche_communities": keywords[:3] if keywords else ["general"]
            },
            "search_keywords": {
                "broad_terms": keywords[:3],
                "specific_terms": keywords[3:6],
                "long_tail": [' '.join(keywords[i:i+2]) for i in range(0, min(4, len(keywords)-1))]
            }
        }

    def _generate_search_keywords_with_llm(self, content_analysis: Dict) -> List[str]:
        """
        Generate advanced search query strings using LLM with Reddit search operators (OR, AND)
        """
        try:
            system_message = """You are a Reddit search expert who creates advanced search queries using Reddit's search operators.
            You generate sophisticated search strings that use OR and AND operators to find the most relevant communities.
            Return only valid JSON with advanced search query strings."""
            
            sharing_goal = content_analysis.get('sharing_goal', 'discussion')
            domain = content_analysis.get('domain', 'general')
            topics = content_analysis.get('topics', [])
            tone = content_analysis.get('tone', 'casual')
            target_audience = content_analysis.get('target_audience', {})
            
            human_message = f"""
            Create advanced Reddit search queries using OR and AND operators based on this analysis:
            
            Domain: {domain}
            Topics: {', '.join(topics)}
            Sharing Goal: {sharing_goal}
            Tone: {tone}
            Target Audience: {target_audience.get('expertise_level', 'intermediate')}
            
            Generate 8-10 advanced search query strings that use:
            - OR operator to find related terms: "machine learning OR ML OR artificial intelligence"
            - AND operator to combine concepts: "startup AND funding AND advice"
            - Combinations: "(python OR programming) AND (beginner OR tutorial)"
            - Specific vs broad queries
            - Professional vs community-focused searches
            
            Focus on creating queries that will find:
            1. Broad communities (general discussion spaces)
            2. Niche specialized communities
            3. Professional/industry communities
            4. Help/advice communities
            5. Show-and-tell communities
            
            Return as JSON:
            {{
                "search_queries": [
                    "query1 OR query2 AND query3",
                    "(term1 OR term2) AND community",
                    "specific_topic AND (help OR advice OR discussion)"
                ]
            }}
            
            Example for a Python web development topic:
            {{
                "search_queries": [
                    "python OR programming AND web",
                    "(django OR flask) AND development",
                    "web development AND (beginner OR tutorial)",
                    "python AND (community OR discussion)",
                    "(coding OR programming) AND help"
                ]
            }}
            """
            
            response = llm_service.generate_response_sync(system_message, human_message)
            
            import json
            result = json.loads(response)
            return result.get('search_queries', [])
            
        except Exception as e:
            # Enhanced fallback with compound search queries
            topics = content_analysis.get('topics', [])
            domain = content_analysis.get('domain', 'general')
            
            fallback_queries = []
            
            if topics:
                # Create OR queries for related topics
                if len(topics) >= 2:
                    fallback_queries.append(f"{topics[0]} OR {topics[1]}")
                    
                # Create AND queries combining concepts
                if len(topics) >= 3:
                    fallback_queries.append(f"{topics[0]} AND {topics[2]}")
                    fallback_queries.append(f"({topics[0]} OR {topics[1]}) AND discussion")
                
                # Add specific queries based on sharing goal
                sharing_goal = content_analysis.get('sharing_goal', 'discussion')
                if 'advice' in sharing_goal:
                    fallback_queries.append(f"{topics[0]} AND (help OR advice)")
                elif 'share' in sharing_goal:
                    fallback_queries.append(f"{topics[0]} AND (show OR share OR feedback)")
                else:
                    fallback_queries.append(f"{topics[0]} AND community")
                
                # Add domain-specific queries
                fallback_queries.append(f"{domain} AND discussion")
                fallback_queries.append(f"({topics[0]} OR {domain}) AND community")
            
            # Add some general fallback queries
            fallback_queries.extend([
                f"{domain} OR discussion",
                "community AND help",
                "discussion AND advice"
            ])
            
            return fallback_queries[:10]  # Limit to 10 queries

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
                    'is_self': thread.get('is_self', False),
                    # Enhanced LLM-based metrics
                    'llm_alignment_score': thread.get('llm_alignment_score', 0.0),
                    'alignment_reasons': thread.get('alignment_reasons', []),
                    'llm_recommended': thread.get('llm_recommended', False),
                    'keyword_matches': thread.get('keyword_matches', 0),
                    'query_pattern_matches': thread.get('query_pattern_matches', 0)
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
    
    def _find_subreddits_with_keywords(self, search_queries: List[str], content_analysis: Dict) -> List[Dict]:
        """
        Find subreddits using advanced search queries with OR/AND operators
        """
        all_subreddits = []
        
        # 1. Execute advanced search queries
        for query in search_queries[:8]:  # Limit to 8 queries to avoid rate limits
            try:
                search_results = list(self.reddit.subreddits.search(query, limit=6))
                for subreddit in search_results:
                    # Different thresholds based on query complexity
                    min_subscribers = 5000 if 'AND' in query or 'OR' in query else 10000
                    
                    if subreddit.subscribers > min_subscribers:
                        # Score based on query complexity and match
                        relevance_score = 0.8 if 'AND' in query else 0.7 if 'OR' in query else 0.6
                        all_subreddits.append(self._create_subreddit_info(
                            subreddit, relevance_score, f'advanced_search: {query[:30]}...'
                        ))
            except Exception as e:
                # If advanced search fails, try simpler version
                simple_query = query.replace(' AND ', ' ').replace(' OR ', ' ').replace('(', '').replace(')', '')
                try:
                    search_results = list(self.reddit.subreddits.search(simple_query, limit=4))
                    for subreddit in search_results:
                        if subreddit.subscribers > 5000:
                            all_subreddits.append(self._create_subreddit_info(
                                subreddit, 0.5, f'simple_search: {simple_query[:30]}...'
                            ))
                except Exception:
                    continue
        
        # 2. Use suggested subreddits from content analysis as backup
        subreddit_strategy = content_analysis.get('subreddit_strategy', {})
        suggested_subs = (subreddit_strategy.get('broad_communities', []) + 
                         subreddit_strategy.get('niche_communities', []))
        
        for sub_name in suggested_subs:
            try:
                subreddit = self.reddit.subreddit(sub_name.removeprefix('r/'))
                if subreddit.subscribers > 1000:
                    all_subreddits.append(self._create_subreddit_info(subreddit, 0.9, 'llm_suggested'))
            except Exception:
                continue
        
        # Remove duplicates and sort
        return self._deduplicate_and_sort_subreddits(all_subreddits)
    
    def _create_subreddit_info(self, subreddit, relevance_score: float, source: str) -> Dict:
        """Helper method to create standardized subreddit info"""
        return {
            'name': subreddit.display_name,
            'title': subreddit.title,
            'subscribers': subreddit.subscribers,
            'description': getattr(subreddit, 'public_description', '')[:200],
            'relevance_score': relevance_score,
            'source': source
        }
    
    def _deduplicate_and_sort_subreddits(self, subreddits: List[Dict]) -> List[Dict]:
        """Remove duplicates and sort subreddits by relevance and size"""
        seen = set()
        unique_subreddits = []
        for sub in subreddits:
            if sub['name'] not in seen:
                seen.add(sub['name'])
                unique_subreddits.append(sub)
        
        # Sort by relevance score and subscriber count
        unique_subreddits.sort(key=lambda x: (x['relevance_score'], x['subscribers']), reverse=True)
        return unique_subreddits[:12]  # Top 12 subreddits
    
    def _get_candidate_threads_enhanced(self, subreddits: List[Dict], search_queries: List[str]) -> List[Dict]:
        """
        Enhanced thread collection with advanced query-based filtering
        """
        all_threads = []
        
        # Extract individual keywords from advanced queries for filtering
        keywords_for_filtering = []
        for query in search_queries:
            # Parse query to extract individual terms
            clean_query = query.replace(' AND ', ' ').replace(' OR ', ' ').replace('(', '').replace(')', '')
            query_keywords = [term.strip() for term in clean_query.split() if len(term.strip()) > 2]
            keywords_for_filtering.extend(query_keywords)
        
        # Remove duplicates and limit
        keywords_for_filtering = list(set(keywords_for_filtering))[:10]
        
        for subreddit_info in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_info['name'])
                
                # Get mix of hot, new, and top threads
                hot_threads = list(subreddit.hot(limit=12))
                new_threads = list(subreddit.new(limit=6))
                top_threads = list(subreddit.top(time_filter='week', limit=6))
                
                for submission in hot_threads + new_threads + top_threads:
                    if submission.stickied or submission.score < 2:
                        continue
                    
                    # Advanced keyword relevance check
                    thread_text = f"{submission.title} {submission.selftext}".lower()
                    keyword_matches = sum(1 for keyword in keywords_for_filtering 
                                        if keyword.lower() in thread_text)
                    
                    # Bonus for query pattern matches (if thread matches query structure)
                    query_pattern_bonus = 0
                    for query in search_queries[:3]:  # Check top 3 queries
                        if self._matches_query_pattern(thread_text, query):
                            query_pattern_bonus += 1
                    
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
                        'full_text': f"{submission.title} {submission.selftext}",
                        'keyword_matches': keyword_matches + query_pattern_bonus,
                        'query_pattern_matches': query_pattern_bonus
                    }
                    
                    all_threads.append(thread_data)
                    
            except Exception as e:
                continue
        
        # Sort by keyword matches, query pattern matches, and engagement
        all_threads.sort(key=lambda x: (
            x['keyword_matches'], 
            x['query_pattern_matches'], 
            x['score'], 
            x['num_comments']
        ), reverse=True)
        return all_threads
    
    def _matches_query_pattern(self, text: str, query: str) -> bool:
        """
        Check if text matches the pattern of an advanced search query
        """
        try:
            text_lower = text.lower()
            
            # Handle OR queries - at least one term should match
            if ' OR ' in query:
                or_parts = query.split(' OR ')
                for part in or_parts:
                    clean_part = part.strip().replace('(', '').replace(')', '')
                    if clean_part.lower() in text_lower:
                        return True
            
            # Handle AND queries - all terms should match
            elif ' AND ' in query:
                and_parts = query.split(' AND ')
                matches = 0
                for part in and_parts:
                    clean_part = part.strip().replace('(', '').replace(')', '')
                    if clean_part.lower() in text_lower:
                        matches += 1
                return matches == len(and_parts)
            
            # Simple query - direct match
            else:
                clean_query = query.strip().replace('(', '').replace(')', '')
                return clean_query.lower() in text_lower
                
        except Exception:
            return False
    
    def _llm_filter_and_rank_threads(self, original_content: str, content_analysis: Dict, candidate_threads: List[Dict]) -> List[Dict]:
        """
        Use LLM to filter and rank threads by alignment with user's sharing goal
        """
        if not candidate_threads:
            return []
        
        try:
            sharing_goal = content_analysis.get('sharing_goal', 'share and discuss')
            domain = content_analysis.get('domain', 'general')
            
            # Prepare thread summaries for LLM analysis
            thread_summaries = []
            for i, thread in enumerate(candidate_threads[:20]):  # Limit to top 20 for LLM analysis
                summary = {
                    'index': i,
                    'title': thread['title'],
                    'subreddit': thread['subreddit'],
                    'score': thread['score'],
                    'comments': thread['num_comments'],
                    'preview': thread['full_text'][:200] + '...' if len(thread['full_text']) > 200 else thread['full_text']
                }
                thread_summaries.append(summary)
            
            system_message = """You are a Reddit content strategist expert at matching user content with optimal discussion threads.
            Analyze threads and rank them by how well they align with the user's sharing goal and content type.
            Return only valid JSON."""
            
            human_message = f"""
            A user wants to {sharing_goal} about this content in the {domain} domain:
            
            User's content: "{original_content[:300]}..."
            
            Rank these Reddit threads (1-20) by how well they align with the user's goal:
            {thread_summaries}
            
            Consider:
            1. Audience alignment (would this community appreciate the content?)
            2. Discussion potential (likely to generate meaningful engagement?)
            3. Relevance match (does the thread topic align with user's content?)
            4. Community activity (active discussion vs dead thread?)
            5. Appropriateness (is this the right place for this type of sharing?)
            
            Return top 10 threads ranked by alignment score:
            {{
                "ranked_threads": [
                    {{
                        "index": 0,
                        "alignment_score": 0.95,
                        "alignment_reasons": ["reason1", "reason2"],
                        "recommended": true
                    }}
                ]
            }}
            """
            
            response = llm_service.generate_response_sync(system_message, human_message)
            
            import json
            ranking_result = json.loads(response)
            
            # Apply LLM ranking to threads
            ranked_threads = []
            for thread_rank in ranking_result.get('ranked_threads', []):
                thread_index = thread_rank.get('index', 0)
                if thread_index < len(candidate_threads):
                    thread = candidate_threads[thread_index].copy()
                    thread['llm_alignment_score'] = thread_rank.get('alignment_score', 0.5)
                    thread['alignment_reasons'] = thread_rank.get('alignment_reasons', [])
                    thread['llm_recommended'] = thread_rank.get('recommended', False)
                    thread['final_score'] = self._calculate_enhanced_final_score(thread, content_analysis)
                    ranked_threads.append(thread)
            
            # Sort by final score
            ranked_threads.sort(key=lambda x: x.get('final_score', 0), reverse=True)
            
            return ranked_threads
            
        except Exception as e:
            # Fallback to basic scoring if LLM fails
            for thread in candidate_threads:
                thread['llm_alignment_score'] = 0.5
                thread['alignment_reasons'] = ["Basic scoring used"]
                thread['llm_recommended'] = True
                thread['final_score'] = self._calculate_basic_final_score(thread)
            
            candidate_threads.sort(key=lambda x: x.get('final_score', 0), reverse=True)
            return candidate_threads[:15]
    
    def _calculate_enhanced_final_score(self, thread: Dict, content_analysis: Dict) -> float:
        """Calculate final score incorporating LLM alignment and query pattern matching"""
        llm_score = thread.get('llm_alignment_score', 0.5)
        engagement_score = self._calculate_engagement_score_from_data(thread)
        subreddit_score = thread.get('subreddit_info', {}).get('relevance_score', 0.5)
        keyword_match_score = min(thread.get('keyword_matches', 0) / 3, 1.0)
        query_pattern_score = min(thread.get('query_pattern_matches', 0) / 2, 1.0)
        
        # Weighted combination with emphasis on LLM alignment and query patterns
        final_score = (
            llm_score * 0.4 +              # 40% LLM alignment
            query_pattern_score * 0.2 +    # 20% advanced query pattern matching
            engagement_score * 0.15 +      # 15% engagement
            subreddit_score * 0.15 +       # 15% subreddit relevance
            keyword_match_score * 0.1      # 10% basic keyword matching
        )
        
        return final_score
    
    def _calculate_basic_final_score(self, thread: Dict) -> float:
        """Fallback final score calculation"""
        engagement_score = self._calculate_engagement_score_from_data(thread)
        subreddit_score = thread.get('subreddit_info', {}).get('relevance_score', 0.5)
        keyword_match_score = min(thread.get('keyword_matches', 0) / 3, 1.0)
        
        return (engagement_score * 0.4 + subreddit_score * 0.3 + keyword_match_score * 0.3)