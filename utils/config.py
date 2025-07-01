"""
Configuration settings for Reddrop MVP
"""

import os
from typing import Dict, Any

class Config:
    """Application configuration"""
    
    # Reddit API Configuration
    REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
    REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
    REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'Reddrop:v1.0:by-user')
    
    # OpenAI API Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    OPENAI_MAX_TOKENS = int(os.getenv('OPENAI_MAX_TOKENS', '300'))
    OPENAI_TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', '0.7'))
    
    # Azure OpenAI API Configuration
    AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
    AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
    AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
    
    # General LLM Configuration
    LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.7'))
    
    # Application Settings
    MAX_SUBREDDITS = int(os.getenv('MAX_SUBREDDITS', '10'))
    MAX_THREADS_PER_SUBREDDIT = int(os.getenv('MAX_THREADS_PER_SUBREDDIT', '10'))
    MAX_KEYWORDS = int(os.getenv('MAX_KEYWORDS', '10'))
    MIN_THREAD_SCORE = int(os.getenv('MIN_THREAD_SCORE', '5'))
    MIN_THREAD_COMMENTS = int(os.getenv('MIN_THREAD_COMMENTS', '2'))
    MIN_SUBREDDIT_SUBSCRIBERS = int(os.getenv('MIN_SUBREDDIT_SUBSCRIBERS', '1000'))
    
    # Database Configuration (for future use)
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///reddrop.db')
    
    @classmethod
    def get_reddit_config(cls) -> Dict[str, Any]:
        """Get Reddit API configuration"""
        return {
            'client_id': cls.REDDIT_CLIENT_ID,
            'client_secret': cls.REDDIT_CLIENT_SECRET,
            'user_agent': cls.REDDIT_USER_AGENT
        }
    
    @classmethod
    def get_openai_config(cls) -> Dict[str, Any]:
        """Get OpenAI API configuration"""
        return {
            'api_key': cls.OPENAI_API_KEY,
            'model': cls.OPENAI_MODEL,
            'max_tokens': cls.OPENAI_MAX_TOKENS,
            'temperature': cls.OPENAI_TEMPERATURE
        }
    
    @classmethod
    def get_azure_openai_config(cls) -> Dict[str, Any]:
        """Get Azure OpenAI API configuration"""
        return {
            'api_key': cls.AZURE_OPENAI_API_KEY,
            'endpoint': cls.AZURE_OPENAI_ENDPOINT,
            'deployment_name': cls.AZURE_OPENAI_DEPLOYMENT_NAME,
            'api_version': cls.AZURE_OPENAI_API_VERSION,
            'temperature': cls.LLM_TEMPERATURE
        }
    
    @classmethod
    def get_llm_config(cls) -> Dict[str, Any]:
        """Get general LLM configuration"""
        return {
            'temperature': cls.LLM_TEMPERATURE,
            'openai': cls.get_openai_config(),
            'azure_openai': cls.get_azure_openai_config()
        }
    
    @classmethod
    def get_app_config(cls) -> Dict[str, Any]:
        """Get application configuration"""
        return {
            'max_subreddits': cls.MAX_SUBREDDITS,
            'max_threads_per_subreddit': cls.MAX_THREADS_PER_SUBREDDIT,
            'max_keywords': cls.MAX_KEYWORDS,
            'min_thread_score': cls.MIN_THREAD_SCORE,
            'min_thread_comments': cls.MIN_THREAD_COMMENTS,
            'min_subreddit_subscribers': cls.MIN_SUBREDDIT_SUBSCRIBERS
        }
    
    @classmethod
    def is_configured(cls) -> Dict[str, bool]:
        """Check which services are configured"""
        return {
            'reddit': bool(cls.REDDIT_CLIENT_ID and cls.REDDIT_CLIENT_SECRET),
            'openai': bool(cls.OPENAI_API_KEY)
        }
