"""
LLM Service for Reddrop - Reusable AI/LLM functionality
Supports Azure OpenAI and regular OpenAI
"""

import os
import asyncio
from typing import Dict, List, Optional, Any
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from langchain.globals import set_llm_cache

# Disable LangChain's caching
set_llm_cache(None)

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class LLMService:
    """Reusable LLM service for content generation, adaptation, and comment creation"""
    
    def __init__(self):
        """Initialize LLM client with Azure OpenAI or OpenAI"""
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate LLM client based on available environment variables"""
        try:
            # Check for Azure OpenAI configuration first
            azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
            azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
            azure_deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
            azure_api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
            
            if azure_api_key and azure_endpoint and azure_deployment:
                # Use Azure OpenAI
                self.client = AzureChatOpenAI(
                    deployment_name=azure_deployment,
                    openai_api_key=azure_api_key,
                    azure_endpoint=azure_endpoint,
                    api_version=azure_api_version,
                    temperature=float(os.getenv('LLM_TEMPERATURE', '0.7'))
                )
                self.client_type = 'azure'
                return
            
            # No valid configuration found
            if hasattr(st, 'warning'):
                st.warning("LLM API not configured. Using demo mode.")
            self.client = None
            self.client_type = 'demo'
            
        except Exception as e:
            if hasattr(st, 'error'):
                st.error(f"Error initializing LLM client: {str(e)}")
            self.client = None
            self.client_type = 'demo'
    
    async def generate_response(self, system_message: str, human_message: str, **kwargs) -> str:
        """
        Generate a response using the LLM client
        
        Args:
            system_message: System/instruction message
            human_message: Human/user message
            **kwargs: Additional parameters
            
        Returns:
            Generated response string
        """
        if not self.client:
            return self._get_demo_response(system_message, human_message)
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_message),
                HumanMessage(content=human_message)
            ])
            
            chain = prompt | self.client
            response = await chain.ainvoke({})
            return response.content.strip()
            
        except Exception as e:
            if hasattr(st, 'error'):
                st.error(f"Error generating LLM response: {str(e)}")
            return self._get_demo_response(system_message, human_message)
    
    def generate_response_sync(self, system_message: str, human_message: str, **kwargs) -> str:
        """
        Synchronous wrapper for generate_response
        
        Args:
            system_message: System/instruction message
            human_message: Human/user message
            **kwargs: Additional parameters
            
        Returns:
            Generated response string
        """
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self.generate_response(system_message, human_message, **kwargs)
            )
        except RuntimeError:
            # If no event loop exists, create one
            return asyncio.run(
                self.generate_response(system_message, human_message, **kwargs)
            )
    
    async def adapt_content(self, original_content: str, thread_context: Dict, goal: str) -> str:
        """
        Adapt user content to match thread tone and style
        
        Args:
            original_content: The user's original content
            thread_context: Dictionary containing thread information
            goal: User's engagement goal ('promotional' or 'conversational')
            
        Returns:
            Adapted content string
        """
        system_message = """You are an expert Reddit comment writer who helps users create engaging, authentic comments that match the tone and style of specific subreddit communities."""
        
        human_message = self._build_adaptation_prompt(original_content, thread_context, goal)
        
        return await self.generate_response(system_message, human_message)
    
    def adapt_content_sync(self, original_content: str, thread_context: Dict, goal: str) -> str:
        """
        Synchronous wrapper for adapt_content
        
        Args:
            original_content: The user's original content
            thread_context: Dictionary containing thread information
            goal: User's engagement goal ('promotional' or 'conversational')
            
        Returns:
            Adapted content string
        """
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self.adapt_content(original_content, thread_context, goal)
            )
        except RuntimeError:
            # If no event loop exists, create one
            return asyncio.run(
                self.adapt_content(original_content, thread_context, goal)
            )
    
    async def generate_comment(self, thread_context: Dict, content_context: str, style: str = "conversational") -> str:
        """
        Generate a new comment for a Reddit thread
        
        Args:
            thread_context: Dictionary containing thread information
            content_context: Context or topic to comment about
            style: Comment style ('conversational', 'promotional', 'informative', etc.)
            
        Returns:
            Generated comment string
        """
        system_message = """You are an expert Reddit user who creates authentic, engaging comments that add value to discussions. Your comments should be natural, helpful, and appropriate for the specific subreddit community."""
        
        human_message = self._build_comment_prompt(thread_context, content_context, style)
        
        return await self.generate_response(system_message, human_message)
    
    def _build_adaptation_prompt(self, content: str, thread: Dict, goal: str) -> str:
        """Build the prompt for content adaptation"""
        
        thread_info = f"""
Thread Context:
- Subreddit: r/{thread.get('subreddit', 'unknown')}
- Title: {thread.get('title', 'No title')}
- Content: {thread.get('selftext', 'No content')[:300]}
- Comments: {thread.get('num_comments', 0)}
- Score: {thread.get('score', 0)}
"""
        
        goal_instruction = {
            'promotional': "Make the comment subtly promotional while being helpful and valuable to the community. Avoid being overly salesy.",
            'conversational': "Make the comment natural and conversational, focusing on genuine engagement and discussion.",
            'informative': "Make the comment informative and educational, sharing knowledge and insights.",
            'supportive': "Make the comment supportive and encouraging, offering help or empathy."
        }
        
        prompt = f"""
Please adapt the following content to be a natural Reddit comment that fits well in the given thread:

Original Content:
{content}

{thread_info}

Goal: {goal_instruction.get(goal, goal_instruction['conversational'])}

Requirements:
1. Match the tone and style typical of r/{thread.get('subreddit', 'unknown')}
2. Make it relevant to the thread topic
3. Keep it authentic and engaging
4. Make it 1-3 paragraphs long
5. Use Reddit-appropriate language and formatting
6. Include relevant context or personal experience if helpful
7. End with a question or call for discussion if appropriate

Adapted Comment:
"""
        
        return prompt
    
    def _build_comment_prompt(self, thread: Dict, content_context: str, style: str) -> str:
        """Build the prompt for comment generation"""
        
        thread_info = f"""
Thread Context:
- Subreddit: r/{thread.get('subreddit', 'unknown')}
- Title: {thread.get('title', 'No title')}
- Content: {thread.get('selftext', 'No content')[:300]}
- Comments: {thread.get('num_comments', 0)}
- Score: {thread.get('score', 0)}
"""
        
        style_instructions = {
            'conversational': "Create a natural, friendly comment that encourages discussion",
            'promotional': "Create a subtly promotional comment that provides value while mentioning relevant products/services",
            'informative': "Create an educational comment that shares knowledge and insights",
            'supportive': "Create a supportive comment that offers help, encouragement, or empathy",
            'questioning': "Create a comment that asks thoughtful questions to spark discussion",
            'sharing': "Create a comment that shares personal experience or anecdotes"
        }
        
        prompt = f"""
Please generate a Reddit comment for the following thread:

{thread_info}

Content Context/Topic: {content_context}

Style: {style_instructions.get(style, style_instructions['conversational'])}

Requirements:
1. Be authentic and natural for r/{thread.get('subreddit', 'unknown')}
2. Add value to the discussion
3. Be 1-3 paragraphs long
4. Use appropriate Reddit formatting and tone
5. Be relevant to the thread topic
6. Encourage engagement if appropriate

Generated Comment:
"""
        
        return prompt
    
    def _get_demo_response(self, system_message: str, human_message: str) -> str:
        """Return demo response when LLM client is not available"""
        if "adapt" in human_message.lower():
            return """This is interesting! I've been thinking about this exact topic lately. 
            
Your perspective really resonates with what we're seeing in the community. I'd love to hear more about your experience with this.

Has anyone else encountered similar situations? Would be great to get different viewpoints on this!"""
        
        elif "generate" in human_message.lower() or "comment" in human_message.lower():
            return """Great discussion! This topic is really relevant right now.
            
I've had some experience with this, and I think the key points mentioned here are spot on. It's always interesting to see how different approaches work for different people.

What's been your experience with this? I'm curious to hear how others in the community have tackled similar challenges."""
        
        else:
            return """Thanks for sharing this! It's an interesting perspective that I hadn't considered before.
            
This kind of discussion is exactly why I love this community - always learning something new from different viewpoints.

Looking forward to seeing how this develops!"""
    
    def is_available(self) -> bool:
        """Check if LLM service is available (not in demo mode)"""
        return self.client is not None
    
    def get_client_info(self) -> Dict[str, Any]:
        """Get information about the current LLM client"""
        return {
            'client_type': self.client_type,
            'available': self.is_available(),
            'model': getattr(self.client, 'model_name', None) or getattr(self.client, 'deployment_name', None)
        }


# Global instance for easy access
llm_service = LLMService()
