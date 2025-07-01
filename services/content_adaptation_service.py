from typing import Dict
import streamlit as st
from .llm_service import llm_service

class ContentAdaptationService:
    """Service for adapting content using LLM service"""
    
    def __init__(self):
        """Initialize service - uses global LLM service"""
        self.llm_service = llm_service
    
    def adapt_content(self, original_content: str, thread_context: Dict, goal: str, status_callback=None) -> str:
        """
        Adapt user content to match thread tone and style
        
        Args:
            original_content: The user's original content
            thread_context: Dictionary containing thread information
            goal: User's engagement goal ('promotional' or 'conversational')
            status_callback: Optional function to call for status updates
            
        Returns:
            Adapted content string
        """
        if not self.llm_service.is_available():
            # Return demo adaptation when LLM service is not available
            return self._get_demo_adaptation(original_content, thread_context, goal, status_callback)
        
        try:
            if status_callback:
                status_callback(f"Analyzing r/{thread_context.get('subreddit', 'unknown')} community style...")
            
            if status_callback:
                status_callback("Building context-aware adaptation prompt...")
            
            if status_callback:
                status_callback("Generating personalized comment with AI...")
            
            # Use the LLM service for content adaptation
            adapted_content = self.llm_service.adapt_content_sync(original_content, thread_context, goal)
            
            if status_callback:
                status_callback("Applying final formatting and Reddit conventions...")
            
            return adapted_content
            
        except Exception as e:
            st.error(f"Error adapting content: {str(e)}")
            return self._get_demo_adaptation(original_content, thread_context, goal, status_callback)
    
        if not self.client:
            # Return demo adaptation when OpenAI API is not available
            return self._get_demo_adaptation(original_content, thread_context, goal)
        
        try:
            # Construct the prompt for content adaptation
            prompt = self._build_adaptation_prompt(original_content, thread_context, goal)
            
            # Call OpenAI API
            response = self.client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert Reddit comment writer who helps users create engaging, authentic comments that match the tone and style of specific subreddit communities."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            adapted_content = response.choices[0].message.content.strip()
            return adapted_content
            
        except Exception as e:
            st.error(f"Error adapting content: {str(e)}")
            return self._get_demo_adaptation(original_content, thread_context, goal)
    
    def _build_adaptation_prompt(self, content: str, thread: Dict, goal: str) -> str:
        """Build the prompt for content adaptation"""
        
        thread_info = f"""
Thread Context:
- Subreddit: r/{thread['subreddit']}
- Title: {thread['title']}
- Content: {thread.get('selftext', 'No content')[:300]}
- Comments: {thread['num_comments']}
- Score: {thread['score']}
"""
        
        goal_instruction = {
            'promotional': "Make the comment subtly promotional while being helpful and valuable to the community. Avoid being overly salesy.",
            'conversational': "Make the comment natural and conversational, focusing on genuine engagement and discussion."
        }
        
        prompt = f"""
Please adapt the following content to be a natural Reddit comment that fits well in the given thread:

Original Content:
{content}

{thread_info}

Goal: {goal_instruction.get(goal, goal_instruction['conversational'])}

Requirements:
1. Match the tone and style typical of r/{thread['subreddit']}
2. Make it relevant to the thread topic
3. Keep it authentic and engaging
4. Make it 1-3 paragraphs long
5. Use Reddit-appropriate language and formatting
6. Include relevant context or personal experience if helpful
7. End with a question or call for discussion if appropriate

Adapted Comment:
"""
        
        return prompt
    
    def _get_demo_adaptation(self, original_content: str, thread_context: Dict, goal: str, status_callback=None) -> str:
        """Return demo adapted content when OpenAI API is not available"""
        
        if status_callback:
            status_callback("Demo mode: Analyzing subreddit style...")
            import time
            time.sleep(0.3)
            status_callback("Demo mode: Adapting content tone...")
            time.sleep(0.3)
            status_callback("Demo mode: Applying Reddit formatting...")
            time.sleep(0.2)
        
        subreddit = thread_context.get('subreddit', 'technology')
        title = thread_context.get('title', 'Discussion thread')
        
        if goal == 'promotional':
            adapted = f"""Great question! I've been working on something related to this topic. {original_content[:100]}...
            
This aligns perfectly with what we're seeing in r/{subreddit}. I'd love to hear what others think about this approach.

Has anyone else experienced similar challenges? Would be happy to share more insights if helpful!"""
        else:
            adapted = f"""This is such an interesting topic! {original_content[:100]}...
            
I've been thinking about this exact issue lately, especially in the context of r/{subreddit}. Your point about "{title[:50]}..." really resonates with me.

What's been your experience with this? I'm curious to hear different perspectives from the community."""
        
        return adapted
