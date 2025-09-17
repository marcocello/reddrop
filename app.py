import time
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import datetime
from dotenv import load_dotenv
from utils.helpers import initialize_session_state
from services.reddit_service import RedditService
from services.content_adaptation_service import ContentAdaptationService

# Load environment variables from .env file
load_dotenv()


# Initialize session state
initialize_session_state()

st.set_page_config(
    page_title="Reddrop - Drop into Reddit's most relevant discussions.",
    page_icon='💧',
    layout='wide',
)

# Initialize services
if 'reddit_service' not in st.session_state:
    st.session_state.reddit_service = RedditService()
if 'content_service' not in st.session_state:
    st.session_state.content_service = ContentAdaptationService()

# Initialize two-step process state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'content_analysis' not in st.session_state:
    st.session_state.content_analysis = None
if 'search_queries' not in st.session_state:
    st.session_state.search_queries = []
if 'discovered_subreddits' not in st.session_state:
    st.session_state.discovered_subreddits = []

with st.sidebar:
    # Add the Reddrop logo
    st.image("logo.svg", width=200, output_format="PNG")
    
    # Add "New analysis" link
    if st.button("New analysis", key="new_analysis", use_container_width=True, icon="📝", type="secondary"):
        # Clear all session state for a fresh start
        for key in ['df', 'recommended_threads', 'keywords', 'generated_comments', 'sent_comments']:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.content = ""
        st.rerun()
    
    st.markdown("---")  # Add a separator line
    


    # st.subheader("📝 Recent Analyses")
    
    # Load search history from data folder
    search_history = []
    if os.path.exists('data'):
        for file in sorted(os.listdir('data'), reverse=True):
            if file.endswith('.json'):
                try:
                    with open(f'data/{file}', 'r') as f:
                        data = json.load(f)
                        # Use analysis_name if available, otherwise fallback to first 5 words of content
                        if 'analysis_name' in data:
                            title = data['analysis_name']
                        else:
                            title = ' '.join(data['content'].split()[:5])
                        timestamp = datetime.datetime.fromisoformat(data['timestamp'])
                        search_history.append({
                            'title': title,
                            'timestamp': timestamp,
                            'filename': file
                        })
                except Exception as e:
                    st.error(f"Error loading {file}: {e}")
    
    # Group analyses by date categories
    today = datetime.datetime.now().date()
    yesterday = today - datetime.timedelta(days=1)
    week_ago = today - datetime.timedelta(days=7)
    month_ago = today - datetime.timedelta(days=30)
    
    grouped_analyses = {
        'Today': [],
        'Yesterday': [],
        'This Week': [],
        'Last Week': [],
        'Last Month': [],
        'Older': []
    }
    
    for analysis in search_history:
        analysis_date = analysis['timestamp'].date()
        
        if analysis_date == today:
            grouped_analyses['Today'].append(analysis)
        elif analysis_date == yesterday:
            grouped_analyses['Yesterday'].append(analysis)
        elif analysis_date > week_ago:
            grouped_analyses['This Week'].append(analysis)
        elif analysis_date > today - datetime.timedelta(days=14):
            grouped_analyses['Last Week'].append(analysis)
        elif analysis_date > month_ago:
            grouped_analyses['Last Month'].append(analysis)
        else:
            grouped_analyses['Older'].append(analysis)
    
    # Display grouped analyses
    for group_name, analyses in grouped_analyses.items():
        if analyses:  # Only show groups that have analyses
            st.markdown(f"<h3 style='color: #999999; font-size: 0.9em; font-weight: normal; margin-bottom: 0.5rem;'>{group_name}</h3>", unsafe_allow_html=True)
            for analysis in analyses:
                # Create button for each analysis
                if st.button(
                    analysis['title'],
                    key=f"load_{analysis['filename']}", 
                    # help="Click to load this analysis",
                    use_container_width=True,
                    type="tertiary"
                ):
                    # Load the data
                    with open(f"data/{analysis['filename']}", 'r') as f:
                        data = json.load(f)
                    
                    # Populate session state including the content in text input
                    st.session_state.content = data['content']
                    all_threads = data['threads']
                    
                    # Recreate df_data
                    df_data = []
                    for thread in all_threads:
                        # Only convert timestamp if it's not already a string
                        if isinstance(thread['created_utc'], (int, float)):
                            thread['created_utc'] = datetime.datetime.fromtimestamp(thread['created_utc'], datetime.UTC).isoformat()
                        df_data.append({
                            'Subreddit': f"r/{thread['subreddit']}",
                            'Thread Title': thread['title'],
                            'URL': thread.get('url', ''),
                            'Comments': thread['num_comments'],
                            'Upvotes': thread['score'],
                            "Semantic Similarity": thread.get('semantic_similarity', 0.0),
                            'User Has Commented': "✅" if thread.get('user_has_commented', True) else "❌",
                            'Created': thread['created_utc'],
                            'Thread ID': thread['id']
                        })
                    
                    st.session_state.df = pd.DataFrame(df_data)
                    st.session_state.recommended_threads = all_threads
                    st.session_state.keywords = all_threads[0].get('keywords_used', []) if all_threads else []
                    
                    # Clear previous generated comments
                    st.session_state.generated_comments = {}
                    st.session_state.sent_comments = set()
                    
                    st.success("Loaded previous search results!")
                    st.rerun()

import streamlit as st

st.markdown("Enter the idea, topic, or message you want to share on Reddit...")
content = st.text_area(
            "Enter the idea, topic, or message you want to share on Reddit...",
            value=st.session_state.get('content', ''),
            placeholder="",
            help="We'll find relevant subreddits and threads for your content.",
            label_visibility="collapsed"
        )

quick = st.checkbox("Quick test", value=True, key="quick_test")


if st.button("Find Relevant Threads", type="primary"):
    if not content.strip():
        st.warning("Please enter your content first!")
    else:
        # Save content to session state
        st.session_state.content = content
        
        with st.status("🔍 Analyzing your content and finding relevant threads...", expanded=False) as status:
            # Create a callback function to update status
            def update_status(message):
                st.write(message)
            
            if quick:
                all_threads = st.session_state.reddit_service.discover_relevant_threads(
                    content = content, 
                    status_callback = update_status,
                    time_filter = 'week',
                    subreddit_limit = 3,
                    threads_limit = 5
                )               
            else:
                # Get all threads with real-time status updates
                all_threads = st.session_state.reddit_service.discover_relevant_threads(
                    content = content, 
                    status_callback = update_status)

            status.update(
                label="Analyze completed.", state="complete", expanded=False)
            # Create dataframe for display
            if all_threads:
                df_data = []
                for thread in all_threads:
                    import datetime
                    import re
                    # Only convert timestamp if it's not already a string
                    if isinstance(thread['created_utc'], (int, float)):
                        thread['created_utc'] = datetime.datetime.fromtimestamp(thread['created_utc'], datetime.UTC).isoformat()
                    df_data.append({
                        'Subreddit': f"r/{thread['subreddit']}",
                        'Thread Title': thread['title'],
                        'URL': thread.get('url', ''),
                        'Comments': thread['num_comments'],
                        'Upvotes': thread['score'],
                        "Semantic Similarity": thread.get('semantic_similarity', 0.0),
                        'User Has Commented': "✅" if thread.get('user_has_commented', True) else "❌",
                        'Created': thread['created_utc'],
                        'Thread ID': thread['id']
                    })
                

                st.session_state.df = pd.DataFrame(df_data)
                st.session_state.recommended_threads = all_threads
                st.session_state.keywords = all_threads[0].get('keywords_used', []) if all_threads else []

                # Save search data to JSON
                timestamp = datetime.datetime.now().isoformat()
                analysis_name = ' '.join(content.split()[:5])  # First 5 words as analysis name
                
                # Clean threads data by removing verbose text fields
                cleaned_threads = []
                for thread in all_threads:
                    cleaned_thread = thread.copy()
                    # Remove selftext and full_text to reduce file size
                    cleaned_thread.pop('selftext', None)
                    cleaned_thread.pop('full_text', None)
                    cleaned_threads.append(cleaned_thread)
                
                search_data = {
                    'analysis_name': analysis_name,
                    'content': content,
                    'timestamp': timestamp,
                    'subreddits': list(set(thread['subreddit'] for thread in all_threads)),
                    'threads': cleaned_threads
                }
                filename = f"data/{timestamp.replace(':', '-')}.json"
                os.makedirs('data', exist_ok=True)
                with open(filename, 'w') as f:
                    json.dump(search_data, f, indent=4)

            else:
                st.error("No relevant threads found. Please try with different content.")

if 'df' in st.session_state and not st.session_state.df.empty:
    st.markdown("##### Relevant Threads Found")

    # Initialize generated comments storage if not exists
    if 'generated_comments' not in st.session_state:
        st.session_state.generated_comments = {}

    # Create a search box for filtering threads
    search_query = st.text_input(
        "Search Threads",
        placeholder="Type keywords to filter threads...",
        help="Search threads by title, subreddit, or other attributes."
    )

    # Filter the dataframe based on the search query
    if search_query:
        filtered_df = st.session_state.df[
            st.session_state.df.apply(
                lambda row: search_query.lower() in row.to_string().lower(), axis=1
            )
        ]
    else:
        filtered_df = st.session_state.df

    # # Add user_has_commented column for display before filtering
    # filtered_df['User Has Commented'] = [
    #     "✅" if thread.get('user_has_commented', False) else "❌"
    #     for thread in filtered_df.to_dict('records')
    # ]

    # Create a toggle to exclude threads where the user has already commented
    exclude_commented = st.toggle(
        "Exclude threads where I have already commented",
        value=False,
        help="Enable this to hide threads where you have already commented."
    )

    # Filter the dataframe based on the toggle
    if exclude_commented:
        filtered_df = filtered_df[~filtered_df['User Has Commented'].str.contains("✅")]

    # Create an interactive table with selection
    display_df = filtered_df.copy()

    # Add a selection column for easier identification
    display_df.insert(0, 'Select', False)

    col1, col2 = st.columns([2, 1])

    with col1:
        # Display interactive table with row selection
        edited_df = st.data_editor(
            display_df[['Select', 'Subreddit', 'Thread Title', 'URL', 'Comments', 'Semantic Similarity', 'Upvotes', 'Created', 'User Has Commented']],
            hide_index=True,
            use_container_width=True,
            column_config={
                "Select": st.column_config.CheckboxColumn(
                    "Select",
                    help="Select threads to generate comments for",
                    default=False,
                ),
                "Thread Title": st.column_config.TextColumn(
                    "Thread Title",
                    help="Click to see full title",
                    width="medium",
                ),
                "URL": st.column_config.LinkColumn(
                    "URL",
                    display_text="link",
                    help="Click to open thread in Reddit",
                ),
                "User Has Commented": st.column_config.TextColumn(
                    "User Has Commented",
                    help="Indicates whether you have already commented on this thread",
                    width="small",
                ),
            },
            height=600,
            row_height=80,
            disabled=["Subreddit", "Thread Title", "Comments", "Upvotes", "Created"],
            key="thread_selection_table"
        )
    
    with col2:
    # Get selected threads
        selected_indices = []
        if edited_df is not None:
            selected_indices = [i for i, selected in enumerate(edited_df['Select']) if selected]
        
        # Show selected threads in expanders
        if selected_indices:
            st.markdown(f"##### Selected Threads ({len(selected_indices)})")
            st.markdown("Generate and manage comments for your selected threads:")
            
            for idx in selected_indices:
                thread = st.session_state.recommended_threads[idx]
                thread_id = thread['id']
                
                with st.expander(f"r/{thread['subreddit']} - {thread['title'][:60]}{'...' if len(thread['title']) > 60 else ''}", expanded=False):
                    if st.button("✨ Generate Comment", key=f"generate_{thread_id}", type="secondary"):
                        with st.status("🤖 Generating adapted comment for this thread...") as status:
                            # Create a callback function to update status
                            def update_status(message):
                                st.write(message)
                            
                            adapted_comment = st.session_state.content_service.adapt_content(
                                original_content=st.session_state.content,
                                thread_context=thread,
                                goal=st.session_state.get('goal', 'conversational'),
                                status_callback=update_status
                            )
                            
                            # Store the generated comment
                            st.session_state.generated_comments[thread_id] = adapted_comment
                            st.rerun()
                    
                    # Display generated comment if it exists
                    if thread_id in st.session_state.generated_comments:
                        st.markdown("**💬 Generated Comment:**")
                        
                        # Display the adapted comment in an editable text area
                        edited_comment = st.text_area(
                            "Edit your comment:",
                            value=st.session_state.generated_comments[thread_id],
                            height=120,
                            key=f"comment_edit_{thread_id}",
                            label_visibility="collapsed"
                        )
                        
                        # Update the stored comment if edited
                        if edited_comment != st.session_state.generated_comments[thread_id]:
                            st.session_state.generated_comments[thread_id] = edited_comment
                        
                        # Action buttons
                        col1, col2, col3 = st.columns([2, 2, 1])
                        with col1:
                            if st.button("📤 Send Comment", key=f"send_{thread_id}", type="primary"):
                                with st.status("Sending comment to Reddit...") as status:
                                    st.write("Preparing comment for submission...")
                                    time.sleep(0.3)
                                    st.write(f"Connecting to r/{thread['subreddit']}...")
                                    time.sleep(0.3)
                                    st.write("Posting comment to thread...")
                                    try:
                                        # Send the comment using Reddit service
                                        success = st.session_state.reddit_service.post_comment(
                                            thread_id=thread_id,
                                            comment_text=edited_comment
                                        )
                                        
                                        if success:
                                            st.write("✅ Comment posted successfully!")
                                            st.success("✅ Comment posted successfully!")
                                            # Mark as sent in session state
                                            if 'sent_comments' not in st.session_state:
                                                st.session_state.sent_comments = set()
                                            st.session_state.sent_comments.add(thread_id)
                                        else:
                                            st.error("❌ Failed to post comment. Please try again.")
                                    except Exception as e:
                                        st.error(f"❌ Error posting comment: {str(e)}")
                        
                        with col2:
                            if st.button("🗑️ Clear Comment", key=f"clear_{thread_id}"):
                                del st.session_state.generated_comments[thread_id]
                                if 'sent_comments' in st.session_state and thread_id in st.session_state.sent_comments:
                                    st.session_state.sent_comments.remove(thread_id)
                                st.rerun()
                        
                        with col3:
                            # Show status if comment was sent
                            if 'sent_comments' in st.session_state and thread_id in st.session_state.sent_comments:
                                st.success("✅ Sent")
                            else:
                                st.info("📝 Draft")
