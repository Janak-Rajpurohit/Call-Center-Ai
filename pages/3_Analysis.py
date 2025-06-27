import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
from datetime import datetime
import numpy as np
import emoji
from utils import sentiment

# Set page config
st.set_page_config(
    page_title="📊 Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin: 1rem 0;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #2A2D3E 0%, #32364A 100%);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        height: 8rem;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.1);
        transition: transform 0.2s, box-shadow 0.2s;
        backdrop-filter: blur(10px);
        margin-bottom: 2rem;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 20px rgba(0,0,0,0.2);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.25rem 0;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #B8B9BC;
        margin-top: 0.3rem;
    }

    .metric-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
        color: #667eea;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #dfdfdf;
        border-radius: 4px 4px 0px 0px;
        color: #0f5d95;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6D7995;
        color: white;
    }
                
        /* Main app styling */
        .main {
            padding-top: 1rem;
        }
        
        /* Header styling */
        .main-header {
            background: linear-gradient(135deg, #2A2D3E 0%, #32364A 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        }
        
        .main-header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        /* Chat container */
        .chat-container {
            background: linear-gradient(135deg, #2A2D3E 0%, #32364A 100%);
            border-radius: 15px;
            padding: 1rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            margin-bottom: 1rem;
            min-height: 400px;
            max-height: 500px;
            overflow-y: auto;
            border: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
        }
        
        /* Message styling */
        .user-message {
            background: linear-gradient(135deg, #2A2D3E 0%, #32364A 100%);
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 20px 20px 5px 20px;
            margin: 0.5rem 0 0.5rem auto;
            max-width: 80%;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            word-wrap: break-word;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .ai-message {
            background: linear-gradient(135deg, #32364A 0%, #2A2D3E 100%);
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 20px 20px 20px 5px;
            margin: 0.5rem auto 0.5rem 0;
            max-width: 80%;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            word-wrap: break-word;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        /* Input container */
        .input-container {
            position: sticky;
            bottom: 0;
            background: linear-gradient(135deg, #2A2D3E 0%, #32364A 100%);
            padding: 1rem;
            border-radius: 15px;
            box-shadow: 0 -4px 20px rgba(0,0,0,0.2);
            border: 1px solid rgba(255,255,255,0.1);
            margin-top: 1rem;
            backdrop-filter: blur(10px);
        }
        
        /* Status indicators */
        .status-success {
            background: linear-gradient(135deg, #2A2D3E 0%, #32364A 100%);
            color: #4facfe;
            padding: 0.5rem 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            text-align: center;
            border: 1px solid rgba(79,172,254,0.3);
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #2A2D3E 0%, #32364A 100%);
            color: white;
            border: 1px solid rgba(255,255,255,0.1);
            padding: 0.5rem 2rem;
            border-radius: 25px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        /* Input field styling */
        .stTextInput > div > div > input {
            background: linear-gradient(135deg, #2A2D3E 0%, #32364A 100%);
            color: white;
            border-radius: 25px;
            border: 1px solid rgba(255,255,255,0.1);
            padding: 0.75rem 1.5rem;
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102,126,234,0.1);
        }
        
        /* Welcome message */
        .welcome-message {
            background: linear-gradient(135deg, #2A2D3E 0%, #32364A 100%);
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            margin: 2rem 0;
            color: white;
            border: 1px solid rgba(255,255,255,0.1);
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        
        /* Scrollbar styling */
        .chat-container::-webkit-scrollbar {
            width: 6px;
        }
        
        .chat-container::-webkit-scrollbar-track {
            background: #2A2D3E;
            border-radius: 10px;
        }
        
        .chat-container::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }
        
        .chat-container::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }

</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📊 Interactive Analytics Dashboard</h1>', unsafe_allow_html=True)

chat_file = "data/chat_log.csv"
operation_file = "data/operations_log.csv"

def clean_text_for_wordcloud(text):
    """Clean text for word cloud generation"""
    if pd.isna(text):
        return ""
    # Remove special characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    # Remove common stop words
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
    words = [word for word in text.split() if word not in stop_words and len(word) > 2]
    return ' '.join(words)

def emoji_helper(df,col):

    emojis = []
    for message in df[col]:
        emojis.extend([c for c in message if c in emoji.UNICODE_EMOJI['en']])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))),columns=['Emoji','Count'])

    return emoji_df


def create_wordcloud(text_data, title, colormap='viridis'):
    """Create a word cloud visualization"""
    if not text_data or len(text_data.strip()) == 0:
        return None
    
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white',
                         colormap=colormap,
                         max_words=100,
                         relative_scaling=0.5,
                         random_state=42).generate(text_data)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig

def analyze_chat_data(df):
    """Analyze chat log data"""
    st.markdown('<h2 class="sub-header">💬 Chat Analytics</h2>', unsafe_allow_html=True)
    
    # Second-wise Timeline with Sentiment (Plotly version)
    st.title("Second-wise Timeline with Sentiment (Interactive)")
    df_sentiment = sentiment.polarity_score(df,"User")
    df_timeline = df.copy().reset_index(drop=True)
    df_sentiment = df_sentiment.reset_index(drop=True)
    df_timeline = pd.concat([df_timeline, df_sentiment[['Negative', 'Neutral', 'Positive']]], axis=1)
    df_timeline['Timestamp'] = df_timeline['Timestamp'].astype(str)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_timeline['Timestamp'], y=df_timeline['Positive'], mode='lines+markers', name='Positive', line=dict(color='#00FF7F', width=3), marker=dict(size=8, symbol='circle')))
    fig.add_trace(go.Scatter(x=df_timeline['Timestamp'], y=df_timeline['Negative'], mode='lines+markers', name='Negative', line=dict(color='#FF4C4C', width=3), marker=dict(size=8, symbol='diamond')))
    fig.add_trace(go.Scatter(x=df_timeline['Timestamp'], y=df_timeline['Neutral'], mode='lines+markers', name='Neutral', line=dict(color='#A9A9A9', width=3, dash='dot'), marker=dict(size=8, symbol='square')))
    fig.update_layout(
        xaxis_title='Time (hh:mm:ss)',
        yaxis_title='Sentiment Score',
        title='<b>User Sentiment Over Conversation Timeline (Second-wise)</b>',
        legend_title='Sentiment',
        template='plotly_dark',
        hovermode='x unified',
        autosize=True,
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(family='Segoe UI, Arial', size=16, color='#FFF'),
        height=500,
        width=1100,
        plot_bgcolor='#18191A',
        paper_bgcolor='#18191A',
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#333', tickangle=45)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333', rangemode='tozero')
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

    st.title("Sentiment Analysis")
    df_sentiment=sentiment.polarity_score(df,"User")
    st.dataframe(df_sentiment, use_container_width=True)
    

    # Metrics
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(df)}</div>
                <div class="metric-label">💬 Total Conversations</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        avg_user_len = f"{df['User'].str.len().mean():.0f}"
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_user_len}</div>
                <div class="metric-label">👤 Avg User Message Length</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        avg_ai_len = f"{df['AI'].str.len().mean():.0f}"
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_ai_len}</div>
                <div class="metric-label">🤖 Avg AI Response Length</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        if 'Timestamp' in df.columns:
            date_range = len(pd.to_datetime(df['Timestamp'], errors='coerce').dt.date.unique())
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{date_range}</div>
                    <div class="metric-label">📅 Days of Activity</div>
                </div>
            """, unsafe_allow_html=True)
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["☁️ Word Clouds","📈 Message Trends", "📊 Message Analytics", "📅 Time Analysis"])
    
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # User word cloud
            user_text = ' '.join([clean_text_for_wordcloud(text) for text in df['User'].dropna()])
            if user_text:
                wordcloud_fig = create_wordcloud(user_text, "User Messages Word Cloud")
                if wordcloud_fig:
                    st.pyplot(wordcloud_fig)
                else:
                    st.info("Not enough text data for User word cloud")
            else:
                st.info("No user data available for word cloud")
        
        with col2:
            # AI word cloud
            ai_text = ' '.join([clean_text_for_wordcloud(text) for text in df['AI'].dropna()])
            if ai_text:
                wordcloud_fig = create_wordcloud(ai_text, "AI Responses Word Cloud", 'Oranges')
                if wordcloud_fig:
                    st.pyplot(wordcloud_fig)
                else:
                    st.info("Not enough text data for AI word cloud")
            else:
                st.info("No AI data available for word cloud")

    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Message length distribution
            fig = px.histogram(df, x=df['User'].str.len(), 
                             title="User Message Length Distribution",
                             labels={'x': 'Message Length (characters)', 'y': 'Count'},
                             color_discrete_sequence=['#1f77b4'])
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # AI response length distribution
            fig = px.histogram(df, x=df['AI'].str.len(), 
                             title="AI Response Length Distribution",
                             labels={'x': 'Response Length (characters)', 'y': 'Count'},
                             color_discrete_sequence=['#ff7f0e'])
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Message length comparison
            comparison_data = pd.DataFrame({
                'Type': ['User Messages', 'AI Responses'],
                'Average Length': [df['User'].str.len().mean(), df['AI'].str.len().mean()]
            })
            fig = px.bar(comparison_data, x='Type', y='Average Length',
                        title="Average Message Length Comparison",
                        color='Type',
                        color_discrete_sequence=['#1f77b4', '#ff7f0e'])
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Scatter plot of message lengths
            fig = px.scatter(df, x=df['User'].str.len(), y=df['AI'].str.len(),
                           title="User vs AI Message Length Correlation",
                           labels={'x': 'User Message Length', 'y': 'AI Response Length'},
                           opacity=0.6,
                           color_discrete_sequence=['#2ca02c'])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        if 'Timestamp' in df.columns:
            try:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
                df['Date'] = df['Timestamp'].dt.date
                df['Hour'] = df['Timestamp'].dt.hour
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Messages by date
                    daily_counts = df['Date'].value_counts().sort_index()
                    fig = px.line(x=daily_counts.index, y=daily_counts.values,
                                title="Messages Over Time",
                                labels={'x': 'Date', 'y': 'Number of Messages'})
                    fig.update_traces(line_color='#1f77b4')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Messages by hour
                    hourly_counts = df['Hour'].value_counts().sort_index()
                    fig = px.bar(x=hourly_counts.index, y=hourly_counts.values,
                               title="Messages by Hour of Day",
                               labels={'x': 'Hour', 'y': 'Number of Messages'},
                               color_discrete_sequence=['#ff7f0e'])
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.warning(f"Could not parse timestamp data: {e}")
        else:
            st.info("No timestamp data available for time analysis")
    
    
    #Emoji-Analysis:-
    emoji_df = emoji_helper(df,"User")
    st.title("Emoji Analysis")
    col1,col2 = st.columns(2)
    with col1:
        st.dataframe(emoji_df)
    with col2:
        if not emoji_df.empty:
            fig_emoji = px.bar(
                emoji_df.head(10),
                x='Count',
                y='Emoji',
                orientation='h',
                color='Count',
                color_continuous_scale='sunset',
                title='Top 10 Emojis Used'
            )
            fig_emoji.update_layout(
                template='plotly_dark',
                height=400,
                font=dict(color='#FFF'),
                plot_bgcolor='#18191A',
                paper_bgcolor='#18191A',
            )
            st.plotly_chart(fig_emoji, use_container_width=True)
        else:
            st.write('No emojis found for this user.')

def analyze_operation_data(df):
    """Analyze operation log data"""
    st.markdown('<h2 class="sub-header">⚙️ Operations Analytics</h2>', unsafe_allow_html=True)
    
    # Metrics
    # Metrics for Operations
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(df)}</div>
                <div class="metric-label">⚙️ Total Operations</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        if 'customer_name' in df.columns:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{df['customer_name'].nunique()}</div>
                    <div class="metric-label">👥 Unique Customers</div>
                </div>
            """, unsafe_allow_html=True)

    with col3:
        if 'operation_type' in df.columns:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{df['operation_type'].nunique()}</div>
                    <div class="metric-label">🔄 Operation Types</div>
                </div>
            """, unsafe_allow_html=True)

    with col4:
        if 'status' in df.columns:
            success_rate = (df['status'].str.lower().str.contains('success|complete|done', na=False).sum() / len(df)) * 100
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{success_rate:.1f}%</div>
                    <div class="metric-label">📈 Success Rate</div>
                </div>
            """, unsafe_allow_html=True)
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Operation Types", "📈 Status Analysis", "👥 Customer Insights", "📅 Time Trends"])
    
    with tab1:
        if 'operation_type' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Operation type counts
                op_counts = df['operation_type'].value_counts()
                fig = px.pie(values=op_counts.values, names=op_counts.index,
                           title="Operation Type Distribution",
                           color_discrete_sequence=px.colors.qualitative.Set3)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Operation type bar chart
                fig = px.bar(x=op_counts.index, y=op_counts.values,
                           title="Operation Type Counts",
                           labels={'x': 'Operation Type', 'y': 'Count'},
                           color=op_counts.values,
                           color_continuous_scale='viridis')
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No 'operation_type' column found in the data")
    
    with tab2:
        if 'status' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Status distribution
                status_counts = df['status'].value_counts()
                fig = px.pie(values=status_counts.values, names=status_counts.index,
                           title="Status Distribution",
                           color_discrete_sequence=px.colors.qualitative.Pastel)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Status by operation type
                if 'operation_type' in df.columns:
                    status_op = df.groupby(['operation_type', 'status']).size().reset_index(name='count')
                    fig = px.bar(status_op, x='operation_type', y='count', color='status',
                               title="Status by Operation Type",
                               labels={'count': 'Count', 'operation_type': 'Operation Type'})
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No 'status' column found in the data")
    
    with tab3:
        if 'customer_name' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Top customers by operation count
                customer_counts = df['customer_name'].value_counts().head(10)
                fig = px.bar(x=customer_counts.values, y=customer_counts.index,
                           title="Top 10 Customers by Operations",
                           labels={'x': 'Number of Operations', 'y': 'Customer'},
                           orientation='h',
                           color_discrete_sequence=['#2ca02c'])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Customer operation distribution
                customer_op_dist = df['customer_name'].value_counts()
                fig = px.histogram(x=customer_op_dist.values,
                                 title="Customer Operation Count Distribution",
                                 labels={'x': 'Operations per Customer', 'y': 'Number of Customers'},
                                 color_discrete_sequence=['#d62728'])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No 'customer_name' column found in the data")
    
    with tab4:
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df['Date'] = df['timestamp'].dt.date
                df['Hour'] = df['timestamp'].dt.hour
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Operations by date
                    daily_ops = df['Date'].value_counts().sort_index()
                    fig = px.line(x=daily_ops.index, y=daily_ops.values,
                                title="Operations Over Time",
                                labels={'x': 'Date', 'y': 'Number of Operations'})
                    fig.update_traces(line_color='#ff7f0e')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Operations by hour
                    hourly_ops = df['Hour'].value_counts().sort_index()
                    fig = px.bar(x=hourly_ops.index, y=hourly_ops.values,
                               title="Operations by Hour of Day",
                               labels={'x': 'Hour', 'y': 'Number of Operations'},
                               color_discrete_sequence=['#9467bd'])
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.warning(f"Could not parse timestamp data: {e}")
        else:
            st.info("No timestamp data available for time analysis")

# Main app logic
if chat_file is not None or operation_file is not None:
    
    # Process chat data
    if chat_file is not None:
        try:
            chat_df = pd.read_csv(chat_file)
            st.success(f"✅ Chat log loaded successfully! ({len(chat_df)} records)")
            analyze_chat_data(chat_df)
            st.markdown("---")
        except Exception as e:
            st.error(f"❌ Error loading chat file: {e}")
    
    # Process operation data
    if operation_file is not None:
        try:
            ops_df = pd.read_csv(operation_file)
            st.success(f"✅ Operation log loaded successfully! ({len(ops_df)} records)")
            analyze_operation_data(ops_df)
        except Exception as e:
            st.error(f"❌ Error loading operation file: {e}")

else:
    # Welcome message
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin: 2rem 0;">
        <h2>🚀 Welcome to Your Analytics Dashboard!</h2>
        <p style="font-size: 1.2rem; margin-top: 1rem;">
            Upload your CSV files using the sidebar to get started with powerful analytics and visualizations.
        </p>
        <div style="margin-top: 2rem; display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
            <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 8px; min-width: 200px;">
                <h4>📈 Chat Analytics</h4>
                <p>Word clouds, message trends, time analysis</p>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 8px; min-width: 200px;">
                <h4>⚙️ Operation Analytics</h4>
                <p>Status tracking, customer insights, performance metrics</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    st.markdown("### ✨ Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📊 Interactive Visualizations**
        - Dynamic Plotly charts
        - Hover effects and zoom
        - Real-time filtering
        """)
    
    with col2:
        st.markdown("""
        **☁️ Word Cloud Analysis**
        - User message patterns
        - AI response trends
        - Smart text cleaning
        """)
    
    with col3:
        st.markdown("""
        **📈 Comprehensive Analytics**
        - Time-based trends
        - Performance metrics
        - Customer insights
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    Built with ❤️ using Streamlit, Plotly, and WordCloud
</div>
""", unsafe_allow_html=True)