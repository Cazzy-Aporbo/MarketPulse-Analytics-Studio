"""
MarketPulse Dashboard: Visual Symphony of Market Intelligence
Cazandra Aporbo, MS
May 2025

This dashboard isn't just data visualization. It's a carefully orchestrated
visual experience that transforms complex market dynamics into intuitive,
actionable insights. Every color, transition, and layout decision serves
a purpose: making the invisible patterns of markets visible.

I built this after studying how professional traders actually use information.
They don't need more data. They need the right data, presented at the right
time, in a way that enables split-second decisions.

The pastel palette isn't just aesthetic. It reduces eye strain during long
trading sessions while maintaining enough contrast for quick pattern recognition.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
import logging

# Configure Streamlit for optimal performance
st.set_page_config(
    page_title="MarketPulse Analytics Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "MarketPulse Analytics Studio | Built by Cazandra Aporbo, MS"
    }
)

# Custom logging that doesn't clutter the UI
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VisualTheme:
    """
    The visual identity of MarketPulse.
    
    I spent weeks perfecting this palette. Each color was chosen for specific
    psychological and functional reasons. Pastels reduce cognitive load while
    maintaining visual hierarchy. The result: you can stare at this dashboard
    for hours without fatigue.
    """
    
    # Primary palette - carefully chosen pastels
    colors = {
        'sage': '#87CEEB',        # Sky blue - primary actions, bullish signals
        'lavender': '#E6E6FA',    # Soft purple - secondary elements
        'rose': '#FFE4E1',        # Misty rose - warnings, bearish signals
        'mint': '#98FB98',        # Pale green - success, profits
        'peach': '#FFDAB9',       # Peach puff - attention, highlights
        'pearl': '#F8F8FF',       # Ghost white - backgrounds
        'dust': '#F5F5F5',        # White smoke - subtle backgrounds
        'mist': '#F0F8FF',        # Alice blue - hover states
        'sand': '#FAF0E6',        # Linen - neutral elements
        'coral': '#F08080',       # Light coral - errors, stops
        'foam': '#B0E0E6',        # Powder blue - information
        'lilac': '#DDA0DD',       # Plum - special indicators
        'butter': '#FFFFE0',      # Light yellow - alerts
        'slate': '#708090',       # Slate gray - text
        'charcoal': '#36454F'     # Charcoal - headers
    }
    
    # Gradients for depth and sophistication
    gradients = {
        'dawn': ['#FFE4E1', '#E6E6FA', '#87CEEB'],  # Rose to lavender to sky
        'dusk': ['#87CEEB', '#DDA0DD', '#FFE4E1'],  # Sky to plum to rose
        'ocean': ['#E0FFFF', '#B0E0E6', '#87CEEB'], # Light cyan to powder to sky
        'meadow': ['#F0FFF0', '#98FB98', '#90EE90'], # Honeydew to mint to light green
        'sunset': ['#FFFFE0', '#FFDAB9', '#F08080']  # Yellow to peach to coral
    }
    
    # Typography system
    fonts = {
        'primary': '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, Helvetica, Arial',
        'mono': '"SF Mono", "Monaco", "Cascadia Code", "Roboto Mono", monospace',
        'display': '"SF Pro Display", -apple-system, BlinkMacSystemFont, sans-serif'
    }
    
    # Layout specifications
    layout = {
        'card_radius': '12px',
        'button_radius': '8px',
        'padding_small': '8px',
        'padding_medium': '16px',
        'padding_large': '24px',
        'transition': 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',  # Smooth animations
        'shadow_small': '0 2px 4px rgba(0,0,0,0.05)',
        'shadow_medium': '0 4px 8px rgba(0,0,0,0.08)',
        'shadow_large': '0 8px 16px rgba(0,0,0,0.12)'
    }


class ChartFactory:
    """
    Factory for creating consistent, beautiful charts.
    
    Every chart follows the same design language. This isn't just about
    aesthetics - consistency reduces cognitive load and speeds up pattern
    recognition. When every chart works the same way, traders can focus
    on the data, not the interface.
    """
    
    def __init__(self, theme: VisualTheme):
        self.theme = theme
        self.default_height = 400
        self.animation_duration = 750
        
    def create_price_chart(self, data: pd.DataFrame, 
                          title: str = "Price Action") -> go.Figure:
        """
        The main price chart. This is what traders stare at all day.
        I use candlesticks because they encode four data points elegantly.
        """
        
        fig = go.Figure()
        
        # Candlestick with custom colors
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            increasing=dict(
                line=dict(color=self.theme.colors['mint'], width=1),
                fillcolor=self.theme.colors['mint']
            ),
            decreasing=dict(
                line=dict(color=self.theme.colors['rose'], width=1),
                fillcolor=self.theme.colors['rose']
            ),
            whiskerwidth=0.8,
            opacity=0.9
        ))
        
        # Add volume bars with transparency
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color=self.theme.colors['foam'],
            opacity=0.3,
            yaxis='y2'
        ))
        
        # Layout with dual axis
        fig.update_layout(
            title={
                'text': title,
                'font': {'size': 20, 'color': self.theme.colors['charcoal']}
            },
            template='plotly_white',
            height=self.default_height,
            xaxis=dict(
                rangeslider=dict(visible=False),
                gridcolor=self.theme.colors['dust'],
                showgrid=True,
                gridwidth=0.5
            ),
            yaxis=dict(
                title='Price',
                side='right',
                gridcolor=self.theme.colors['dust'],
                showgrid=True,
                gridwidth=0.5
            ),
            yaxis2=dict(
                title='Volume',
                overlaying='y',
                side='left',
                showgrid=False,
                range=[0, data['Volume'].max() * 4]  # Compress volume bars
            ),
            hovermode='x unified',
            paper_bgcolor=self.theme.colors['pearl'],
            plot_bgcolor=self.theme.colors['pearl'],
            font=dict(family=self.theme.fonts['primary']),
            margin=dict(l=60, r=60, t=80, b=60),
            transition={'duration': self.animation_duration}
        )
        
        return fig
    
    def create_sentiment_gauge(self, sentiment_score: float,
                              title: str = "Market Sentiment") -> go.Figure:
        """
        A gauge that shows market sentiment at a glance.
        Inspired by classic analog meters but with modern aesthetics.
        """
        
        # Determine color based on sentiment
        if sentiment_score > 0.3:
            gauge_color = self.theme.colors['mint']
            status = "Bullish"
        elif sentiment_score < -0.3:
            gauge_color = self.theme.colors['rose']
            status = "Bearish"
        else:
            gauge_color = self.theme.colors['foam']
            status = "Neutral"
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=sentiment_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 18}},
            number={'suffix': "", 'font': {'size': 30}},
            gauge={
                'axis': {'range': [-1, 1], 'tickwidth': 1},
                'bar': {'color': gauge_color, 'thickness': 0.8},
                'bgcolor': self.theme.colors['dust'],
                'borderwidth': 2,
                'bordercolor': self.theme.colors['slate'],
                'steps': [
                    {'range': [-1, -0.3], 'color': self.theme.colors['rose']},
                    {'range': [-0.3, 0.3], 'color': self.theme.colors['mist']},
                    {'range': [0.3, 1], 'color': self.theme.colors['mint']}
                ],
                'threshold': {
                    'line': {'color': self.theme.colors['charcoal'], 'width': 3},
                    'thickness': 0.75,
                    'value': sentiment_score
                }
            }
        ))
        
        fig.update_layout(
            height=250,
            paper_bgcolor=self.theme.colors['pearl'],
            font={'color': self.theme.colors['slate'], 
                  'family': self.theme.fonts['primary']},
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        # Add status annotation
        fig.add_annotation(
            text=f"<b>{status}</b>",
            x=0.5, y=0.25,
            showarrow=False,
            font=dict(size=16, color=gauge_color)
        )
        
        return fig
    
    def create_heatmap(self, correlation_matrix: pd.DataFrame,
                      title: str = "Feature Correlations") -> go.Figure:
        """
        Correlation heatmap with intelligent color scaling.
        Red-blue diverging palette helps identify relationships instantly.
        """
        
        # Custom colorscale using pastels
        colorscale = [
            [0.0, self.theme.colors['rose']],
            [0.25, self.theme.colors['peach']],
            [0.5, self.theme.colors['pearl']],
            [0.75, self.theme.colors['mist']],
            [1.0, self.theme.colors['sage']]
        ]
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale=colorscale,
            colorbar=dict(
                thickness=15,
                len=0.7,
                bgcolor=self.theme.colors['pearl'],
                bordercolor=self.theme.colors['slate'],
                borderwidth=1
            ),
            text=correlation_matrix.values,
            texttemplate='%{text:.2f}',
            textfont=dict(size=10),
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': title,
                'font': {'size': 18, 'color': self.theme.colors['charcoal']}
            },
            height=400,
            paper_bgcolor=self.theme.colors['pearl'],
            plot_bgcolor=self.theme.colors['pearl'],
            font=dict(family=self.theme.fonts['primary']),
            margin=dict(l=80, r=80, t=80, b=80)
        )
        
        return fig
    
    def create_prediction_timeline(self, predictions: pd.DataFrame,
                                  title: str = "Prediction Accuracy") -> go.Figure:
        """
        Timeline showing prediction accuracy over time.
        Green for correct, red for wrong, with running accuracy overlay.
        """
        
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("Predictions vs Reality", "Cumulative Accuracy")
        )
        
        # Scatter plot for predictions
        correct_mask = predictions['correct'] == 1
        wrong_mask = predictions['correct'] == 0
        
        # Correct predictions
        fig.add_trace(
            go.Scatter(
                x=predictions.index[correct_mask],
                y=predictions['price'][correct_mask],
                mode='markers',
                name='Correct',
                marker=dict(
                    color=self.theme.colors['mint'],
                    size=8,
                    symbol='circle',
                    line=dict(color=self.theme.colors['mint'], width=1)
                )
            ),
            row=1, col=1
        )
        
        # Wrong predictions
        fig.add_trace(
            go.Scatter(
                x=predictions.index[wrong_mask],
                y=predictions['price'][wrong_mask],
                mode='markers',
                name='Wrong',
                marker=dict(
                    color=self.theme.colors['rose'],
                    size=8,
                    symbol='x',
                    line=dict(color=self.theme.colors['rose'], width=1)
                )
            ),
            row=1, col=1
        )
        
        # Price line
        fig.add_trace(
            go.Scatter(
                x=predictions.index,
                y=predictions['price'],
                mode='lines',
                name='Price',
                line=dict(color=self.theme.colors['slate'], width=1, dash='dot'),
                opacity=0.5
            ),
            row=1, col=1
        )
        
        # Cumulative accuracy
        cumulative_accuracy = predictions['correct'].expanding().mean()
        fig.add_trace(
            go.Scatter(
                x=predictions.index,
                y=cumulative_accuracy,
                mode='lines',
                name='Accuracy',
                line=dict(color=self.theme.colors['lilac'], width=2),
                fill='tozeroy',
                fillcolor=self.theme.colors['lavender'] + '30'  # Add transparency
            ),
            row=2, col=1
        )
        
        # 50% reference line
        fig.add_hline(
            y=0.5,
            line_dash="dash",
            line_color=self.theme.colors['slate'],
            opacity=0.5,
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': title,
                'font': {'size': 20, 'color': self.theme.colors['charcoal']}
            },
            height=600,
            showlegend=True,
            paper_bgcolor=self.theme.colors['pearl'],
            plot_bgcolor=self.theme.colors['pearl'],
            hovermode='x unified',
            font=dict(family=self.theme.fonts['primary']),
            margin=dict(l=60, r=60, t=100, b=60)
        )
        
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", tickformat=".0%", row=2, col=1)
        
        return fig
    
    def create_feature_importance(self, importance_data: pd.DataFrame,
                                 title: str = "Feature Importance") -> go.Figure:
        """
        Horizontal bar chart showing feature importance.
        Gradient coloring from least to most important.
        """
        
        # Sort by importance
        importance_data = importance_data.sort_values('importance', ascending=True)
        
        # Create gradient colors based on importance
        n_features = len(importance_data)
        colors = px.colors.sample_colorscale(
            [[0, self.theme.colors['mist']],
             [0.5, self.theme.colors['foam']],
             [1, self.theme.colors['sage']]],
            n_features
        )
        
        fig = go.Figure(go.Bar(
            x=importance_data['importance'],
            y=importance_data['feature'],
            orientation='h',
            marker=dict(
                color=importance_data['importance'],
                colorscale=[
                    [0, self.theme.colors['dust']],
                    [0.5, self.theme.colors['lavender']],
                    [1, self.theme.colors['lilac']]
                ],
                showscale=False
            ),
            text=importance_data['importance'].round(3),
            textposition='outside',
            textfont=dict(size=10)
        ))
        
        fig.update_layout(
            title={
                'text': title,
                'font': {'size': 18, 'color': self.theme.colors['charcoal']}
            },
            height=max(400, n_features * 25),  # Dynamic height
            paper_bgcolor=self.theme.colors['pearl'],
            plot_bgcolor=self.theme.colors['pearl'],
            font=dict(family=self.theme.fonts['primary']),
            margin=dict(l=150, r=60, t=80, b=60),
            xaxis=dict(
                title='Importance Score',
                gridcolor=self.theme.colors['dust'],
                showgrid=True,
                gridwidth=0.5
            ),
            yaxis=dict(
                showgrid=False
            )
        )
        
        return fig
    
    def create_risk_radar(self, risk_metrics: Dict[str, float],
                         title: str = "Risk Profile") -> go.Figure:
        """
        Radar chart showing multiple risk dimensions.
        Helps visualize overall risk exposure at a glance.
        """
        
        categories = list(risk_metrics.keys())
        values = list(risk_metrics.values())
        
        # Close the radar chart
        categories.append(categories[0])
        values.append(values[0])
        
        fig = go.Figure(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            fillcolor=self.theme.colors['lavender'] + '40',  # Transparent fill
            line=dict(color=self.theme.colors['lilac'], width=2),
            marker=dict(
                color=self.theme.colors['lilac'],
                size=8
            ),
            text=[f'{v:.2f}' for v in values],
            hovertemplate='%{theta}: %{r:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    gridcolor=self.theme.colors['dust'],
                    gridwidth=1
                ),
                angularaxis=dict(
                    gridcolor=self.theme.colors['dust'],
                    gridwidth=1
                ),
                bgcolor=self.theme.colors['pearl']
            ),
            title={
                'text': title,
                'font': {'size': 18, 'color': self.theme.colors['charcoal']}
            },
            height=400,
            paper_bgcolor=self.theme.colors['pearl'],
            font=dict(family=self.theme.fonts['primary']),
            margin=dict(l=80, r=80, t=100, b=80)
        )
        
        return fig


class DashboardEngine:
    """
    The main dashboard orchestrator.
    
    This class manages the entire dashboard lifecycle: data flow, state management,
    real-time updates, and user interactions. It's the conductor of our visual
    symphony, ensuring every component plays in harmony.
    """
    
    def __init__(self):
        self.theme = VisualTheme()
        self.chart_factory = ChartFactory(self.theme)
        
        # Initialize session state
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.last_update = datetime.now()
            st.session_state.selected_ticker = 'AAPL'
            st.session_state.time_range = '3mo'
            st.session_state.refresh_rate = 60  # seconds
            st.session_state.dark_mode = False
            
    def inject_custom_css(self):
        """
        Inject custom CSS for a polished look.
        This is where the dashboard gets its professional feel.
        """
        
        css = f"""
        <style>
        /* Import fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global styles */
        .stApp {{
            background: linear-gradient(135deg, {self.theme.colors['pearl']} 0%, {self.theme.colors['mist']} 100%);
            font-family: {self.theme.fonts['primary']};
        }}
        
        /* Headers */
        h1, h2, h3 {{
            color: {self.theme.colors['charcoal']};
            font-weight: 600;
            letter-spacing: -0.02em;
        }}
        
        /* Cards */
        .metric-card {{
            background: {self.theme.colors['pearl']};
            border-radius: {self.theme.layout['card_radius']};
            padding: {self.theme.layout['padding_large']};
            box-shadow: {self.theme.layout['shadow_medium']};
            transition: {self.theme.layout['transition']};
            border: 1px solid {self.theme.colors['dust']};
        }}
        
        .metric-card:hover {{
            box-shadow: {self.theme.layout['shadow_large']};
            transform: translateY(-2px);
        }}
        
        /* Metrics */
        [data-testid="metric-container"] {{
            background: {self.theme.colors['pearl']};
            padding: {self.theme.layout['padding_medium']};
            border-radius: {self.theme.layout['button_radius']};
            box-shadow: {self.theme.layout['shadow_small']};
        }}
        
        /* Buttons */
        .stButton > button {{
            background: linear-gradient(135deg, {self.theme.colors['sage']} 0%, {self.theme.colors['foam']} 100%);
            color: white;
            border: none;
            border-radius: {self.theme.layout['button_radius']};
            padding: 8px 24px;
            font-weight: 500;
            transition: {self.theme.layout['transition']};
            box-shadow: {self.theme.layout['shadow_small']};
        }}
        
        .stButton > button:hover {{
            box-shadow: {self.theme.layout['shadow_medium']};
            transform: translateY(-1px);
        }}
        
        /* Sidebar */
        .sidebar .sidebar-content {{
            background: {self.theme.colors['pearl']};
        }}
        
        /* Select boxes */
        .stSelectbox > div > div {{
            background: {self.theme.colors['pearl']};
            border-radius: {self.theme.layout['button_radius']};
            border: 1px solid {self.theme.colors['dust']};
        }}
        
        /* Info boxes */
        .stAlert {{
            background: {self.theme.colors['mist']};
            border-left: 4px solid {self.theme.colors['foam']};
            border-radius: {self.theme.layout['button_radius']};
        }}
        
        /* Tables */
        .dataframe {{
            background: {self.theme.colors['pearl']};
            border: none;
        }}
        
        .dataframe th {{
            background: {self.theme.colors['dust']};
            color: {self.theme.colors['charcoal']};
            font-weight: 600;
        }}
        
        .dataframe td {{
            background: {self.theme.colors['pearl']};
            color: {self.theme.colors['slate']};
        }}
        
        /* Animations */
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .fade-in {{
            animation: fadeIn 0.5s ease-out;
        }}
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {{
            width: 10px;
            height: 10px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: {self.theme.colors['dust']};
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: {self.theme.colors['lavender']};
            border-radius: 5px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: {self.theme.colors['lilac']};
        }}
        </style>
        """
        
        st.markdown(css, unsafe_allow_html=True)
    
    def render_header(self):
        """
        Render the dashboard header with title and key metrics.
        First impressions matter.
        """
        
        col1, col2, col3 = st.columns([2, 3, 1])
        
        with col1:
            st.markdown(
                "<h1 style='margin: 0;'>MarketPulse</h1>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<p style='color: {self.theme.colors['slate']}; margin: 0;'>"
                f"Analytics Studio | Real-time Market Intelligence</p>",
                unsafe_allow_html=True
            )
        
        with col2:
            # Status indicators
            status_col1, status_col2, status_col3 = st.columns(3)
            
            with status_col1:
                market_status = "Open" if datetime.now().hour in range(9, 16) else "Closed"
                status_color = self.theme.colors['mint'] if market_status == "Open" else self.theme.colors['coral']
                st.markdown(
                    f"<div style='text-align: center; padding: 8px; "
                    f"background: {status_color}20; border-radius: 8px; "
                    f"border: 1px solid {status_color};'>"
                    f"<b>Market:</b> {market_status}</div>",
                    unsafe_allow_html=True
                )
            
            with status_col2:
                st.markdown(
                    f"<div style='text-align: center; padding: 8px; "
                    f"background: {self.theme.colors['foam']}20; border-radius: 8px; "
                    f"border: 1px solid {self.theme.colors['foam']};'>"
                    f"<b>Models:</b> Active</div>",
                    unsafe_allow_html=True
                )
            
            with status_col3:
                latency = np.random.randint(10, 50)  # Mock latency
                latency_color = self.theme.colors['mint'] if latency < 30 else self.theme.colors['butter']
                st.markdown(
                    f"<div style='text-align: center; padding: 8px; "
                    f"background: {latency_color}20; border-radius: 8px; "
                    f"border: 1px solid {latency_color};'>"
                    f"<b>Latency:</b> {latency}ms</div>",
                    unsafe_allow_html=True
                )
        
        with col3:
            # Last update time
            time_since = (datetime.now() - st.session_state.last_update).seconds
            st.markdown(
                f"<div style='text-align: right; color: {self.theme.colors['slate']};'>"
                f"Updated {time_since}s ago<br>"
                f"<small>{datetime.now().strftime('%H:%M:%S')}</small></div>",
                unsafe_allow_html=True
            )
    
    def render_sidebar(self):
        """
        Render the control sidebar.
        All the knobs and switches to control the dashboard.
        """
        
        with st.sidebar:
            st.markdown(
                f"<h2 style='color: {self.theme.colors['charcoal']};'>Control Panel</h2>",
                unsafe_allow_html=True
            )
            
            # Ticker selection
            st.selectbox(
                "Stock Ticker",
                options=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA'],
                key='selected_ticker'
            )
            
            # Time range
            st.select_slider(
                "Time Range",
                options=['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y'],
                value='3mo',
                key='time_range'
            )
            
            # Model configuration
            st.markdown("### Model Settings")
            
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.5,
                max_value=0.95,
                value=0.7,
                step=0.05,
                help="Minimum confidence for predictions"
            )
            
            risk_tolerance = st.slider(
                "Risk Tolerance",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="0 = Conservative, 1 = Aggressive"
            )
            
            # Advanced settings
            with st.expander("Advanced Settings"):
                st.number_input(
                    "Refresh Rate (seconds)",
                    min_value=10,
                    max_value=300,
                    value=60,
                    key='refresh_rate'
                )
                
                st.checkbox(
                    "Show Technical Indicators",
                    value=True,
                    key='show_technical'
                )
                
                st.checkbox(
                    "Show Sentiment Analysis",
                    value=True,
                    key='show_sentiment'
                )
                
                st.checkbox(
                    "Enable Predictions",
                    value=True,
                    key='enable_predictions'
                )
            
            # Info section
            st.markdown("---")
            st.markdown(
                f"<div style='padding: 16px; background: {self.theme.colors['mist']}; "
                f"border-radius: 8px; border-left: 4px solid {self.theme.colors['foam']};'>"
                f"<b>About MarketPulse</b><br>"
                f"<small>Real-time market sentiment analysis powered by ensemble ML. "
                f"Built by Cazandra Aporbo, MS</small></div>",
                unsafe_allow_html=True
            )
    
    def render_main_dashboard(self):
        """
        Render the main dashboard content.
        This is where all the magic happens.
        """
        
        # Generate mock data for demonstration
        dates = pd.date_range(end=datetime.now(), periods=100)
        mock_data = pd.DataFrame({
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 102,
            'Low': np.random.randn(100).cumsum() + 98,
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        # Ensure OHLC relationships
        mock_data['High'] = mock_data[['Open', 'High', 'Low', 'Close']].max(axis=1)
        mock_data['Low'] = mock_data[['Open', 'High', 'Low', 'Close']].min(axis=1)
        
        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Current Price",
                f"${mock_data['Close'].iloc[-1]:.2f}",
                f"{np.random.uniform(-2, 2):.2f}%"
            )
        
        with col2:
            st.metric(
                "Volume",
                f"{mock_data['Volume'].iloc[-1]/1e6:.1f}M",
                f"{np.random.uniform(-10, 10):.1f}%"
            )
        
        with col3:
            st.metric(
                "Sentiment",
                f"{np.random.uniform(-1, 1):.3f}",
                "Bullish" if np.random.random() > 0.5 else "Bearish"
            )
        
        with col4:
            st.metric(
                "Model Signal",
                "BUY" if np.random.random() > 0.5 else "HOLD",
                f"{np.random.uniform(0.5, 0.9):.1%} conf"
            )
        
        with col5:
            st.metric(
                "Risk Score",
                f"{np.random.uniform(1, 10):.1f}/10",
                "Low" if np.random.random() > 0.5 else "Medium"
            )
        
        # Main charts
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Price Analysis", "Sentiment Dashboard", "Model Performance", "Risk Analytics"]
        )
        
        with tab1:
            # Price chart
            st.plotly_chart(
                self.chart_factory.create_price_chart(mock_data),
                use_container_width=True
            )
            
            # Technical indicators
            if st.session_state.get('show_technical', True):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Mock RSI chart
                    rsi_data = pd.DataFrame({
                        'RSI': np.random.uniform(30, 70, 100)
                    }, index=dates)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=rsi_data.index,
                        y=rsi_data['RSI'],
                        mode='lines',
                        line=dict(color=self.theme.colors['lilac'], width=2),
                        fill='tozeroy',
                        fillcolor=self.theme.colors['lavender'] + '20'
                    ))
                    fig.add_hline(y=70, line_dash="dash", line_color=self.theme.colors['coral'])
                    fig.add_hline(y=30, line_dash="dash", line_color=self.theme.colors['mint'])
                    fig.update_layout(
                        title="RSI Indicator",
                        height=300,
                        paper_bgcolor=self.theme.colors['pearl']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Mock MACD
                    macd_data = pd.DataFrame({
                        'MACD': np.random.randn(100).cumsum(),
                        'Signal': np.random.randn(100).cumsum()
                    }, index=dates)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=macd_data.index,
                        y=macd_data['MACD'],
                        name='MACD',
                        line=dict(color=self.theme.colors['foam'], width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=macd_data.index,
                        y=macd_data['Signal'],
                        name='Signal',
                        line=dict(color=self.theme.colors['coral'], width=2)
                    ))
                    fig.update_layout(
                        title="MACD",
                        height=300,
                        paper_bgcolor=self.theme.colors['pearl']
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Sentiment analysis
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Sentiment gauge
                sentiment_score = np.random.uniform(-1, 1)
                st.plotly_chart(
                    self.chart_factory.create_sentiment_gauge(sentiment_score),
                    use_container_width=True
                )
                
                # News sentiment breakdown
                st.markdown(
                    f"<div style='padding: 16px; background: {self.theme.colors['dust']}; "
                    f"border-radius: 8px; margin-top: 16px;'>"
                    f"<b>Recent News Sentiment</b><br>"
                    f"<div style='margin-top: 8px;'>"
                    f"<span style='color: {self.theme.colors['mint']};'>Positive: 45%</span><br>"
                    f"<span style='color: {self.theme.colors['rose']};'>Negative: 30%</span><br>"
                    f"<span style='color: {self.theme.colors['foam']};'>Neutral: 25%</span>"
                    f"</div></div>",
                    unsafe_allow_html=True
                )
            
            with col2:
                # Sentiment timeline
                sentiment_timeline = pd.DataFrame({
                    'sentiment': np.random.uniform(-1, 1, 100),
                    'volume': np.random.uniform(0, 100, 100)
                }, index=dates)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=sentiment_timeline.index,
                    y=sentiment_timeline['sentiment'],
                    mode='lines',
                    line=dict(
                        color=self.theme.colors['lilac'],
                        width=2
                    ),
                    fill='tozeroy',
                    fillcolor=self.theme.colors['lavender'] + '20',
                    name='Sentiment'
                ))
                fig.update_layout(
                    title="Sentiment Timeline",
                    height=400,
                    paper_bgcolor=self.theme.colors['pearl']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Model performance metrics
            predictions_df = pd.DataFrame({
                'price': mock_data['Close'].values,
                'correct': np.random.choice([0, 1], size=100, p=[0.3, 0.7])
            }, index=dates)
            
            st.plotly_chart(
                self.chart_factory.create_prediction_timeline(predictions_df),
                use_container_width=True
            )
            
            # Performance metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<h4 style='color: {self.theme.colors['charcoal']};'>Model Accuracy</h4>"
                    f"<h2 style='color: {self.theme.colors['mint']};'>67.3%</h2>"
                    f"<small>Last 100 predictions</small></div>",
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<h4 style='color: {self.theme.colors['charcoal']};'>Win Rate</h4>"
                    f"<h2 style='color: {self.theme.colors['foam']};'>58.2%</h2>"
                    f"<small>Profitable trades</small></div>",
                    unsafe_allow_html=True
                )
            
            with col3:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<h4 style='color: {self.theme.colors['charcoal']};'>Sharpe Ratio</h4>"
                    f"<h2 style='color: {self.theme.colors['lilac']};'>1.84</h2>"
                    f"<small>Risk-adjusted returns</small></div>",
                    unsafe_allow_html=True
                )
        
        with tab4:
            # Risk analytics
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk radar
                risk_metrics = {
                    'Market Risk': 0.7,
                    'Volatility': 0.5,
                    'Liquidity': 0.3,
                    'Sentiment': 0.6,
                    'Technical': 0.4,
                    'Model Confidence': 0.8
                }
                st.plotly_chart(
                    self.chart_factory.create_risk_radar(risk_metrics),
                    use_container_width=True
                )
            
            with col2:
                # Feature importance
                importance_df = pd.DataFrame({
                    'feature': ['Sentiment', 'Volume', 'RSI', 'MACD', 'Price MA', 'Volatility'],
                    'importance': np.random.uniform(0.1, 1, 6)
                })
                st.plotly_chart(
                    self.chart_factory.create_feature_importance(importance_df),
                    use_container_width=True
                )
    
    def run(self):
        """
        Main execution loop for the dashboard.
        This brings everything together.
        """
        
        # Apply custom styling
        self.inject_custom_css()
        
        # Render components
        self.render_header()
        self.render_sidebar()
        
        # Main content area
        self.render_main_dashboard()
        
        # Auto-refresh logic
        if st.session_state.get('auto_refresh', False):
            time.sleep(st.session_state.refresh_rate)
            st.experimental_rerun()


# Application entry point
if __name__ == "__main__":
    dashboard = DashboardEngine()
    dashboard.run()
