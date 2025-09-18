"""
Charts: Visual Poetry for Financial Data
Cazandra Aporbo, MS
May 2025

These aren't just charts. They're visual narratives that make complex patterns
instantly understandable. I spent months perfecting the color psychology,
animation timing, and information hierarchy. Every pixel serves a purpose.

The secret to great financial visualization isn't showing more data.
It's showing the right data in a way that creates immediate understanding.
These charts do that.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import colorsys

# The palette that took weeks to perfect
# Each color chosen for psychological impact and accessibility
PALETTE = {
    # Primary pastels - soft but distinctive
    'rose': '#FFE4E1',      # Misty rose - negative sentiment
    'lavender': '#E6E6FA',  # Lavender - neutral/transition
    'sky': '#F0F8FF',       # Alice blue - positive sentiment
    'mint': '#F0FFF0',      # Honeydew - profit/success
    'blush': '#FFF0F5',     # Lavender blush - attention/alert
    
    # Secondary pastels - supporting actors
    'sand': '#FAFAD2',      # Light goldenrod - volume/quantity
    'pearl': '#FFF8DC',     # Cornsilk - background/subtle
    'frost': '#F0FFFF',     # Azure - cold/analytical
    'peach': '#FFDAB9',     # Peach puff - warm/human
    'sage': '#E0EED6',      # Sage green - growth/stability
    
    # Accent colors - used sparingly for emphasis
    'coral': '#FFB6C1',     # Light pink - strong negative
    'periwinkle': '#CCCCFF',# Periwinkle - strong positive
    'butter': '#FFFFE0',    # Light yellow - highlight/focus
    
    # Neutral grays - structure and text
    'smoke': '#F5F5F5',     # White smoke - backgrounds
    'silver': '#E8E8E8',    # Light gray - grid lines
    'steel': '#C0C0C0',     # Silver - borders
    'charcoal': '#4A4A4A',  # Dark gray - primary text
    'midnight': '#2C3E50'   # Dark blue-gray - headers
}

# Pre-calculated color gradients for smooth transitions
GRADIENTS = {
    'sentiment': ['#FFB6C1', '#FFE4E1', '#E6E6FA', '#F0F8FF', '#CCCCFF'],
    'heat': ['#FFB6C1', '#FFDAB9', '#FFFFE0', '#F0FFF0', '#E0EED6'],
    'cool': ['#F0FFFF', '#F0F8FF', '#E6E6FA', '#CCCCFF', '#E0EED6'],
    'diverging': ['#FFB6C1', '#FFE4E1', '#F5F5F5', '#F0F8FF', '#CCCCFF'],
    'sequential': ['#F5F5F5', '#E6E6FA', '#CCCCFF', '#E0EED6', '#F0FFF0']
}


class ChartConfig:
    """
    Configuration that makes charts memorable.
    
    I learned that consistency creates recognition. When users see
    these charts, they immediately know it's MarketPulse.
    """
    
    # Typography that's clean and readable
    FONT_FAMILY = "'SF Pro Display', 'Helvetica Neue', 'Arial', sans-serif"
    FONT_SIZE_TITLE = 24
    FONT_SIZE_SUBTITLE = 16
    FONT_SIZE_LABEL = 12
    FONT_SIZE_TICK = 10
    
    # Animation that feels natural
    ANIMATION_DURATION = 750  # Milliseconds
    ANIMATION_EASING = 'cubic-in-out'
    
    # Layout breathing room
    MARGIN = dict(l=80, r=80, t=100, b=80)
    PADDING = 0.05
    
    # Grid that guides but doesn't distract
    GRID_COLOR = PALETTE['silver']
    GRID_WIDTH = 0.5
    
    # Hover that informs
    HOVER_BGCOLOR = 'rgba(255, 255, 255, 0.95)'
    HOVER_BORDERCOLOR = PALETTE['steel']
    
    @staticmethod
    def get_layout_template() -> dict:
        """Base layout that all charts inherit."""
        return dict(
            font=dict(
                family=ChartConfig.FONT_FAMILY,
                size=ChartConfig.FONT_SIZE_LABEL,
                color=PALETTE['charcoal']
            ),
            plot_bgcolor=PALETTE['smoke'],
            paper_bgcolor='white',
            margin=ChartConfig.MARGIN,
            hovermode='closest',
            hoverlabel=dict(
                bgcolor=ChartConfig.HOVER_BGCOLOR,
                bordercolor=ChartConfig.HOVER_BORDERCOLOR,
                font=dict(
                    family=ChartConfig.FONT_FAMILY,
                    size=ChartConfig.FONT_SIZE_LABEL
                )
            ),
            xaxis=dict(
                gridcolor=ChartConfig.GRID_COLOR,
                gridwidth=ChartConfig.GRID_WIDTH,
                zerolinecolor=ChartConfig.GRID_COLOR,
                zerolinewidth=1
            ),
            yaxis=dict(
                gridcolor=ChartConfig.GRID_COLOR,
                gridwidth=ChartConfig.GRID_WIDTH,
                zerolinecolor=ChartConfig.GRID_COLOR,
                zerolinewidth=1
            )
        )


class CandlestickArtist:
    """
    Creates candlestick charts that traders actually want to look at.
    
    Traditional candlesticks are ugly. Mine are beautiful while remaining
    functional. The secret is subtle gradients and careful spacing.
    """
    
    @staticmethod
    def create(df: pd.DataFrame, 
               title: str = "Price Action",
               show_volume: bool = True,
               show_ma: bool = True) -> go.Figure:
        """
        Create a candlestick chart that's both beautiful and informative.
        """
        
        # Determine color for each candle based on direction
        colors_up = PALETTE['mint']
        colors_down = PALETTE['rose']
        
        # Create subplots if showing volume
        if show_volume:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3],
                subplot_titles=(title, "Volume")
            )
            row_price = 1
            row_volume = 2
        else:
            fig = go.Figure()
            row_price = 1
            row_volume = None
        
        # Add candlestick trace
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price',
                increasing=dict(
                    line=dict(color=colors_up, width=1),
                    fillcolor=colors_up
                ),
                decreasing=dict(
                    line=dict(color=colors_down, width=1),
                    fillcolor=colors_down
                ),
                whiskerwidth=0,
                hoverlabel=dict(namelength=-1)
            ),
            row=row_price, col=1
        )
        
        # Add moving averages if requested
        if show_ma and len(df) > 20:
            # 20-day MA
            ma20 = df['Close'].rolling(window=20).mean()
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=ma20,
                    name='MA20',
                    line=dict(
                        color=PALETTE['periwinkle'],
                        width=2,
                        dash='solid'
                    ),
                    opacity=0.7,
                    hoverlabel=dict(namelength=-1)
                ),
                row=row_price, col=1
            )
            
            # 50-day MA if we have enough data
            if len(df) > 50:
                ma50 = df['Close'].rolling(window=50).mean()
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=ma50,
                        name='MA50',
                        line=dict(
                            color=PALETTE['lavender'],
                            width=2,
                            dash='dash'
                        ),
                        opacity=0.7,
                        hoverlabel=dict(namelength=-1)
                    ),
                    row=row_price, col=1
                )
        
        # Add volume bars if requested
        if show_volume and row_volume:
            # Color volume bars based on price direction
            volume_colors = [
                colors_up if df['Close'].iloc[i] >= df['Open'].iloc[i] 
                else colors_down
                for i in range(len(df))
            ]
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker=dict(
                        color=volume_colors,
                        line=dict(width=0)
                    ),
                    opacity=0.5,
                    hoverlabel=dict(namelength=-1)
                ),
                row=row_volume, col=1
            )
        
        # Update layout with our template
        layout_update = ChartConfig.get_layout_template()
        layout_update.update(dict(
            title=dict(
                text=title,
                font=dict(size=ChartConfig.FONT_SIZE_TITLE)
            ),
            xaxis_rangeslider_visible=False,
            height=600 if show_volume else 500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor=PALETTE['steel'],
                borderwidth=1
            )
        ))
        
        fig.update_layout(layout_update)
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=row_volume if show_volume else row_price, col=1)
        fig.update_yaxes(title_text="Price", row=row_price, col=1)
        if show_volume:
            fig.update_yaxes(title_text="Volume", row=row_volume, col=1)
        
        return fig


class HeatmapComposer:
    """
    Heatmaps that reveal hidden correlations.
    
    Most heatmaps are information overload. Mine use color psychology
    to make patterns jump out. The eye naturally finds what matters.
    """
    
    @staticmethod
    def create_correlation_matrix(data: pd.DataFrame,
                                 title: str = "Feature Correlation Matrix",
                                 mask_insignificant: bool = True) -> go.Figure:
        """
        Create a correlation heatmap that's actually readable.
        """
        
        # Calculate correlation
        corr_matrix = data.corr()
        
        # Mask insignificant correlations if requested
        if mask_insignificant:
            # Only show correlations > 0.3 or < -0.3
            mask = (np.abs(corr_matrix) < 0.3)
            corr_matrix = corr_matrix.where(~mask, 0)
        
        # Create custom colorscale (diverging pastel)
        colorscale = [
            [0.0, PALETTE['coral']],      # Strong negative
            [0.25, PALETTE['rose']],      # Weak negative
            [0.5, PALETTE['smoke']],       # Neutral
            [0.75, PALETTE['sky']],        # Weak positive
            [1.0, PALETTE['periwinkle']]   # Strong positive
        ]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale=colorscale,
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont=dict(size=10),
            colorbar=dict(
                title="Correlation",
                tickmode='linear',
                tick0=-1,
                dtick=0.5,
                thickness=20,
                len=0.7,
                bgcolor='white',
                bordercolor=PALETTE['steel'],
                borderwidth=1
            ),
            hovertemplate='%{x}<br>%{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        # Update layout
        layout_update = ChartConfig.get_layout_template()
        layout_update.update(dict(
            title=dict(
                text=title,
                font=dict(size=ChartConfig.FONT_SIZE_TITLE)
            ),
            height=600,
            width=700,
            xaxis=dict(
                tickangle=-45,
                side='bottom'
            ),
            yaxis=dict(
                autorange='reversed'
            )
        ))
        
        fig.update_layout(layout_update)
        
        return fig
    
    @staticmethod
    def create_time_heatmap(data: pd.DataFrame,
                           value_col: str,
                           title: str = "Temporal Patterns") -> go.Figure:
        """
        Create a time-based heatmap showing patterns across days/hours.
        Perfect for finding when markets are most active.
        """
        
        # Extract time components
        data = data.copy()
        data['hour'] = data.index.hour
        data['day'] = data.index.dayofweek
        
        # Pivot to create matrix
        pivot = data.pivot_table(
            values=value_col,
            index='hour',
            columns='day',
            aggfunc='mean'
        )
        
        # Day names for labels
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        # Create heatmap with custom colorscale
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=day_names[:len(pivot.columns)],
            y=[f"{h:02d}:00" for h in pivot.index],
            colorscale=GRADIENTS['heat'],
            colorbar=dict(
                title=value_col,
                thickness=20,
                len=0.7
            ),
            hovertemplate='%{x} %{y}<br>Value: %{z:.3f}<extra></extra>'
        ))
        
        # Update layout
        layout_update = ChartConfig.get_layout_template()
        layout_update.update(dict(
            title=dict(
                text=title,
                font=dict(size=ChartConfig.FONT_SIZE_TITLE)
            ),
            height=500,
            xaxis=dict(title="Day of Week"),
            yaxis=dict(title="Hour of Day")
        ))
        
        fig.update_layout(layout_update)
        
        return fig


class FlowVisualizer:
    """
    Visualizes the flow of sentiment and momentum.
    
    Static charts miss the story. My flow visualizations show how
    sentiment propagates through markets, how momentum builds and breaks.
    It's like watching the market breathe.
    """
    
    @staticmethod
    def create_sentiment_flow(data: pd.DataFrame,
                            sentiment_col: str = 'sentiment',
                            title: str = "Sentiment Flow Dynamics") -> go.Figure:
        """
        Create a flowing river of sentiment over time.
        """
        
        # Calculate sentiment bands (like Bollinger bands for sentiment)
        sentiment = data[sentiment_col]
        sentiment_ma = sentiment.rolling(window=20, center=True).mean()
        sentiment_std = sentiment.rolling(window=20, center=True).std()
        
        upper_band = sentiment_ma + 2 * sentiment_std
        lower_band = sentiment_ma - 2 * sentiment_std
        
        # Create figure
        fig = go.Figure()
        
        # Add the bands (creating a river effect)
        fig.add_trace(go.Scatter(
            x=data.index,
            y=upper_band,
            name='Upper Band',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=lower_band,
            name='Lower Band',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(230, 230, 250, 0.3)',  # Lavender with transparency
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add the moving average (the river center)
        fig.add_trace(go.Scatter(
            x=data.index,
            y=sentiment_ma,
            name='Sentiment Trend',
            line=dict(
                color=PALETTE['periwinkle'],
                width=3
            ),
            hoverlabel=dict(namelength=-1)
        ))
        
        # Add actual sentiment as scattered points
        # Color based on positive/negative
        colors = [PALETTE['mint'] if s > 0 else PALETTE['rose'] for s in sentiment]
        sizes = [abs(s) * 10 + 5 for s in sentiment]  # Size based on magnitude
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=sentiment,
            mode='markers',
            name='Daily Sentiment',
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(
                    width=1,
                    color='white'
                ),
                opacity=0.6
            ),
            hoverlabel=dict(namelength=-1),
            hovertemplate='Date: %{x}<br>Sentiment: %{y:.3f}<extra></extra>'
        ))
        
        # Add zero line
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color=PALETTE['steel'],
            opacity=0.5
        )
        
        # Update layout
        layout_update = ChartConfig.get_layout_template()
        layout_update.update(dict(
            title=dict(
                text=title,
                font=dict(size=ChartConfig.FONT_SIZE_TITLE)
            ),
            height=400,
            xaxis=dict(title="Time"),
            yaxis=dict(title="Sentiment Score"),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        ))
        
        fig.update_layout(layout_update)
        
        return fig
    
    @staticmethod
    def create_momentum_cascade(data: pd.DataFrame,
                               price_col: str = 'Close',
                               volume_col: str = 'Volume') -> go.Figure:
        """
        Visualize momentum as a cascading waterfall.
        Shows how price movements build on each other.
        """
        
        # Calculate momentum components
        returns = data[price_col].pct_change()
        momentum_5 = returns.rolling(5).mean()
        momentum_20 = returns.rolling(20).mean()
        volume_norm = data[volume_col] / data[volume_col].rolling(20).mean()
        
        # Create subplot figure
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=("Price with Momentum Strength",
                           "Momentum Differential",
                           "Volume Surge Indicator")
        )
        
        # Price line with momentum-based coloring
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[price_col],
                mode='lines',
                name='Price',
                line=dict(
                    color=PALETTE['midnight'],
                    width=2
                ),
                hoverlabel=dict(namelength=-1)
            ),
            row=1, col=1
        )
        
        # Add momentum strength as background
        momentum_strength = momentum_5 - momentum_20
        
        # Positive momentum
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[price_col].where(momentum_strength > 0),
                fill='tozeroy',
                fillcolor='rgba(224, 238, 214, 0.3)',  # Sage with transparency
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        # Negative momentum
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[price_col].where(momentum_strength < 0),
                fill='tozeroy',
                fillcolor='rgba(255, 228, 225, 0.3)',  # Rose with transparency
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        # Momentum differential
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=momentum_strength,
                name='Momentum',
                marker=dict(
                    color=momentum_strength,
                    colorscale=GRADIENTS['diverging'],
                    cmin=-0.02,
                    cmax=0.02,
                    line=dict(width=0)
                ),
                hoverlabel=dict(namelength=-1)
            ),
            row=2, col=1
        )
        
        # Volume surge
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=volume_norm,
                fill='tozeroy',
                name='Volume Ratio',
                line=dict(
                    color=PALETTE['sand'],
                    width=1
                ),
                fillcolor='rgba(250, 250, 210, 0.3)',  # Sand with transparency
                hoverlabel=dict(namelength=-1)
            ),
            row=3, col=1
        )
        
        # Add reference line for normal volume
        fig.add_hline(
            y=1,
            line_dash="dot",
            line_color=PALETTE['steel'],
            opacity=0.5,
            row=3, col=1
        )
        
        # Update layout
        layout_update = ChartConfig.get_layout_template()
        layout_update.update(dict(
            height=700,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        ))
        
        fig.update_layout(layout_update)
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Momentum", row=2, col=1)
        fig.update_yaxes(title_text="Volume Ratio", row=3, col=1)
        
        return fig


class PerformanceRadar:
    """
    Multi-dimensional performance visualization.
    
    Bar charts show one dimension. My radar charts show the full
    performance profile at a glance. Perfect for model comparison.
    """
    
    @staticmethod
    def create_model_comparison(metrics_dict: Dict[str, Dict[str, float]],
                               title: str = "Model Performance Profile") -> go.Figure:
        """
        Create a radar chart comparing multiple models across metrics.
        """
        
        # Extract categories and models
        categories = list(next(iter(metrics_dict.values())).keys())
        models = list(metrics_dict.keys())
        
        # Create figure
        fig = go.Figure()
        
        # Colors for each model
        model_colors = [
            PALETTE['periwinkle'],
            PALETTE['mint'],
            PALETTE['peach'],
            PALETTE['lavender'],
            PALETTE['sand']
        ]
        
        # Add trace for each model
        for i, (model, metrics) in enumerate(metrics_dict.items()):
            values = list(metrics.values())
            
            # Close the radar chart
            values += values[:1]
            categories_closed = categories + categories[:1]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories_closed,
                fill='toself',
                fillcolor=model_colors[i % len(model_colors)],
                opacity=0.3,
                name=model,
                line=dict(
                    color=model_colors[i % len(model_colors)],
                    width=2
                ),
                hoverlabel=dict(namelength=-1)
            ))
        
        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickmode='linear',
                    tick0=0,
                    dtick=0.2,
                    gridcolor=PALETTE['silver'],
                    gridwidth=1
                ),
                angularaxis=dict(
                    gridcolor=PALETTE['silver'],
                    gridwidth=1
                ),
                bgcolor=PALETTE['smoke']
            ),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.1,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor=PALETTE['steel'],
                borderwidth=1
            ),
            title=dict(
                text=title,
                font=dict(
                    size=ChartConfig.FONT_SIZE_TITLE,
                    family=ChartConfig.FONT_FAMILY
                )
            ),
            font=dict(
                family=ChartConfig.FONT_FAMILY,
                size=ChartConfig.FONT_SIZE_LABEL,
                color=PALETTE['charcoal']
            ),
            paper_bgcolor='white',
            height=500,
            margin=dict(l=80, r=200, t=100, b=80)
        )
        
        return fig


class SankeyFlowBuilder:
    """
    Shows how predictions flow into profits and losses.
    
    Sankey diagrams are underused in finance. They perfectly show
    how initial predictions cascade into final outcomes.
    """
    
    @staticmethod
    def create_prediction_flow(predictions_df: pd.DataFrame,
                              title: str = "Prediction to Outcome Flow") -> go.Figure:
        """
        Create a Sankey diagram showing prediction flow to outcomes.
        """
        
        # Define nodes
        nodes = [
            "All Predictions",
            "Bullish Signal",
            "Bearish Signal",
            "True Positive",
            "False Positive",
            "True Negative",
            "False Negative",
            "Profit",
            "Loss"
        ]
        
        # Colors for nodes (pastels)
        node_colors = [
            PALETTE['lavender'],  # All
            PALETTE['mint'],      # Bullish
            PALETTE['rose'],      # Bearish
            PALETTE['sage'],      # TP
            PALETTE['peach'],     # FP
            PALETTE['sage'],      # TN
            PALETTE['peach'],     # FN
            PALETTE['mint'],      # Profit
            PALETTE['coral']      # Loss
        ]
        
        # Calculate flows (example structure)
        total = len(predictions_df)
        bullish = predictions_df['predicted'].sum()
        bearish = total - bullish
        
        # This is example logic - adjust based on actual data structure
        tp = ((predictions_df['predicted'] == 1) & (predictions_df['actual'] == 1)).sum()
        fp = ((predictions_df['predicted'] == 1) & (predictions_df['actual'] == 0)).sum()
        tn = ((predictions_df['predicted'] == 0) & (predictions_df['actual'] == 0)).sum()
        fn = ((predictions_df['predicted'] == 0) & (predictions_df['actual'] == 1)).sum()
        
        # Define links
        links = [
            # From All to Signals
            dict(source=0, target=1, value=bullish),
            dict(source=0, target=2, value=bearish),
            
            # From Bullish to Outcomes
            dict(source=1, target=3, value=tp),  # True Positive
            dict(source=1, target=4, value=fp),  # False Positive
            
            # From Bearish to Outcomes
            dict(source=2, target=5, value=tn),  # True Negative
            dict(source=2, target=6, value=fn),  # False Negative
            
            # From Outcomes to Profit/Loss
            dict(source=3, target=7, value=tp),  # TP to Profit
            dict(source=4, target=8, value=fp),  # FP to Loss
            dict(source=5, target=7, value=tn),  # TN to Profit (avoided loss)
            dict(source=6, target=8, value=fn),  # FN to Loss (missed opportunity)
        ]
        
        # Create Sankey
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color=PALETTE['steel'], width=0.5),
                label=nodes,
                color=node_colors,
                hovertemplate='%{label}<br>Count: %{value}<extra></extra>'
            ),
            link=dict(
                source=[link['source'] for link in links],
                target=[link['target'] for link in links],
                value=[link['value'] for link in links],
                color='rgba(230, 230, 250, 0.4)'  # Lavender with transparency
            )
        )])
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(
                    size=ChartConfig.FONT_SIZE_TITLE,
                    family=ChartConfig.FONT_FAMILY,
                    color=PALETTE['midnight']
                )
            ),
            font=dict(
                family=ChartConfig.FONT_FAMILY,
                size=ChartConfig.FONT_SIZE_LABEL,
                color=PALETTE['charcoal']
            ),
            paper_bgcolor='white',
            plot_bgcolor=PALETTE['smoke'],
            height=500,
            margin=dict(l=50, r=50, t=100, b=50)
        )
        
        return fig


class MetricCards:
    """
    KPI cards that actually communicate.
    
    Numbers without context are meaningless. My metric cards show
    the number, the trend, the context, and the significance.
    All in a glance.
    """
    
    @staticmethod
    def create_metric_grid(metrics: Dict[str, Dict], 
                          cols: int = 4) -> str:
        """
        Create HTML for a grid of metric cards.
        Returns HTML string for Streamlit markdown.
        """
        
        html_parts = ['<div style="display: grid; grid-template-columns: repeat({}, 1fr); gap: 20px; padding: 20px;">'.format(cols)]
        
        for name, data in metrics.items():
            value = data.get('value', 0)
            change = data.get('change', 0)
            label = data.get('label', name)
            format_str = data.get('format', '.2f')
            
            # Determine color based on change
            if change > 0:
                change_color = PALETTE['mint']
                arrow = '↑'
            elif change < 0:
                change_color = PALETTE['rose']
                arrow = '↓'
            else:
                change_color = PALETTE['lavender']
                arrow = '→'
            
            # Format the card HTML
            card_html = f'''
            <div style="
                background: linear-gradient(135deg, {PALETTE['smoke']} 0%, white 100%);
                border: 1px solid {PALETTE['steel']};
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                transition: transform 0.2s;
                hover: transform: translateY(-2px);
            ">
                <div style="
                    font-size: 14px;
                    color: {PALETTE['charcoal']};
                    margin-bottom: 8px;
                    font-weight: 500;
                ">{label}</div>
                
                <div style="
                    font-size: 32px;
                    color: {PALETTE['midnight']};
                    font-weight: bold;
                    margin-bottom: 8px;
                ">{value:{format_str}}</div>
                
                <div style="
                    font-size: 14px;
                    color: {change_color};
                    display: flex;
                    align-items: center;
                    gap: 4px;
                ">
                    <span style="font-size: 18px;">{arrow}</span>
                    <span>{abs(change):{format_str}}%</span>
                </div>
            </div>
            '''
            
            html_parts.append(card_html)
        
        html_parts.append('</div>')
        
        return ''.join(html_parts)


# Utility functions for color manipulation
def lighten_color(hex_color: str, amount: float = 0.2) -> str:
    """
    Lighten a hex color by a given amount.
    I use this for hover states and gradients.
    """
    # Remove # if present
    hex_color = hex_color.lstrip('#')
    
    # Convert to RGB
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Convert to HSL
    r, g, b = r/255.0, g/255.0, b/255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    
    # Lighten
    l = min(1, l + amount)
    
    # Convert back to RGB
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    
    # Convert to hex
    return f'#{r:02x}{g:02x}{b:02x}'


def create_gradient(start_color: str, end_color: str, steps: int = 10) -> List[str]:
    """
    Create a gradient between two colors.
    Essential for smooth color transitions in visualizations.
    """
    start = start_color.lstrip('#')
    end = end_color.lstrip('#')
    
    # Convert to RGB
    start_rgb = tuple(int(start[i:i+2], 16) for i in (0, 2, 4))
    end_rgb = tuple(int(end[i:i+2], 16) for i in (0, 2, 4))
    
    # Create gradient
    gradient = []
    for i in range(steps):
        ratio = i / (steps - 1)
        r = int(start_rgb[0] + ratio * (end_rgb[0] - start_rgb[0]))
        g = int(start_rgb[1] + ratio * (end_rgb[1] - start_rgb[1]))
        b = int(start_rgb[2] + ratio * (end_rgb[2] - start_rgb[2]))
        gradient.append(f'#{r:02x}{g:02x}{b:02x}')
    
    return gradient


# Example usage
if __name__ == "__main__":
    import numpy as np
    
    # Generate sample data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    sample_data = pd.DataFrame({
        'Open': 100 + np.cumsum(np.random.randn(len(dates)) * 2),
        'High': 102 + np.cumsum(np.random.randn(len(dates)) * 2),
        'Low': 98 + np.cumsum(np.random.randn(len(dates)) * 2),
        'Close': 100 + np.cumsum(np.random.randn(len(dates)) * 2),
        'Volume': np.random.randint(1000000, 10000000, len(dates)),
        'sentiment': np.random.randn(len(dates)) * 0.3
    }, index=dates)
    
    # Fix OHLC relationships
    sample_data['High'] = sample_data[['Open', 'High', 'Low', 'Close']].max(axis=1)
    sample_data['Low'] = sample_data[['Open', 'High', 'Low', 'Close']].min(axis=1)
    
    # Create candlestick chart
    candlestick = CandlestickArtist.create(sample_data.tail(60), title="DEMO Price Action")
    
    # Create sentiment flow
    sentiment_flow = FlowVisualizer.create_sentiment_flow(sample_data.tail(90))
    
    print("Charts created successfully. Ready for display in dashboard.")
