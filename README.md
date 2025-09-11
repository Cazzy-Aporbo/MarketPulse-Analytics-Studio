# MarketPulse Analytics Studio

<div align="center">

[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Ensemble%20Models-FFE4E1?style=for-the-badge)](https://github.com/Cazandra-Aporbo/MarketPulse-Analytics-Studio)
[![Financial Analysis](https://img.shields.io/badge/Financial%20Analysis-Real--time%20Sentiment-E6E6FA?style=for-the-badge)](https://github.com/Cazandra-Aporbo/MarketPulse-Analytics-Studio)
[![Interactive Dashboard](https://img.shields.io/badge/Interactive%20Dashboard-Advanced%20Visualizations-F0F8FF?style=for-the-badge)](https://github.com/Cazandra-Aporbo/MarketPulse-Analytics-Studio)

</div>

<div align="center">
  
![separator](https://img.shields.io/badge/-FFE4E1?style=flat-square&color=FFE4E1)
![separator](https://img.shields.io/badge/-E6E6FA?style=flat-square&color=E6E6FA)
![separator](https://img.shields.io/badge/-F0F8FF?style=flat-square&color=F0F8FF)
![separator](https://img.shields.io/badge/-FFF0F5?style=flat-square&color=FFF0F5)
![separator](https://img.shields.io/badge/-FAFAD2?style=flat-square&color=FAFAD2)

</div>

<div align="center">
<table width="80%">
<tr>
<td align="center" style="background: linear-gradient(135deg, #FFE4E1 0%, #FFF0F5 100%); padding:30px; border-radius:15px;">
<h3>My Philosophy</h3>
<i>"Markets aren't just numbers. They're human psychology at scale. Every price movement tells a story about fear, greed, and everything in between. I built this to explore those stories."</i>
<br><br>
<b>Cazandra Aporbo, MS</b><br>
Started May 2025
</td>
</tr>
</table>
</div>

<br>

Started this in May 2025 to explore how sentiment analysis could enhance traditional technical analysis in trading. This project demonstrates a framework for combining multiple data sources, ensemble machine learning models, and real-time visualization to analyze market sentiment and its potential impact on price movements.

This is a portfolio project showcasing my approach to financial data science and machine learning architecture. While the demo uses synthetic data for accessibility, the architecture and concepts are production-ready.

<div align="center">
  
![separator](https://img.shields.io/badge/-E6E6FA?style=flat&color=E6E6FA)
![separator](https://img.shields.io/badge/-F0F8FF?style=flat&color=F0F8FF)
![separator](https://img.shields.io/badge/-FFF0F5?style=flat&color=FFF0F5)

</div>

## What This Project Demonstrates

<div align="center">
<table width="100%" cellspacing="10">
<tr>
<td width="25%" align="center" style="background: linear-gradient(180deg, #FFE4E1 0%, #FFB6C1 100%); padding:20px; border-radius:10px;">
<h4>Data Pipeline</h4>
<b>Multi-source ingestion</b><br>
Handles API failures gracefully<br>
Smart caching system<br>
<small>Ready for real feeds</small>
</td>
<td width="25%" align="center" style="background: linear-gradient(180deg, #E6E6FA 0%, #DDA0DD 100%); padding:20px; border-radius:10px;">
<h4>Sentiment Analysis</h4>
<b>Context-aware scoring</b><br>
Financial lexicon approach<br>
Source weighting system<br>
<small>Extensible framework</small>
</td>
<td width="25%" align="center" style="background: linear-gradient(180deg, #F0F8FF 0%, #B0E0E6 100%); padding:20px; border-radius:10px;">
<h4>ML Architecture</h4>
<b>Ensemble approach</b><br>
Time series validation<br>
Feature engineering pipeline<br>
<small>Modular design</small>
</td>
<td width="25%" align="center" style="background: linear-gradient(180deg, #FFF0F5 0%, #FFE4E1 100%); padding:20px; border-radius:10px;">
<h4>Visualization</h4>
<b>Real-time dashboard</b><br>
Interactive Plotly charts<br>
Multiple timeframes<br>
<small>Streamlit powered</small>
</td>
</tr>
</table>
</div>

<br>

The system demonstrates how to properly structure a financial ML project: clean separation of concerns, proper time series validation, and realistic handling of data issues. The sentiment analysis shows weighted source credibility because not all news sources are equal. Bloomberg might get weighted differently than social media posts.

The dashboard provides actionable visualizations. Each chart serves a purpose: identifying sentiment trends, detecting divergences, and spotting potential opportunities. The architecture is designed to be extensible for additional data sources and models.

<div align="center">
  
![separator](https://img.shields.io/badge/-FFE4E1?style=flat&color=FFE4E1)
![separator](https://img.shields.io/badge/-E6E6FA?style=flat&color=E6E6FA)
![separator](https://img.shields.io/badge/-F0F8FF?style=flat&color=F0F8FF)

</div>

## Technical Architecture

<div align="center">

![Python](https://img.shields.io/badge/Python-FFE4E1?style=flat-square&logo=python&logoColor=666)
![scikit--learn](https://img.shields.io/badge/scikit--learn-E6E6FA?style=flat-square&logo=scikit-learn&logoColor=666)
![Streamlit](https://img.shields.io/badge/Streamlit-F0F8FF?style=flat-square&logo=streamlit&logoColor=666)
![Plotly](https://img.shields.io/badge/Plotly-FFF0F5?style=flat-square&logo=plotly&logoColor=666)
![Pandas](https://img.shields.io/badge/Pandas-FAFAD2?style=flat-square&logo=pandas&logoColor=666)
![NumPy](https://img.shields.io/badge/NumPy-FFE4B5?style=flat-square&logo=numpy&logoColor=666)
![yfinance](https://img.shields.io/badge/yfinance-E0FFFF?style=flat-square&logo=yahoo&logoColor=666)
![Docker](https://img.shields.io/badge/Docker-FFE4E1?style=flat-square&logo=docker&logoColor=666)

</div>

<div align="center">
<table width="100%" style="border-spacing: 5px;">
<tr>
<td colspan="3" align="center" style="background: linear-gradient(90deg, #FAFAD2 0%, #FFE4B5 100%); padding:15px; border-radius:10px;">
<b>KEY CONCEPTS EXPLORED</b>
</td>
</tr>
<tr>
<td width="30%" style="background-color:#FFE4E1; padding:15px; border-radius:8px;"><b>Concept</b></td>
<td width="30%" style="background-color:#FFE4E1; padding:15px; border-radius:8px;"><b>Implementation</b></td>
<td width="40%" style="background-color:#FFE4E1; padding:15px; border-radius:8px;"><b>Why It Matters</b></td>
</tr>
<tr>
<td style="background-color:#FFF5F5; padding:12px;">Sentiment Momentum</td>
<td style="background-color:#FFF5F5; padding:12px;">Rolling window analysis</td>
<td style="background-color:#FFF5F5; padding:12px;">Rate of sentiment change often predicts better than absolute sentiment</td>
</tr>
<tr>
<td style="background-color:#F0F8FF; padding:12px;">Source Weighting</td>
<td style="background-color:#F0F8FF; padding:12px;">Configurable credibility scores</td>
<td style="background-color:#F0F8FF; padding:12px;">Professional sources typically more reliable than social media</td>
</tr>
<tr>
<td style="background-color:#FFF5F5; padding:12px;">Feature Engineering</td>
<td style="background-color:#FFF5F5; padding:12px;">Technical + sentiment indicators</td>
<td style="background-color:#FFF5F5; padding:12px;">Combining multiple signal types improves prediction potential</td>
</tr>
<tr>
<td style="background-color:#FFF0F5; padding:12px;">Time Series Validation</td>
<td style="background-color:#FFF0F5; padding:12px;">Walk-forward analysis</td>
<td style="background-color:#FFF0F5; padding:12px;">Prevents look-ahead bias in financial models</td>
</tr>
<tr>
<td style="background-color:#FFF5F5; padding:12px;">Risk Management</td>
<td style="background-color:#FFF5F5; padding:12px;">Volatility-based position sizing</td>
<td style="background-color:#FFF5F5; padding:12px;">Essential for any trading strategy</td>
</tr>
<tr>
<td style="background-color:#E6E6FA; padding:12px;">Ensemble Methods</td>
<td style="background-color:#E6E6FA; padding:12px;">Multiple model voting</td>
<td style="background-color:#E6E6FA; padding:12px;">Different models capture different patterns</td>
</tr>
</table>
</div>

<div align="center">
  
![separator](https://img.shields.io/badge/-FFF0F5?style=flat&color=FFF0F5)
![separator](https://img.shields.io/badge/-FAFAD2?style=flat&color=FAFAD2)
![separator](https://img.shields.io/badge/-FFE4B5?style=flat&color=FFE4B5)

</div>

## Machine Learning Approach

<details>
<summary><b>Click to see the ensemble architecture</b></summary>

<div align="center">
<table width="100%">
<tr>
<td colspan="3" align="center" style="background: linear-gradient(90deg, #E6E6FA 0%, #DDA0DD 50%, #E6E6FA 100%); padding:15px; border-radius:10px;">
<b>ENSEMBLE LEARNING FRAMEWORK</b>
</td>
</tr>
<tr style="background-color:#FFE4E1;">
<td style="padding:10px; border-radius:5px;"><b>Model Type</b></td>
<td style="padding:10px; border-radius:5px;"><b>Purpose</b></td>
<td style="padding:10px; border-radius:5px;"><b>Implementation Notes</b></td>
</tr>
<tr style="background-color:#FFF5F5;">
<td style="padding:10px;">Logistic Regression</td>
<td style="padding:10px;">Baseline predictor</td>
<td style="padding:10px;">Simple, interpretable, fast training</td>
</tr>
<tr style="background-color:#F0F8FF;">
<td style="padding:10px;">Random Forest</td>
<td style="padding:10px;">Non-linear patterns</td>
<td style="padding:10px;">Handles feature interactions well</td>
</tr>
<tr style="background-color:#FFF5F5;">
<td style="padding:10px;">XGBoost (planned)</td>
<td style="padding:10px;">Gradient boosting</td>
<td style="padding:10px;">Often best for tabular financial data</td>
</tr>
<tr style="background-color:#FFF0F5;">
<td style="padding:10px;">LSTM (planned)</td>
<td style="padding:10px;">Sequential patterns</td>
<td style="padding:10px;">Captures temporal dependencies in sentiment</td>
</tr>
</table>
</div>

The demo implements basic models to show the architecture. Production systems would add more sophisticated models, hyperparameter tuning, and cross-validation strategies.

</details>

<div align="center">
  
![separator](https://img.shields.io/badge/-E6E6FA?style=flat&color=E6E6FA)
![separator](https://img.shields.io/badge/-F0F8FF?style=flat&color=F0F8FF)
![separator](https://img.shields.io/badge/-FFF0F5?style=flat&color=FFF0F5)

</div>

## Skills Demonstrated

<div align="center">
<table width="100%" cellspacing="8">
<tr>
<td width="20%" align="center" style="background: linear-gradient(135deg, #FFE4E1 0%, #FFB6C1 100%); padding:20px; border-radius:12px;">
<b>Python Development</b><br><br>
Clean OOP design<br>
Error handling<br>
Type hints<br>
Modular architecture
</td>
<td width="20%" align="center" style="background: linear-gradient(135deg, #E6E6FA 0%, #DDA0DD 100%); padding:20px; border-radius:12px;">
<b>Machine Learning</b><br><br>
Ensemble methods<br>
Feature engineering<br>
Proper validation<br>
Model evaluation
</td>
<td width="20%" align="center" style="background: linear-gradient(135deg, #F0F8FF 0%, #B0E0E6 100%); padding:20px; border-radius:12px;">
<b>Data Engineering</b><br><br>
Pipeline design<br>
API integration<br>
Data validation<br>
Caching strategies
</td>
<td width="20%" align="center" style="background: linear-gradient(135deg, #FFF0F5 0%, #FFE4E1 100%); padding:20px; border-radius:12px;">
<b>Financial Analysis</b><br><br>
Technical indicators<br>
Risk metrics<br>
Market structure<br>
Time series analysis
</td>
<td width="20%" align="center" style="background: linear-gradient(135deg, #FAFAD2 0%, #FFE4B5 100%); padding:20px; border-radius:12px;">
<b>Visualization</b><br><br>
Interactive dashboards<br>
Real-time updates<br>
Multi-panel layouts<br>
User experience
</td>
</tr>
</table>
</div>

<div align="center">
  
![separator](https://img.shields.io/badge/-FFE4E1?style=flat&color=FFE4E1)
![separator](https://img.shields.io/badge/-E6E6FA?style=flat&color=E6E6FA)
![separator](https://img.shields.io/badge/-F0F8FF?style=flat&color=F0F8FF)

</div>

## Getting Started

```bash
# Clone the repository
git clone https://github.com/Cazandra-Aporbo/MarketPulse-Analytics-Studio.git
cd MarketPulse-Analytics-Studio

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API key (optional - demo works without it)
cp config/config.example.yaml config/config.yaml
# Edit config.yaml to add your Finnhub API key for real data

# Run the application
python run.py

# Open your browser to http://localhost:8501
```

The demo runs with synthetic data by default, so you can explore the functionality without needing API keys. Add a Finnhub API key to work with real market data.

<div align="center">
  
![separator](https://img.shields.io/badge/-FFE4E1?style=flat&color=FFE4E1)
![separator](https://img.shields.io/badge/-E6E6FA?style=flat&color=E6E6FA)
![separator](https://img.shields.io/badge/-F0F8FF?style=flat&color=F0F8FF)

</div>

## Key Features

<div align="center">
<table width="100%" style="border-collapse: separate; border-spacing: 3px;">
<tr>
<td colspan="2" align="center" style="background: linear-gradient(90deg, #FFE4B5 0%, #FFDEAD 100%); padding:15px; border-radius:10px;">
<b>IMPLEMENTED FEATURES</b>
</td>
</tr>
<tr>
<td width="40%" style="background-color:#FFE4E1; padding:12px; border-radius:5px;"><b>Feature</b></td>
<td width="60%" style="background-color:#FFE4E1; padding:12px; border-radius:5px;"><b>Description</b></td>
</tr>
<tr>
<td style="background-color:#FFF5F5; padding:10px;">Real-time Data Pipeline</td>
<td style="background-color:#FFF5F5; padding:10px;">Fetches stock data with fallback to synthetic data on API failure</td>
</tr>
<tr>
<td style="background-color:#F0F8FF; padding:10px;">Sentiment Analysis</td>
<td style="background-color:#F0F8FF; padding:10px;">Financial lexicon-based approach with configurable source weights</td>
</tr>
<tr>
<td style="background-color:#FFF5F5; padding:10px;">Technical Indicators</td>
<td style="background-color:#FFF5F5; padding:10px;">RSI, moving averages, volatility measures, mean reversion signals</td>
</tr>
<tr>
<td style="background-color:#FFF0F5; padding:10px;">Feature Engineering</td>
<td style="background-color:#FFF0F5; padding:10px;">Combines price, volume, and sentiment features</td>
</tr>
<tr>
<td style="background-color:#FFF5F5; padding:10px;">Ensemble Models</td>
<td style="background-color:#FFF5F5; padding:10px;">Multiple models with weighted voting system</td>
</tr>
<tr>
<td style="background-color:#FAFAD2; padding:10px;">Interactive Dashboard</td>
<td style="background-color:#FAFAD2; padding:10px;">Multi-panel visualization with Streamlit and Plotly</td>
</tr>
<tr>
<td style="background-color:#FFF5F5; padding:10px;">Risk Metrics</td>
<td style="background-color:#FFF5F5; padding:10px;">Volatility analysis, drawdown tracking, position sizing</td>
</tr>
</table>
</div>

<div align="center">
  
![separator](https://img.shields.io/badge/-FFF0F5?style=flat&color=FFF0F5)
![separator](https://img.shields.io/badge/-FAFAD2?style=flat&color=FAFAD2)
![separator](https://img.shields.io/badge/-FFE4B5?style=flat&color=FFE4B5)

</div>

## Project Structure

```
MarketPulse-Analytics-Studio/
│
├── run.py                    ← Main application entry point
├── requirements.txt          ← Python dependencies
├── config.yaml              ← Configuration settings
│
├── core/                    ← Core modules (to be expanded)
│   ├── data_pipeline.py    ← Data ingestion and processing
│   ├── sentiment_engine.py ← Sentiment analysis logic
│   ├── feature_factory.py  ← Feature engineering
│   └── model_ensemble.py   ← Model orchestration
│
├── models/                  ← Model implementations
│   └── baseline.py         ← Simple models for testing
│
├── visualization/           ← Dashboard components
│   ├── dashboard.py        ← Streamlit application
│   └── charts.py          ← Custom visualizations
│
├── notebooks/              ← Research and exploration
│   └── exploration.ipynb  ← Data analysis notebooks
│
└── tests/                  ← Unit tests
    └── test_pipeline.py    ← Testing data pipeline
```

<div align="center">
  
![separator](https://img.shields.io/badge/-E6E6FA?style=flat&color=E6E6FA)
![separator](https://img.shields.io/badge/-F0F8FF?style=flat&color=F0F8FF)
![separator](https://img.shields.io/badge/-FFF0F5?style=flat&color=FFF0F5)

</div>

## Usage Examples

<div align="center">
<table width="80%">
<tr>
<td style="background: linear-gradient(135deg, #FFE4E1 0%, #FFF0F5 100%); padding:25px; border-radius:12px;">

```python
# Example of how the framework would be used

from marketpulse import MarketPulse

# Initialize the system
mp = MarketPulse(config_path='config/config.yaml')

# Analyze a stock
analysis = mp.analyze(
    ticker='AAPL',
    period='3mo',
    include_sentiment=True
)

# Get feature matrix
features = analysis.get_features()
print(f"Generated {len(features.columns)} features")

# Run ensemble prediction
prediction = mp.predict(features)
print(f"Prediction: {prediction.direction}")
print(f"Confidence: {prediction.confidence:.2%}")

# Generate visualization
mp.visualize(analysis, save_path='analysis.html')
```

</td>
</tr>
</table>
</div>

<div align="center">
  
![separator](https://img.shields.io/badge/-FFE4E1?style=flat&color=FFE4E1)
![separator](https://img.shields.io/badge/-E6E6FA?style=flat&color=E6E6FA)
![separator](https://img.shields.io/badge/-F0F8FF?style=flat&color=F0F8FF)

</div>

## Learning Journey

<div align="center">
<table width="90%">
<tr>
<td width="50%" style="background-color:#E6E6FA; padding:12px; border-radius:8px;"><b>Challenge</b></td>
<td width="50%" style="background-color:#E6E6FA; padding:12px; border-radius:8px;"><b>Solution</b></td>
</tr>
<tr>
<td style="background-color:#FFF5F5; padding:10px;">Handling API failures gracefully</td>
<td style="background-color:#FFF5F5; padding:10px;">Implemented fallback to synthetic data for demo purposes</td>
</tr>
<tr>
<td style="background-color:#FFE4E1; padding:10px;">Time series data leakage</td>
<td style="background-color:#FFE4E1; padding:10px;">Used proper walk-forward validation instead of random splits</td>
</tr>
<tr>
<td style="background-color:#FFF5F5; padding:10px;">Feature explosion</td>
<td style="background-color:#FFF5F5; padding:10px;">Focused on interpretable features with clear financial meaning</td>
</tr>
<tr>
<td style="background-color:#F0F8FF; padding:10px;">Real-time visualization</td>
<td style="background-color:#F0F8FF; padding:10px;">Streamlit + Plotly provides smooth interactive experience</td>
</tr>
<tr>
<td style="background-color:#FFF5F5; padding:10px;">Model interpretability</td>
<td style="background-color:#FFF5F5; padding:10px;">Chose ensemble approach where each model's contribution is clear</td>
</tr>
</table>
</div>

<div align="center">
  
![separator](https://img.shields.io/badge/-E6E6FA?style=flat&color=E6E6FA)
![separator](https://img.shields.io/badge/-F0F8FF?style=flat&color=F0F8FF)
![separator](https://img.shields.io/badge/-FFF0F5?style=flat&color=FFF0F5)

</div>

## Future Enhancements

<div align="center">
<table width="100%" cellspacing="10">
<tr>
<td width="25%" align="center" style="background: linear-gradient(180deg, #FFE4E1 0%, #FFB6C1 100%); padding:18px; border-radius:10px;">
<b>Advanced NLP</b><br><br>
Integrate transformer models<br>
for better context understanding
</td>
<td width="25%" align="center" style="background: linear-gradient(180deg, #E6E6FA 0%, #DDA0DD 100%); padding:18px; border-radius:10px;">
<b>More Data Sources</b><br><br>
Social media APIs<br>
Options flow data
</td>
<td width="25%" align="center" style="background: linear-gradient(180deg, #F0F8FF 0%, #B0E0E6 100%); padding:18px; border-radius:10px;">
<b>Backtesting Engine</b><br><br>
Historical strategy testing<br>
with realistic constraints
</td>
<td width="25%" align="center" style="background: linear-gradient(180deg, #FFF0F5 0%, #FFE4E1 100%); padding:18px; border-radius:10px;">
<b>Risk Management</b><br><br>
Portfolio optimization<br>
Advanced position sizing
</td>
</tr>
</table>
</div>

<div align="center">
  
![separator](https://img.shields.io/badge/-FFE4E1?style=flat&color=FFE4E1)
![separator](https://img.shields.io/badge/-E6E6FA?style=flat&color=E6E6FA)
![separator](https://img.shields.io/badge/-F0F8FF?style=flat&color=F0F8FF)

</div>

## Contributing

This is a portfolio project, but I welcome feedback and suggestions! Feel free to:

- Report bugs or issues
- Suggest new features or improvements
- Share your own experiments with the framework
- Provide feedback on the architecture

<div align="center">
  
![separator](https://img.shields.io/badge/-FFF0F5?style=flat&color=FFF0F5)
![separator](https://img.shields.io/badge/-FAFAD2?style=flat&color=FAFAD2)
![separator](https://img.shields.io/badge/-FFE4B5?style=flat&color=FFE4B5)

</div>

## Disclaimer

**This is an educational project for portfolio purposes.** 

This is NOT financial advice. This system is not intended for actual trading. Markets are complex and risky. Always do your own research and consult with qualified financial advisors before making investment decisions.

The project demonstrates technical skills in:
- Software architecture and design
- Machine learning implementation
- Data pipeline development
- Financial data analysis
- Interactive visualization

<div align="center">
  
![separator](https://img.shields.io/badge/-E6E6FA?style=flat&color=E6E6FA)
![separator](https://img.shields.io/badge/-F0F8FF?style=flat&color=F0F8FF)
![separator](https://img.shields.io/badge/-FFF0F5?style=flat&color=FFF0F5)

</div>

## Contact

LinkedIn or email me if you want to discuss the technical implementation, architecture decisions, or potential applications of sentiment analysis in finance.

---

<div align="center">

<table width="60%">
<tr>
<td align="center" style="background: linear-gradient(135deg, #FAFAD2 0%, #FFE4B5 100%); padding:25px; border-radius:15px;">
<h3>Building the intersection of finance and machine learning</h3>
<b>Cazandra Aporbo, MS</b><br>
Data Scientist | Machine Learning Engineer | Financial Technology Enthusiast<br><br>
<i>"Understanding markets through data, one model at a time"</i>
</td>
</tr>
</table>

</div>
