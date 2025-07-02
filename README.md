# NLP Sentiment Analysis: Comedy Transcript Analysis

## 🎯 Project Overview

This project demonstrates a comprehensive Natural Language Processing (NLP) workflow for sentiment analysis using comedy transcripts from stand-up comedians. The analysis explores text preprocessing, exploratory data analysis, sentiment analysis, and topic modeling techniques to understand the linguistic patterns and emotional content in comedy routines.

## 📊 Dataset

The project analyzes transcripts from 12 prominent stand-up comedians:
- **Louis C.K.** - "Oh My God" (2017)
- **Dave Chappelle** - "Age of Spin" (2017)
- **Ricky Gervais** - "Humanity" (2018)
- **Bo Burnham** - "what." (2013)
- **Bill Burr** - "I'm Sorry You Feel That Way" (2014)
- **Jim Jefferies** - "Bare" (2014)
- **John Mulaney** - "The Comeback Kid" (2015)
- **Hasan Minhaj** - "Homecoming King" (2017)
- **Ali Wong** - "Baby Cobra" (2016)
- **Anthony Jeselnik** - "Thoughts and Prayers" (2015)
- **Mike Birbiglia** - "My Girlfriend's Boyfriend" (2013)
- **Joe Rogan** - "Triggered" (2016)

## 🏗️ Technical Architecture

### Core Technologies
- **Python 3.8+** - Primary programming language
- **Jupyter Notebooks** - Interactive development environment
- **Pandas & NumPy** - Data manipulation and numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning utilities
- **NLTK & spaCy** - Natural language processing
- **TextBlob** - Sentiment analysis
- **Gensim** - Topic modeling
- **BeautifulSoup** - Web scraping

### Project Structure
```
ml-nlp_here_i_come_sentiment_analysis/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── contractions.py              # Contraction mapping utility
├── transcripts/                 # Raw transcript data
│   ├── ali.txt
│   ├── anthony.txt
│   ├── bill.txt
│   ├── bo.txt
│   ├── dave.txt
│   ├── hasan.txt
│   ├── jim.txt
│   ├── joe.txt
│   ├── john.txt
│   ├── louis.txt
│   ├── mike.txt
│   └── ricky.txt
├── *.pkl                        # Processed data files
└── Jupyter Notebooks:
    ├── NLP - Data Cleaning.ipynb
    ├── NLP - EDA.ipynb
    ├── NLP - Sentiment Analysis.ipynb
    └── NLP - Topic Modeling.ipynb
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ml-nlp_here_i_come_sentiment_analysis
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download required NLTK data**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

5. **Download spaCy model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

## 📋 Functional Workflow

### 1. Data Collection & Cleaning (`NLP - Data Cleaning.ipynb`)

**Web Scraping Process:**
- Scrapes transcript data from scrapsfromtheloft.com
- Extracts text content from HTML using BeautifulSoup
- Stores raw transcripts in pickle format

**Text Preprocessing Pipeline:**
- **Lowercase conversion** - Standardizes text case
- **Punctuation removal** - Eliminates special characters
- **Contraction expansion** - Converts contractions to full forms
- **Number removal** - Removes words containing digits
- **Whitespace normalization** - Cleans up spacing issues
- **Stop word removal** - Eliminates common words
- **Lemmatization** - Reduces words to base form

**Key Functions:**
- `url_to_transcript()` - Web scraping utility
- `combine_text()` - Text aggregation
- `clean_text_round1()` - Initial cleaning
- `clean_text_round2()` - Secondary cleaning

### 2. Exploratory Data Analysis (`NLP - EDA.ipynb`)

**Text Statistics Analysis:**
- Word count per comedian
- Average word length
- Vocabulary diversity metrics
- Most frequent words analysis

**Visualization Components:**
- Word frequency distributions
- Word clouds for each comedian
- Comparative text statistics
- Vocabulary overlap analysis

**Key Insights:**
- Text length variations across comedians
- Common vocabulary patterns
- Unique linguistic characteristics per performer

### 3. Sentiment Analysis (`NLP - Sentiment Analysis.ipynb`)

**Sentiment Metrics:**
- **Polarity** (-1 to +1): Measures positive/negative sentiment
- **Subjectivity** (0 to 1): Measures opinion vs. fact content

**Analysis Methods:**
- **Overall sentiment** - Complete transcript analysis
- **Temporal sentiment** - Sentiment progression throughout routines
- **Comparative analysis** - Cross-comedian sentiment comparison

**Visualization Features:**
- Scatter plots of polarity vs. subjectivity
- Time-series sentiment plots
- Sentiment distribution charts

**Key Functions:**
- `split_text()` - Text segmentation for temporal analysis
- TextBlob sentiment extraction
- Sentiment trend visualization

### 4. Topic Modeling (`NLP - Topic Modeling.ipynb`)

**Latent Dirichlet Allocation (LDA):**
- Identifies underlying topics in comedy routines
- Extracts key themes and subjects
- Provides topic-word distributions

**Document-Term Matrix (DTM):**
- Converts text to numerical representation
- Enables machine learning analysis
- Supports topic modeling algorithms

**Topic Analysis Features:**
- Topic identification and labeling
- Topic distribution across comedians
- Word-topic probability matrices
- Topic coherence evaluation

## 🔧 Technical Implementation Details

### Data Processing Pipeline

1. **Raw Data Collection**
   ```python
   def url_to_transcript(url):
       page = requests.get(url).text
       soup = BeautifulSoup(page, "lxml")
       text = [p.text for p in soup.find(class_="post-content").find_all('p')]
       return text
   ```

2. **Text Cleaning**
   ```python
   def clean_text_round1(text):
       text = text.lower()
       text = re.sub('\[.*?\]', '', text)
       text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
       text = re.sub('\w*\d\w*', '', text)
       return text
   ```

3. **Sentiment Analysis**
   ```python
   from textblob import TextBlob
   pol = lambda x: TextBlob(x).sentiment.polarity
   sub = lambda x: TextBlob(x).sentiment.subjectivity
   ```

### Key Algorithms

**Sentiment Analysis:**
- TextBlob's lexicon-based approach
- Linguistic research-based sentiment scoring
- Context-aware sentiment calculation

**Topic Modeling:**
- Latent Dirichlet Allocation (LDA)
- Gensim implementation
- Coherence score optimization

**Text Preprocessing:**
- NLTK tokenization
- spaCy lemmatization
- Custom stop word removal

## 📈 Results & Insights

### Sentiment Analysis Findings
- **Most Positive Comedians:** John Mulaney, Hasan Minhaj
- **Most Negative Comedians:** Bill Burr, Louis C.K.
- **Most Subjective:** Anthony Jeselnik, Ali Wong
- **Most Objective:** Ricky Gervais, Bo Burnham

### Topic Modeling Results
- **Common Themes:** Relationships, social commentary, personal experiences
- **Unique Topics:** Each comedian shows distinct thematic focus
- **Topic Evolution:** Sentiment trends throughout routines

### Linguistic Patterns
- **Vocabulary Diversity:** Varies significantly across comedians
- **Word Frequency:** Common words reveal comedic style
- **Text Structure:** Different approaches to joke construction

## 🛠️ Usage Examples

### Running Sentiment Analysis
```python
import pandas as pd
from textblob import TextBlob

# Load processed data
data = pd.read_pickle('corpus.pkl')

# Calculate sentiment
data['polarity'] = data['transcript'].apply(lambda x: TextBlob(x).sentiment.polarity)
data['subjectivity'] = data['transcript'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
```

### Creating Word Clouds
```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
```

### Topic Modeling
```python
from gensim import models
from sklearn.feature_extraction.text import CountVectorizer

# Create document-term matrix
vectorizer = CountVectorizer(stop_words='english', max_features=1000)
dtm = vectorizer.fit_transform(texts)

# Apply LDA
lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary)
```

## 🔍 Performance Considerations

### Optimization Strategies
- **Batch Processing:** Handle large text corpora efficiently
- **Memory Management:** Use generators for large datasets
- **Caching:** Store intermediate results in pickle files
- **Parallel Processing:** Utilize multiprocessing for heavy computations

### Scalability
- **Modular Design:** Separate concerns for easy scaling
- **Configurable Parameters:** Adjustable analysis parameters
- **Extensible Architecture:** Easy to add new analysis methods

## 🧪 Testing & Validation

### Data Quality Checks
- Text length validation
- Character encoding verification
- Missing data detection
- Duplicate content identification

### Model Validation
- Sentiment accuracy assessment
- Topic coherence evaluation
- Cross-validation techniques
- Performance metrics calculation

## 📚 Dependencies

### Core Dependencies
- **pandas>=1.3.0** - Data manipulation
- **numpy>=1.21.0** - Numerical computing
- **matplotlib>=3.4.0** - Visualization
- **seaborn>=0.11.0** - Statistical graphics

### NLP Libraries
- **spacy>=3.1.0** - Advanced NLP
- **nltk>=3.6.0** - Natural language toolkit
- **textblob>=0.15.0** - Sentiment analysis
- **wordcloud>=1.8.0** - Word cloud generation

### Machine Learning
- **scikit-learn>=1.0.0** - ML algorithms
- **gensim>=4.0.0** - Topic modeling
- **scipy>=1.7.0** - Scientific computing

### Web Scraping
- **requests>=2.25.0** - HTTP requests
- **beautifulsoup4>=4.9.0** - HTML parsing
- **lxml>=4.6.0** - XML/HTML processing

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions
- Include inline comments for complex logic

## 📄 License

This project is for educational and research purposes. Please respect the original content creators' rights when using transcript data.

## 🙏 Acknowledgments

- **Data Source:** Transcripts from scrapsfromtheloft.com
- **NLP Libraries:** NLTK, spaCy, TextBlob communities
- **Visualization:** Matplotlib and Seaborn developers
- **Machine Learning:** Scikit-learn and Gensim teams

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Review existing documentation
- Check the Jupyter notebooks for implementation details

---

**Note:** This project serves as a learning exercise for NLP techniques and sentiment analysis. The analysis provides insights into comedic content but should not be considered definitive academic research.
