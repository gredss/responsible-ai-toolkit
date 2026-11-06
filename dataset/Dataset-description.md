# Indonesian News Dataset

## Overview

This dataset contains 5,000 Indonesian news articles collected from five major Indonesian news sources: CNN Indonesia, Kompas, Detik, Tempo, and Tribun News. The articles are categorized into five domains, with 1,000 articles per domain, covering news published between 2022 and 2026. The dataset was created for Indonesian clickbait detection research with multi-domain evaluation settings.

## Dataset Structure

The dataset consists of five CSV files, one for each domain:

1. indonesian_tech_dataset.csv (1,000 rows)
2. indonesian_politics_dataset.csv (1,000 rows)
3. indonesian_health_dataset.csv (1,000 rows)
4. indonesian_sport_dataset.csv (1,000 rows)
5. indonesian_education_dataset.csv (1,000 rows)

Each CSV file contains the following columns:

- `id`: Unique identifier for each article
- `source`: News source name
- `date`: Publication date (YYYY-MM-DD format)
- `title`: Article headline
- `url`: Direct link to the original article
- `article_text`: Full article content
- `label`: Clickbait annotation label (`clickbait` or `non-clickbait`)

## Domain Coverage

- **Technology**: Gadgets, artificial intelligence, cybersecurity, startups, fintech, e-commerce, gaming, telecommunications, and digital innovation.
- **Politics**: Elections, government policies, political parties, parliamentary activities, political leaders, and democratic processes.
- **Health**: Diseases, medical treatments, public health, nutrition, mental health, vaccines, and medical research.
- **Sport**: Football, badminton, basketball, esports, competitions, athlete achievements, and sports events.
- **Education**: Schools, universities, curriculum, scholarships, educational policies, learning methods, and academic research.

## Data Collection and Exploration

Articles were collected through automated web scraping from official news sitemaps with domain-specific keyword filtering to ensure content relevance. The scraping implementation can be found in [`dataset/scrap.py`](https://github.com/gredss/domain-shift-trustworthy-ai/blob/main/dataset/scrap.py). 

An exploratory data analysis (EDA) of the collected dataset, including source distribution, domain statistics, and data characteristics, is available in [`dataset/dataset_eda.ipynb`](https://github.com/gredss/domain-shift-trustworthy-ai/blob/main/dataset/dataset_eda.ipynb).

## Annotation Process

The annotation guideline was developed based on previous clickbait literature to ensure consistent labeling among annotators. A headline was labeled as **clickbait** if it contained characteristics such as sensational language, information withholding, forward-referencing, emotional triggers, or curiosity-inducing interrogative structures. Headlines that explicitly conveyed the main information without intentionally creating excessive curiosity were labeled as **non-clickbait**.

The dataset was independently annotated by three annotators using a binary classification scheme (`clickbait` and `non-clickbait`). The final label was determined using majority voting when disagreements occurred. Inter-annotator agreement was evaluated using Fleiss’ Kappa, achieving a κ score of **0.72** with an observed agreement of **87%**, indicating substantial agreement among annotators.

## Data Quality

All articles contain substantial textual content (minimum 150 characters) and were deduplicated based on URLs and titles. The dataset represents authentic news articles from reputable Indonesian news organizations, providing a reliable resource for Indonesian clickbait detection research.
