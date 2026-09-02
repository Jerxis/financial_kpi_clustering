# Leveraging Clustering Algorithms to Improve Peer Benchmarking

**NOTE: This README file is for the repository as a whole, not a guide for evaluators of the final dissertation project. For the README for evaluators, check the README.md attached on the submission package in the KLE platform**

## Overview

Traditional business benchmarking relies heavily on manually selecting peer companies based on industry classifications and expert judgement. While this approach is widely used, it can introduce subjectivity and may fail to identify organisations with similar operational and financial characteristics.

This project explores whether unsupervised machine learning techniques can be used to support peer group selection by automatically clustering publicly traded companies based on financial performance indicators and structural characteristics.

The resulting system allows analysts to:

- Automatically group companies into comparable peer clusters
- Explore financial similarities within sectors
- Visualise company positioning using dimensionality reduction techniques
- Generate benchmark statistics for each cluster
- Support more objective and data-driven benchmarking exercises

The project was developed as part of an MSc in Artificial Intelligence and Data Science and is inspired by real-world benchmarking methodologies used in management consulting and performance improvement engagements.

## Business Problem

Benchmarking is only as useful as the quality of the peer group being used.

Companies are often compared against peers selected using:

- Industry classifications
- Revenue ranges
- Geographic regions
- Analyst judgement

However, organisations within the same industry can have dramatically different:

- Cost structures
- Profitability profiles
- Capital intensity
- Growth strategies
- Operational models

This project investigates whether machine learning can identify more homogeneous peer groups using underlying financial characteristics rather than relying solely on predefined classifications.

Methodology

The workflow consists of five major stages:

### 1. Data Collection

Financial statement data is collected using the Yahoo Finance API (yfinance) for publicly traded companies listed on:

- NYSE
- NASDAQ
- AMEX

The pipeline extracts:

- Income Statement data
- Balance Sheet data
- Cash Flow Statement data

Additional metadata is collected for each company, including:

- Sector
- Industry
- Company name
- Market information
### 2. Financial KPI Generation

Raw financial statement metrics are transformed into a comprehensive set of business KPIs.

Examples include:

- Revenue Growth
- EBITDA Margin
- Operating Margin
- Net Profit Margin
- Return on Assets (ROA)
- Return on Equity (ROE)
- Asset Turnover
- Current Ratio
- Debt-to-Equity Ratio
- Free Cash Flow Metrics

KPI definitions are maintained externally through a configurable formula library, allowing the framework to be expanded without modifying core code.

### 3. Data Preparation

The dataset undergoes extensive preprocessing:

- Removal of incomplete records
- Handling of missing values
- Outlier treatment
- Feature selection
- Standardisation using z-score scaling

The resulting feature matrix contains only comparable and machine-learning-ready variables.

### 4. Clustering Analysis

Companies are segmented using unsupervised learning techniques.

To improve cluster quality and visualisation:

- Principal Component Analysis (PCA) is applied
- Elbow Method is used for cluster selection
- Silhouette Score is calculated
- Davies-Bouldin Index is evaluated

Clustering is performed separately by sector to ensure meaningful comparisons between organisations operating in similar economic environments.

### 5. Interactive Benchmarking Tool

A Streamlit application provides an interactive interface that allows users to:

- Select sectors
- Adjust revenue filters
- Choose cluster counts
- Explore cluster compositions
- Visualise PCA projections
- Review benchmark KPI distributions

The tool transforms complex clustering outputs into an analyst-friendly decision-support system.

Project Architecture
Data Collection
       │
       ▼
Financial Statements
       │
       ▼
KPI Calculation Engine
       │
       ▼
Data Cleaning & Preparation
       │
       ▼
Feature Scaling
       │
       ▼
PCA
       │
       ▼
K-Means Clustering
       │
       ▼
Cluster Evaluation
       │
       ▼
Streamlit Benchmarking Tool
### Technologies Used
Python 3.14
Data Processing
Pandas
NumPy
Data Acquisition
yfinance
Machine Learning
Scikit-Learn
Visualisation
Plotly
Matplotlib
Application Layer
Streamlit
Development Environment
Jupyter Notebook

## Key Features
### Automated Financial Data Pipeline

Downloads and processes financial statements for hundreds of publicly traded companies.

### Dynamic KPI Engine

Business metrics are generated using externally maintained formula definitions.

### Sector-Based Clustering

Ensures peer groups remain economically meaningful.

### Interactive Cluster Exploration

Users can visually inspect clusters and identify comparable organisations.

### Benchmark Generation

Cluster-level statistics support peer comparison and performance assessment.

## Example Use Cases
### Management Consulting

Support peer selection during benchmarking engagements.

### Corporate Strategy

Identify companies with similar operating profiles.

### Investment Research

Explore groups of financially comparable firms.

### Financial Analysis

Generate objective benchmark populations for performance evaluation.

# Author

Sergio Araya

MSc Artificial Intelligence and Data Science
Keele University

Former Senior Data Analyst at McKinsey & Company

# Disclaimer

This project is intended for educational and research purposes.

Financial data is sourced from publicly available information through Yahoo Finance. Users should independently verify all financial information before making business or investment decisions.
