#!/usr/bin/env python3
"""
Product Category Classifier - Data Analysis Script
Comprehensive analysis of the multi-country product dataset

Usage:
    python analyze_data.py

This script will:
1. Load all CSV files from the dataset directory
2. Generate comprehensive statistics and insights
3. Create visualizations and save them as images
4. Generate a detailed analysis report

Requirements:
    pip install pandas numpy matplotlib seaborn wordcloud plotly
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import os
from pathlib import Path
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Optional: For word clouds and advanced plots
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    print("WordCloud not available. Install with: pip install wordcloud")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Install with: pip install plotly")

class DataAnalyzer:
    def __init__(self, data_dir="dataset", output_dir="analysis_output"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        
        self.df = None
        self.stats = {}
        
        # Set style for better plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        print(f"Data Analyzer initialized")
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
    
    def load_data(self):
        """Load all CSV files from the dataset directory"""
        print("\n🔄 Loading data files...")
        
        # Find all CSV files
        csv_files = list(self.data_dir.glob("*.csv"))
        
        # If no CSV files, try txt files (for the example data)
        if not csv_files:
            csv_files = list(self.data_dir.glob("*.txt"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV or TXT files found in {self.data_dir}")
        
        print(f"Found {len(csv_files)} files: {[f.name for f in csv_files]}")
        
        # Load all files
        dataframes = []
        file_info = []
        
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                dataframes.append(df)
                
                file_info.append({
                    'file': file_path.name,
                    'rows': len(df),
                    'locale': df['locale'].iloc[0] if 'locale' in df.columns else 'unknown',
                    'categories': df['category'].nunique() if 'category' in df.columns else 0
                })
                
                print(f"  {file_path.name}: {len(df)} rows")
                
            except Exception as e:
                print(f"  Error loading {file_path.name}: {e}")
        
        # Combine all dataframes
        self.df = pd.concat(dataframes, ignore_index=True)
        self.file_info = pd.DataFrame(file_info)
        
        print(f"\nCombined dataset: {len(self.df)} total rows")
        return self.df
    
    def basic_statistics(self):
        """Generate basic statistics about the dataset"""
        print("\nGenerating basic statistics...")
        
        df = self.df
        
        # Basic info
        basic_stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'unique_products': df['product_name'].nunique() if 'product_name' in df.columns else 0,
            'unique_categories': df['category'].nunique() if 'category' in df.columns else 0,
            'unique_brands': df['product_brand'].nunique() if 'product_brand' in df.columns else 0,
            'unique_locales': df['locale'].nunique() if 'locale' in df.columns else 0,
            'missing_product_names': df['product_name'].isna().sum() if 'product_name' in df.columns else 0,
            'missing_brands': df['product_brand'].isna().sum() if 'product_brand' in df.columns else 0,
            'missing_categories': df['category'].isna().sum() if 'category' in df.columns else 0,
        }
        
        # Missing data percentage
        missing_percentages = {}
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_percentages[f'{col}_missing_pct'] = (missing_count / len(df)) * 100
        
        self.stats.update(basic_stats)
        self.stats.update(missing_percentages)
        
        # Print summary
        print(f"Dataset Summary:")
        print(f"  • Total products: {basic_stats['total_rows']:,}")
        print(f"  • Unique categories: {basic_stats['unique_categories']}")
        print(f"  • Unique brands: {basic_stats['unique_brands']}")
        print(f"  • Countries/Locales: {basic_stats['unique_locales']}")
        print(f"  • Missing product names: {basic_stats['missing_product_names']} ({missing_percentages.get('product_name_missing_pct', 0):.1f}%)")
        print(f"  • Missing brands: {basic_stats['missing_brands']} ({missing_percentages.get('product_brand_missing_pct', 0):.1f}%)")
        
        return basic_stats
    
    def locale_analysis(self):
        """Analyze data by locale/country"""
        print("\n Analyzing data by locale...")
        
        if 'locale' not in self.df.columns:
            print(" No 'locale' column found")
            return None
        
        locale_stats = self.df.groupby('locale').agg({
            'product_name': 'count',
            'category': 'nunique',
            'product_brand': 'nunique'
        }).round(2)
        
        locale_stats.columns = ['Products', 'Categories', 'Brands']
        locale_stats = locale_stats.sort_values('Products', ascending=False)
        
        print("Products by Locale:")
        print(locale_stats)
        
        # Save to stats
        self.stats['locale_distribution'] = locale_stats.to_dict()
        
        return locale_stats
    
    def category_analysis(self):
        """Analyze product categories"""
        print("\n Analyzing product categories...")
        
        if 'category' not in self.df.columns:
            print("No 'category' column found")
            return None
        
        # Overall category distribution
        category_counts = self.df['category'].value_counts()
        
        print(f"Top 15 Categories (out of {len(category_counts)} total):")
        print(category_counts.head(15))
        
        # Category distribution by locale
        if 'locale' in self.df.columns:
            category_by_locale = pd.crosstab(self.df['category'], self.df['locale'])
            
            # Get top categories that appear in multiple countries
            categories_multi_country = category_by_locale[category_by_locale.gt(0).sum(axis=1) > 1]
            
            print(f"\nCategories appearing in multiple countries: {len(categories_multi_country)}")
        
        # Category length analysis
        category_lengths = self.df['category'].str.len()
        
        category_stats = {
            'total_categories': len(category_counts),
            'avg_category_length': category_lengths.mean(),
            'max_category_length': category_lengths.max(),
            'min_category_length': category_lengths.min(),
            'top_10_categories': category_counts.head(10).to_dict()
        }
        
        self.stats.update(category_stats)
        return category_counts
    
    def text_analysis(self):
        """Analyze text characteristics of product names"""
        print("\n Analyzing text characteristics...")
        
        if 'product_name' not in self.df.columns:
            print("No 'product_name' column found")
            return None
        
        # Clean and analyze product names
        product_names = self.df['product_name'].dropna()
        
        # Text length analysis
        text_lengths = product_names.str.len()
        word_counts = product_names.str.split().str.len()
        
        # Common words analysis
        all_text = ' '.join(product_names.astype(str).str.lower())
        words = re.findall(r'\b\w+\b', all_text)
        word_freq = Counter(words)
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'cannot'}
        filtered_word_freq = {word: count for word, count in word_freq.items() if word not in stop_words and len(word) > 2}
        
        text_stats = {
            'avg_product_name_length': text_lengths.mean(),
            'max_product_name_length': text_lengths.max(),
            'min_product_name_length': text_lengths.min(),
            'avg_words_per_product': word_counts.mean(),
            'total_unique_words': len(set(words)),
            'most_common_words': dict(Counter(filtered_word_freq).most_common(20))
        }
        
        print(f" Text Analysis Results:")
        print(f"  • Average product name length: {text_stats['avg_product_name_length']:.1f} characters")
        print(f"  • Average words per product: {text_stats['avg_words_per_product']:.1f}")
        print(f"  • Total unique words: {text_stats['total_unique_words']:,}")
        print(f"  • Most common words: {list(Counter(filtered_word_freq).most_common(10))}")
        
        self.stats.update(text_stats)
        return text_stats
    
    def brand_analysis(self):
        """Analyze brand distribution and characteristics"""
        print("\nAnalyzing brands...")
        
        if 'product_brand' not in self.df.columns:
            print("No 'product_brand' column found")
            return None
        
        # Brand distribution
        brands = self.df['product_brand'].dropna()
        brand_counts = brands.value_counts()
        
        # Brands by locale
        if 'locale' in self.df.columns:
            brands_by_locale = pd.crosstab(self.df['product_brand'].fillna('Unknown'), self.df['locale'])
            global_brands = brands_by_locale[brands_by_locale.gt(0).sum(axis=1) > 2]  # Brands in 3+ countries
        
        brand_stats = {
            'total_brands': len(brand_counts),
            'brands_with_multiple_products': (brand_counts > 1).sum(),
            'top_10_brands': brand_counts.head(10).to_dict(),
            'single_product_brands': (brand_counts == 1).sum()
        }
        
        print(f"Brand Analysis:")
        print(f"  • Total unique brands: {brand_stats['total_brands']}")
        print(f"  • Brands with multiple products: {brand_stats['brands_with_multiple_products']}")
        print(f"  • Single-product brands: {brand_stats['single_product_brands']}")
        print(f"  • Top brands: {list(brand_counts.head(5).index)}")
        
        self.stats.update(brand_stats)
        return brand_counts
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\nCreating visualizations...")
        
        # Set up the plotting style
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Dataset Overview
        self._plot_dataset_overview()
        
        # 2. Category Analysis Plots
        self._plot_category_analysis()
        
        # 3. Locale Analysis Plots
        self._plot_locale_analysis()
        
        # 4. Text Analysis Plots
        self._plot_text_analysis()
        
        # 5. Brand Analysis Plots
        self._plot_brand_analysis()
        
        # 6. Create word cloud if available
        if WORDCLOUD_AVAILABLE:
            self._create_wordcloud()
        
        print(f"All visualizations saved to {self.output_dir}/plots/")
    
    def _plot_dataset_overview(self):
        """Create dataset overview plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dataset Overview', fontsize=16, fontweight='bold')
        
        # 1. Data completeness
        missing_data = self.df.isnull().sum()
        missing_pct = (missing_data / len(self.df)) * 100
        
        axes[0, 0].bar(missing_pct.index, missing_pct.values, color='coral')
        axes[0, 0].set_title('Missing Data Percentage by Column')
        axes[0, 0].set_ylabel('Missing %')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Records per file
        if hasattr(self, 'file_info'):
            axes[0, 1].bar(self.file_info['file'], self.file_info['rows'], color='skyblue')
            axes[0, 1].set_title('Records per File')
            axes[0, 1].set_ylabel('Number of Records')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Locale distribution
        if 'locale' in self.df.columns:
            locale_counts = self.df['locale'].value_counts()
            axes[1, 0].pie(locale_counts.values, labels=locale_counts.index, autopct='%1.1f%%')
            axes[1, 0].set_title('Distribution by Locale')
        
        # 4. Categories per locale
        if 'locale' in self.df.columns and 'category' in self.df.columns:
            categories_per_locale = self.df.groupby('locale')['category'].nunique().sort_values(ascending=True)
            axes[1, 1].barh(categories_per_locale.index, categories_per_locale.values, color='lightgreen')
            axes[1, 1].set_title('Unique Categories per Locale')
            axes[1, 1].set_xlabel('Number of Categories')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'dataset_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_category_analysis(self):
        """Create category analysis plots"""
        if 'category' not in self.df.columns:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Category Analysis', fontsize=16, fontweight='bold')
        
        # 1. Top categories
        top_categories = self.df['category'].value_counts().head(15)
        axes[0, 0].barh(top_categories.index, top_categories.values, color='lightcoral')
        axes[0, 0].set_title('Top 15 Categories')
        axes[0, 0].set_xlabel('Number of Products')
        
        # 2. Category distribution (log scale)
        category_counts = self.df['category'].value_counts()
        axes[0, 1].hist(category_counts.values, bins=30, color='lightblue', alpha=0.7)
        axes[0, 1].set_yscale('log')
        axes[0, 1].set_title('Category Frequency Distribution (Log Scale)')
        axes[0, 1].set_xlabel('Products per Category')
        axes[0, 1].set_ylabel('Number of Categories')
        
        # 3. Category name lengths
        if self.df['category'].dtype == 'object':
            category_lengths = self.df['category'].str.len().dropna()
            axes[1, 0].hist(category_lengths, bins=20, color='gold', alpha=0.7)
            axes[1, 0].set_title('Distribution of Category Name Lengths')
            axes[1, 0].set_xlabel('Category Name Length (characters)')
            axes[1, 0].set_ylabel('Frequency')
        
        # 4. Categories by locale heatmap
        if 'locale' in self.df.columns:
            # Get top 10 categories and all locales
            top_cats = self.df['category'].value_counts().head(10).index
            category_locale_matrix = pd.crosstab(self.df['category'], self.df['locale'])
            top_cat_matrix = category_locale_matrix.loc[top_cats]
            
            sns.heatmap(top_cat_matrix, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1, 1])
            axes[1, 1].set_title('Top 10 Categories by Locale')
            axes[1, 1].set_xlabel('Locale')
            axes[1, 1].set_ylabel('Category')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'category_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_locale_analysis(self):
        """Create locale-specific analysis plots"""
        if 'locale' not in self.df.columns:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Locale Analysis', fontsize=16, fontweight='bold')
        
        # 1. Products per locale
        locale_counts = self.df['locale'].value_counts()
        axes[0, 0].bar(locale_counts.index, locale_counts.values, color='lightseagreen')
        axes[0, 0].set_title('Products per Locale')
        axes[0, 0].set_ylabel('Number of Products')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Brands per locale
        if 'product_brand' in self.df.columns:
            brands_per_locale = self.df.groupby('locale')['product_brand'].nunique()
            axes[0, 1].bar(brands_per_locale.index, brands_per_locale.values, color='mediumpurple')
            axes[0, 1].set_title('Unique Brands per Locale')
            axes[0, 1].set_ylabel('Number of Brands')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Average product name length by locale
        if 'product_name' in self.df.columns:
            # avg_length_by_locale = self.df.groupby('locale')['product_name'].str.len().mean()
            avg_length_by_locale = self.df.groupby('locale')['product_name'].apply(lambda x: x.str.len().mean())

            axes[1, 0].bar(avg_length_by_locale.index, avg_length_by_locale.values, color='sandybrown')
            axes[1, 0].set_title('Average Product Name Length by Locale')
            axes[1, 0].set_ylabel('Average Length (characters)')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Category diversity by locale
        if 'category' in self.df.columns:
            categories_per_locale = self.df.groupby('locale')['category'].nunique()
            total_products_per_locale = self.df['locale'].value_counts()
            diversity_ratio = categories_per_locale / total_products_per_locale
            
            axes[1, 1].bar(diversity_ratio.index, diversity_ratio.values, color='lightcoral')
            axes[1, 1].set_title('Category Diversity Ratio by Locale')
            axes[1, 1].set_ylabel('Categories / Products')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'locale_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_text_analysis(self):
        """Create text analysis plots"""
        if 'product_name' not in self.df.columns:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Text Analysis', fontsize=16, fontweight='bold')
        
        product_names = self.df['product_name'].dropna()
        
        # 1. Product name lengths
        name_lengths = product_names.str.len()
        axes[0, 0].hist(name_lengths, bins=30, color='lightblue', alpha=0.7)
        axes[0, 0].set_title('Distribution of Product Name Lengths')
        axes[0, 0].set_xlabel('Length (characters)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(name_lengths.mean(), color='red', linestyle='--', label=f'Mean: {name_lengths.mean():.1f}')
        axes[0, 0].legend()
        
        # 2. Word count per product
        word_counts = product_names.str.split().str.len()
        axes[0, 1].hist(word_counts, bins=20, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('Words per Product Name')
        axes[0, 1].set_xlabel('Number of Words')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(word_counts.mean(), color='red', linestyle='--', label=f'Mean: {word_counts.mean():.1f}')
        axes[0, 1].legend()
        
        # 3. Most frequent words
        all_text = ' '.join(product_names.astype(str).str.lower())
        words = re.findall(r'\b\w+\b', all_text)
        word_freq = Counter(words)
        
        # Filter out common stop words and short words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        filtered_words = {word: count for word, count in word_freq.items() 
                         if word not in stop_words and len(word) > 2}
        
        top_words = Counter(filtered_words).most_common(15)
        words_df = pd.DataFrame(top_words, columns=['word', 'count'])
        
        axes[1, 0].barh(words_df['word'], words_df['count'], color='gold')
        axes[1, 0].set_title('Top 15 Most Frequent Words')
        axes[1, 0].set_xlabel('Frequency')
        
        # 4. Character distribution
        char_counts = Counter(all_text.replace(' ', ''))
        top_chars = Counter(char_counts).most_common(15)
        chars_df = pd.DataFrame(top_chars, columns=['char', 'count'])
        
        axes[1, 1].bar(chars_df['char'], chars_df['count'], color='lightcoral')
        axes[1, 1].set_title('Most Frequent Characters')
        axes[1, 1].set_xlabel('Character')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'text_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_brand_analysis(self):
        """Create brand analysis plots"""
        if 'product_brand' not in self.df.columns:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Brand Analysis', fontsize=16, fontweight='bold')
        
        brands = self.df['product_brand'].dropna()
        brand_counts = brands.value_counts()
        
        # 1. Top brands
        top_brands = brand_counts.head(15)
        axes[0, 0].barh(top_brands.index, top_brands.values, color='skyblue')
        axes[0, 0].set_title('Top 15 Brands by Product Count')
        axes[0, 0].set_xlabel('Number of Products')
        
        # 2. Brand distribution
        axes[0, 1].hist(brand_counts.values, bins=30, color='lightgreen', alpha=0.7)
        axes[0, 1].set_yscale('log')
        axes[0, 1].set_title('Brand Product Count Distribution (Log Scale)')
        axes[0, 1].set_xlabel('Products per Brand')
        axes[0, 1].set_ylabel('Number of Brands')
        
        # 3. Brand name lengths
        brand_lengths = brands.str.len()
        axes[1, 0].hist(brand_lengths, bins=20, color='orange', alpha=0.7)
        axes[1, 0].set_title('Brand Name Length Distribution')
        axes[1, 0].set_xlabel('Length (characters)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(brand_lengths.mean(), color='red', linestyle='--', 
                          label=f'Mean: {brand_lengths.mean():.1f}')
        axes[1, 0].legend()
        
        # 4. Brands across locales
        if 'locale' in self.df.columns:
            brand_locale_counts = self.df.groupby('product_brand')['locale'].nunique().sort_values(ascending=False)
            global_brands = brand_locale_counts[brand_locale_counts > 1].head(10)
            
            if len(global_brands) > 0:
                axes[1, 1].bar(range(len(global_brands)), global_brands.values, color='mediumpurple')
                axes[1, 1].set_title('Brands Present in Multiple Locales')
                axes[1, 1].set_xlabel('Brand Rank')
                axes[1, 1].set_ylabel('Number of Locales')
                axes[1, 1].set_xticks(range(len(global_brands)))
                axes[1, 1].set_xticklabels(global_brands.index, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'brand_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_wordcloud(self):
        """Create word cloud from product names"""
        if not WORDCLOUD_AVAILABLE or 'product_name' not in self.df.columns:
            return
        
        print("Creating word cloud...")
        
        # Prepare text for word cloud
        product_names = self.df['product_name'].dropna()
        all_text = ' '.join(product_names.astype(str).str.lower())
        
        # Create word cloud
        wordcloud = WordCloud(
            width=1600, height=800,
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(all_text)
        
        plt.figure(figsize=(16, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Product Names Word Cloud', fontsize=20, fontweight='bold', pad=20)
        plt.savefig(self.output_dir / 'plots' / 'wordcloud.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self):
        """Generate a comprehensive analysis report"""
        print("\nGenerating analysis report...")
        
        report_path = self.output_dir / 'reports' / 'data_analysis_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Product Category Dataset - Analysis Report\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write(f"This report presents a comprehensive analysis of the product category dataset containing **{self.stats.get('total_rows', 0):,}** products across **{self.stats.get('unique_locales', 0)}** countries/locales.\n\n")
            
            f.write("### Key Findings\n")
            f.write(f"- **Categories**: {self.stats.get('unique_categories', 0)} unique product categories\n")
            f.write(f"- **Brands**: {self.stats.get('unique_brands', 0)} unique brands\n")
            f.write(f"- **Countries**: {self.stats.get('unique_locales', 0)} different locales\n")
            f.write(f"- **Data Quality**: {self.stats.get('missing_product_names', 0)} missing product names ({self.stats.get('product_name_missing_pct', 0):.1f}%)\n")
            f.write(f"- **Text Complexity**: Average {self.stats.get('avg_words_per_product', 0):.1f} words per product name\n\n")
            
            f.write("## Dataset Overview\n\n")
            f.write("### Basic Statistics\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Total Records | {self.stats.get('total_rows', 0):,} |\n")
            f.write(f"| Unique Products | {self.stats.get('unique_products', 0):,} |\n")
            f.write(f"| Unique Categories | {self.stats.get('unique_categories', 0):,} |\n")
            f.write(f"| Unique Brands | {self.stats.get('unique_brands', 0):,} |\n")
            f.write(f"| Countries/Locales | {self.stats.get('unique_locales', 0)} |\n\n")
            
            f.write("### Data Quality Assessment\n")
            f.write(f"| Column | Missing Count | Missing % |\n")
            f.write(f"|--------|---------------|----------|\n")
            for col in ['product_name', 'product_brand', 'category']:
                missing_key = f'{col}_missing_pct'
                if missing_key in self.stats:
                    missing_key_name = f"{col.split('_')[0]}_missing_count"
                    f.write(
                        f"| {col} | {self.stats.get(missing_key_name, 'N/A')} | {self.stats[missing_key]:.1f}% |\n"
                    )
                    # f.write(f"| {col} | {self.stats.get(f'{col.split(\"_\")[0]}_missing_count', 'N/A')} | {self.stats[missing_key]:.1f}% |\n")
            f.write("\n")
            
            if hasattr(self, 'file_info'):
                f.write("### Files Overview\n")
                f.write("| File | Records | Locale | Categories |\n")
                f.write("|------|---------|--------|------------|\n")
                for _, row in self.file_info.iterrows():
                    f.write(f"| {row['file']} | {row['rows']:,} | {row['locale']} | {row['categories']} |\n")
                f.write("\n")
            
            f.write("## Category Analysis\n\n")
            f.write("### Category Distribution Insights\n")
            f.write(f"- **Total Categories**: {self.stats.get('unique_categories', 0)}\n")
            f.write(f"- **Average Category Name Length**: {self.stats.get('avg_category_length', 0):.1f} characters\n")
            f.write(f"- **Longest Category Name**: {self.stats.get('max_category_length', 0)} characters\n\n")
            
            if 'top_10_categories' in self.stats:
                f.write("### Top 10 Categories\n")
                f.write("| Rank | Category | Product Count |\n")
                f.write("|------|----------|---------------|\n")
                for i, (cat, count) in enumerate(self.stats['top_10_categories'].items(), 1):
                    f.write(f"| {i} | {cat} | {count:,} |\n")
                f.write("\n")
            
            f.write("## Text Analysis\n\n")
            f.write("### Product Name Characteristics\n")
            f.write(f"- **Average Length**: {self.stats.get('avg_product_name_length', 0):.1f} characters\n")
            f.write(f"- **Average Words**: {self.stats.get('avg_words_per_product', 0):.1f} words per product\n")
            f.write(f"- **Vocabulary Size**: {self.stats.get('total_unique_words', 0):,} unique words\n\n")
            
            if 'most_common_words' in self.stats:
                f.write("### Most Frequent Words\n")
                f.write("| Rank | Word | Frequency |\n")
                f.write("|------|------|----------|\n")
                for i, (word, freq) in enumerate(list(self.stats['most_common_words'].items())[:10], 1):
                    f.write(f"| {i} | {word} | {freq:,} |\n")
                f.write("\n")
            
            f.write("## Brand Analysis\n\n")
            f.write("### Brand Distribution\n")
            f.write(f"- **Total Brands**: {self.stats.get('total_brands', 0):,}\n")
            f.write(f"- **Multi-product Brands**: {self.stats.get('brands_with_multiple_products', 0):,}\n")
            f.write(f"- **Single-product Brands**: {self.stats.get('single_product_brands', 0):,}\n\n")
            
            if 'top_10_brands' in self.stats:
                f.write("### Top 10 Brands\n")
                f.write("| Rank | Brand | Product Count |\n")
                f.write("|------|-------|---------------|\n")
                for i, (brand, count) in enumerate(self.stats['top_10_brands'].items(), 1):
                    f.write(f"| {i} | {brand} | {count:,} |\n")
                f.write("\n")
            
            f.write("## Country/Locale Analysis\n\n")
            if 'locale_distribution' in self.stats:
                f.write("### Products by Locale\n")
                f.write("| Locale | Products | Categories | Brands |\n")
                f.write("|--------|----------|------------|--------|\n")
                for locale, stats in self.stats['locale_distribution']['Products'].items():
                    products = self.stats['locale_distribution']['Products'][locale]
                    categories = self.stats['locale_distribution']['Categories'][locale]
                    brands = self.stats['locale_distribution']['Brands'][locale]
                    f.write(f"| {locale} | {products:,} | {categories} | {brands} |\n")
                f.write("\n")
            
            f.write("## ML\n\n")
            
            f.write("## Visualizations\n\n")
            f.write("The following visualizations have been generated and saved in the `plots/` directory:\n\n")
            f.write("1. **dataset_overview.png** - Overall dataset statistics and distribution\n")
            f.write("2. **category_analysis.png** - Category frequency and distribution analysis\n")
            f.write("3. **locale_analysis.png** - Country-wise data analysis\n")
            f.write("4. **text_analysis.png** - Product name text characteristics\n")
            f.write("5. **brand_analysis.png** - Brand distribution and frequency analysis\n")
            if WORDCLOUD_AVAILABLE:
                f.write("6. **wordcloud.png** - Word cloud of most frequent terms\n")
            f.write("\n")
        
        print(f"Report saved to: {report_path}")
        
        # Also create a summary JSON file
        summary_path = self.output_dir / 'reports' / 'analysis_summary.json'
        import json
        with open(summary_path, 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)
        
        print(f"Summary JSON saved to: {summary_path}")
        
        return report_path
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 Starting complete data analysis...")
        
        try:
            # Load data
            self.load_data()
            
            # Basic statistics
            self.basic_statistics()
            
            # Detailed analyses
            self.locale_analysis()
            self.category_analysis()
            self.text_analysis()
            self.brand_analysis()
            
            # Create visualizations
            self.create_visualizations()
            
            # Generate report
            report_path = self.generate_report()
            
            print("\nAnalysis Complete!")
            print(f"All outputs saved to: {self.output_dir}")
            print(f"Plots available in: {self.output_dir}/plots/")
            print(f"Reports available in: {self.output_dir}/reports/")
            print(f"\nMain report: {report_path}")
            
            return True
            
        except Exception as e:
            print(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function to run the analysis"""
    print("=" * 60)
    print("PRODUCT CATEGORY DATASET ANALYSIS")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = DataAnalyzer(
        data_dir="dataset",  # Adjust path as needed
        output_dir="analysis_output"
    )
    
    # Run complete analysis
    success = analyzer.run_complete_analysis()
    
    if success:
        print("\nAnalysis completed successfully!")
        print("\nYou can now:")
        print("   1. Check the visualizations in analysis_output/plots/")
        print("   2. Read the comprehensive report in analysis_output/reports/")
        print("   3. Use the insights for your ML model development")
        print("\n Include key findings in your project documentation!")
    else:
        print("\n Analysis failed. Check the error messages above.")

if __name__ == "__main__":
    # Add requirements for this script
    required_packages = [
        "pandas", "numpy", "matplotlib", "seaborn", 
        "wordcloud", "plotly"  # Optional but recommended
    ]
    
    print("Required packages for this analysis script:")
    print("pip install pandas numpy matplotlib seaborn wordcloud plotly")
    print("\n")
    
    main()