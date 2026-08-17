import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple
import glob
import re

class CategoryUnifier:
    def __init__(self):
        self.category_mappings = {
            'vin': 'wine',           
            'vino': 'wine',          
            'Vino': 'wine',          
            'wine': 'wine',          
            'Wine': 'wine',          
            'Vin': 'wine',          
            'wijn': 'wine',         
            'vinho tinto': 'wine', 
            'vinho branco': 'wine', 
            'vinho tinto': 'wine', 
            'Vinho': 'wine',
            'Spumante': 'wine',
            'Prosecco': 'wine',
            'Vino bianco': 'wine',
            'Vino rosso': 'wine',
            'Lambrusco': 'wine',
            'Chianti': 'wine',
            'Champagne': 'wine',
            'champagne': 'wine',
            "vino blanco": 'wine',
            "vino crianza": 'wine',
            "vino reserva": 'wine',
            "vino rosado": 'wine',
            "vino tinto": 'wine',
            "vino verdejo": 'wine',
            "red wine": 'wine',
            "white wine": 'wine',
            "rose wine": 'wine',

            'fromage': 'cheese',     
            'Fromage': 'cheese',     
            'formaggio': 'cheese',   
            'Formaggio': 'cheese',   
            'queso': 'cheese',       
            'Queso': 'cheese',       
            'cheese': 'cheese',      
            'Cheese': 'cheese',      
            'käse': 'cheese',        
            'queijo': 'cheese',
            'Gorgonzola': 'cheese',
            'Parmigiano': 'cheese',
            'Formaggio Grattugiato': 'cheese',
            'Mozzarella': 'cheese',
            'mozzarella': 'cheese',
            'Certosa': 'cheese',
            'Emmental': 'cheese',
            'Stracchino': 'cheese',
            'Ricotta': 'cheese',
            'Mozzarella di bufala': 'cheese',
            'Pecorino': 'cheese',
            'Sottilette': 'cheese',
            'Asiago': 'cheese',
            'Provolone': 'cheese',
            'Robiola': 'cheese',
            'brie': 'cheese',
            'mascarpone': 'cheese',
            'Mascarpone': 'cheese',
            'Galbanino': 'cheese',
            'Formaggio spalmabile': 'cheese',
            
            'coca-cola': 'beverages',
            'Coca Cola': 'beverages',
            'Coca-cola': 'beverages',
            'Coca cola': 'beverages',
            'beverages': 'beverages',
            'Beverages': 'beverages',
            'Bebidas': 'beverages',
            'bebidas': 'beverages',
            'boissons': 'beverages',
            'Getränke': 'beverages',
            'Getranke': 'beverages',
            'Bibite': 'beverages',
            'Energy drink': 'beverages',
            'energy drink': 'beverages',
            'Energy Drink': 'beverages',
            'Bevande analcoliche': 'beverages',
            'coca cola zero': 'beverages',
            'Bitter': 'beverages',

            'Lavatrice': 'washing machine',
            'lavatrice': 'washing machine',
            'Washing machine': 'washing machine',
            'washing machine': 'washing machine',

            'Spirits': 'spirits',
            'Liquore': 'spirits',
            'liqueur': 'spirits',
            'spirits': 'spirits',
            'vodka': 'spirits',
            'Vodka': 'spirits',
            'Whisky': 'spirits',
            'whisky': 'spirits',
            'whiskey': 'spirits',
            'Rum': 'spirits',
            'rum': 'spirits',
            'Gin': 'spirits',
            
            'juegos': 'games',       
            'plush toys': 'games',
            'toys': 'games',
            'jeux': 'games',         
            'games': 'games',        
            'giochi': 'games',       
            'spiele': 'games',       
            'jogos': 'games',
            'Monopoly': 'games',
            'monopoly': 'games',
            'Giochi per bambini': 'games',
            'peluche': 'games',
            'Peluche': 'games',
            'Puzzle': 'games',
            'puzzle': 'games',
            'Ps5': 'games',
            'PS5': 'games',
            'Nintendo Switch': 'games',
            'Hot Wheels': 'games',
            'Baby Doll': 'games',
            'car games': 'games',
            'console': 'games',
            'Console': 'games',
            
            'cerveza': 'beer',       
            'bière': 'beer',         
            'birra': 'beer',         
            'beer': 'beer',          
            'bier': 'beer',          
            'cerveja': 'beer',
            'birra moretti': 'beer',
            'birra peroni': 'beer',
            
            'pharmacy': 'pharmacy',
            'Pharmacy': 'pharmacy',
            'farmacia': 'pharmacy',
            'pharmacie': 'pharmacy',
            'apotheke': 'pharmacy',
            'produtos farmaceuticos': 'pharmacy',
            'parapharmacie': 'pharmacy',
            'productos farmacéuticos': 'pharmacy',
            'medicines': 'pharmacy',
            "medicamentos": 'pharmacy',
            "medicine": 'pharmacy',
            
            'carne': 'meat',
            'Carne': 'meat',
            'steak': 'meat',
            'Steak': 'meat',
            'viande': 'meat',
            'meat': 'meat',
            'fleisch': 'meat',
            'carnes': 'meat',
            'Beef': 'meat',
            'beef': 'meat',
            'Roast Beef': 'meat',
            'roast beef': 'meat',
            'salami': 'meat',
            'Salami': 'meat',
            'Cotolette di pollo': 'meat',
            'pollo': 'meat',
            'Pollo': 'meat',
            'bacon': 'meat',
            'Bacon': 'meat',
            'pancetta': 'meat',
            'salumi': 'meat',
            'Prosciutto crudo': 'meat',
            'Prosciutto cotto': 'meat',
            'Salsicce': 'meat',
            'Hamburger': 'meat',
            'Wurstel': 'meat',
            'Cotoletta': 'meat',
            'Vitello': 'meat',
            'Petto di pollo': 'meat',
            'Bistecca': 'meat',
            'Mortadella': 'meat',
            'Salame': 'meat',
            'Salumi': 'meat',
            'Speck': 'meat',     
            'Tacchino': 'meat',
            'Carne macinata': 'meat',
            'cordon bleu': 'meat',
            'Prosciutto di Parma': 'meat',

            'dog food': 'pet food',
            'Dog food': 'pet food',
            'Cibo per cani': 'pet food',
            'cibo per cani': 'pet food',
            'Cibo per gatti': 'pet food',
            'cibo per gatti': 'pet food',
            'cat food': 'pet food',
            'Cat food': 'pet food',
            'Pet care': 'pet food',

            'Books': 'books',
            'books': 'books',
            'libros': 'books',
            'Libros': 'books',
            'livres': 'books',
            'Livres': 'books',
            'Libri': 'books',
            'libri': 'books',

            'Patatine': 'potatoes',
            'potato chips': 'potatoes',
            'Patatine fritte': 'potatoes',
            'patatine': 'potatoes',
            'Patatas fritas': 'potatoes',
            'patatas fritas': 'potatoes',
            'chips': 'potatoes',
            'Chips': 'potatoes',
            'Pommes frites': 'potatoes',
            'pommes frites': 'potatoes',
            'Kartoffelchips': 'potatoes',
            'kartoffelchips': 'potatoes',
            'Patate': 'potatoes',
            'patate': 'potatoes',
            'Patate surgelate': 'potatoes',

            'Pasta Barilla': 'pasta',
            "Pasta all'uovo": 'pasta',
            'Gnocchi': 'pasta',
            'gnocchi': 'pasta',
            'macarrão': 'pasta',
            'Pasta di semola': 'pasta',
            'Pasta': 'pasta',
            'pasta': 'pasta',
            'pâtes': 'pasta',
            'pasta alimenticia': 'pasta',
            'spaghetti': 'pasta',
            'Spaghetti': 'pasta',
            'Penne': 'pasta',
            'Fusilli': 'pasta',
            'Pasta fresca': 'pasta',
            'Lasagne': 'pasta',
            'Tortellini': 'pasta',

            'pain': 'bread',
            'pan': 'bread',
            'pane': 'bread',
            'bread': 'bread',
            'brot': 'bread',
            'Pancarrè': 'bread',
            'Panini': 'bread',
            'Piadine': 'bread',
            'Focaccia': 'bread',
            'Pan Bauletto': 'bread',
            'Baguette': 'bread',

            'Alimenti': 'food',
            'food': 'food',
            "alimentación": 'food',
            "alimentation": 'food', 
            "alimentazione": 'food', 
            "alimentação": 'food',

            'uova': 'eggs',
            'Uova': 'eggs',
            'eggs': 'eggs',
            'Eggs': 'eggs',
            'Oeufs': 'eggs',
            'Ovos': 'eggs',
            'Eier': 'eggs',
            'Huevos': 'eggs',

            'Accessori cucina': 'kitchen accessories',
            "air fryer": 'kitchen accessories',
            "friggitrice": 'kitchen accessories',
            'frigoriferi': 'kitchen accessories',
            'accessoires de cuisine': 'kitchen accessories',    
            'accessori casa': 'kitchen accessories',
            'toaster': 'kitchen accessories',
            'fridge': 'kitchen accessories',
            'Electric Oven': 'kitchen accessories',
            'Fridge': 'kitchen accessories',
            'accessori cucina': 'kitchen accessories',
            'utensilios de cocina': 'kitchen accessories',
            'kitchen accessories': 'kitchen accessories',
            'Kitchen Accessories': 'kitchen accessories',
            'Kitchen accessories': 'kitchen accessories',
            'Kettle': 'kitchen accessories',
            'Dishwasher': 'kitchen accessories',
            'Kitchen appliances': 'kitchen accessories',
            'oven': 'kitchen accessories',
            'forno': 'kitchen accessories',

            'Papel Higiénico': 'toilet paper',
            'papel higiénico': 'toilet paper',
            'toilet paper': 'toilet paper',
            'Toilet paper': 'toilet paper',
            'Carta igienica': 'toilet paper',

            'Biscotti': 'biscuits',
            'biscotti': 'biscuits',
            'Oro Saiwa': 'biscuits',
            'Biscuits': 'biscuits',
            'biscuits': 'biscuits',
            'Cookies': 'biscuits',
            'cookies': 'biscuits',
            'Galletas': 'biscuits',
            'Gallette': 'biscuits',

            'Pasticceria': 'pastry',
            'pasticceria': 'pastry',
            'Pastry': 'pastry',
            'pastry': 'pastry',
            'Pâtisserie': 'pastry',
            'pâtisserie': 'pastry',
            'Bäckerei': 'pastry',
            'bäckerei': 'pastry',
            'Bakery': 'pastry',
            'bakery': 'pastry',
            'Panettone': 'pastry',
            'panettone': 'pastry',
            'Pandoro': 'pastry',
            'Cornetti': 'pastry',
            'Caramelle': 'pastry',
            'Dessert': 'pastry',
            'brioche': 'pastry',

            'Profumi': 'perfumes',
            'profumi': 'perfumes',
            'Perfumes': 'perfumes',
            'perfumes': 'perfumes',

            'pizza': 'pizza',
            'Pizza': 'pizza',
            'Pizza Ristorante': 'pizza',
            'pizza Ristorante': 'pizza',

            'kiwi': 'fruit',
            'Arance': 'fruit',
            'Mandarini': 'fruit',
            'Kiwi': 'fruit',
            'oranges': 'fruit',
            'Oranges': 'fruit',
            'frutta': 'fruit',
            'Frutta': 'fruit',
            'fruit': 'fruit',
            'Ananas': 'fruit',
            'Fruit': 'fruit',
            'fruits': 'fruit',
            'Fruits': 'fruit',
            'manzana': 'fruit',
            'Manzana': 'fruit',
            'apple': 'fruit',
            'Apple': 'fruit',
            'banane': 'fruit',
            'bananas': 'fruit',
            'banana': 'fruit',
            'Banane': 'fruit',
            'Mele': 'fruit',
            'Pere': 'fruit',
            'Prugne': 'fruit',
            
            'lait': 'milk',
            'leche': 'milk',
            'latte': 'milk',
            'milk': 'milk',
            'milch': 'milk',
            'leite': 'milk',
            'almond milk': 'milk',
            'Latte Granarolo': 'milk',
            'Latte parzialmente scremato': 'milk',
            'Latte intero': 'milk',
            'Latte uht': 'milk',
            
            'café': 'coffee',
            'Nescafè': 'coffee',
            'caffè': 'coffee',
            'coffee': 'coffee',
            'kaffee': 'coffee',
            'cafè': 'coffee',
            'Caffè Kimbo': 'coffee',
            'Caffè': 'coffee',
            'Capsule caffè': 'coffee',
            
            'eau': 'water',
            'agua': 'water',
            'acqua': 'water',
            'water': 'water',
            'wasser': 'water',
            'água': 'water',
            'Acqua San Benedetto': 'water',
            "Acqua Sant'Anna": 'water',
            
            'huile': 'oil',
            'aceite': 'oil',
            'olio': 'oil',
            'oil': 'oil',
            'öl': 'oil',
            'Olio di semi': 'oil',
            "olio per friggere": 'oil',
            "olio extravergine di oliva": 'oil',
            
            'poisson': 'fish',
            'pescado': 'fish',
            'pesce': 'fish',
            'orata': 'fish',
            'Alici': 'fish',
            'fish': 'fish',
            'fisch': 'fish',
            'Bastoncini di pesce': 'fish',
            'Tonno': 'fish',
            'Gamberi': 'fish',
            'Filetti di salmone': 'fish',
            'salmon': 'fish',
            'Vongole': 'fish',
            'Baccalà': 'fish',
            'Frutti di mare': 'fish',
            'Tonno Rio mare': 'fish',
            'Salmone affumicato': 'fish',
            'Merluzzo': 'fish',
            'Filetti di sgombro': 'fish',    
            'Sgombro': 'fish',
            'Bastoncini Findus': 'fish',
            'Insalata di mare': 'fish',
            'Platessa': 'fish',
            'Filetti di merluzzo': 'fish',
            'Sushi': 'fish',
            'sushi': 'fish',
            'Cozze': 'fish',
            'Trota': 'fish',

            'tablet android': 'tablet', 
            'Tablet android': 'tablet',
            'tablet Android': 'tablet',
            'tablet': 'tablet',
            'Tablet': 'tablet',
            'tablet Samsung': 'tablet',
            'tablet samsung': 'tablet',
            'ipad': 'tablet',
            'Ipad': 'tablet',
            'iPad': 'tablet',
        
            'Samsung Tv': 'tv',
            'Tv': 'tv',
            'TV': 'tv',
            'tv': 'tv',
            'Samsung TV': 'tv',
            'Smart Tv': 'tv',
            'Smart tv': 'tv',
            'smart tv': 'tv',
            'Smart TV': 'tv',
            'Tv 4k': 'tv',
            'Tv 4K': 'tv',
            'tv led': 'tv',
            'Tv led': 'tv',
            'Monitor tv': 'tv',
            'monitor tv': 'tv',
            'monitor': 'tv',
            'Monitor': 'tv',

            'iphone': 'iphone',
            'iPhone': 'iphone',
            'Iphone': 'iphone',
            'IPhone': 'iphone',
            'iPhone': 'iphone',
            'iphone 13': 'iphone',
            'iPhone 13': 'iphone',
            'Iphone 13': 'iphone',
            'Iphone 12': 'iphone',
            'iphone 12': 'iphone',
            'iPhone 12': 'iphone',

            'Smartphone': 'smartphone',
            'smartphone': 'smartphone',
            'Smartphone Samsung': 'smartphone', 
            'Smartphone android': 'smartphone',
            'Samsung Galaxy': 'smartphone',
            'smartphones': 'smartphone',

        }
        
        self.case_normalize_first = True
        
    def normalize_category(self, category):
        """
        Normalize a category name to its unified form
        """
        if not category or pd.isna(category):
            return 'unknown'
        
        category = str(category).strip()
        
        if category in self.category_mappings:
            return self.category_mappings[category]
        
        category_lower = category.lower()
        if category_lower in self.category_mappings:
            return self.category_mappings[category_lower]
        
        return category_lower
    
    def unify_categories(self, categories):
        """
        Unify a list of categories
        
        Args:
            categories: List of category names
            
        Returns:
            List of unified category names
        """
        return [self.normalize_category(cat) for cat in categories]
    
    def get_mapping_stats(self, categories):
        """
        Get statistics about the unification process
        
        Args:
            categories: List of original categories
            
        Returns:
            Dict with unification statistics
        """
        from collections import Counter
        
        original_counts = Counter(categories)
        unified_categories = self.unify_categories(categories)
        unified_counts = Counter(unified_categories)
        
        unified_groups = {}
        for orig_cat in set(categories):
            unified_cat = self.normalize_category(orig_cat)
            if unified_cat not in unified_groups:
                unified_groups[unified_cat] = []
            unified_groups[unified_cat].append(orig_cat)
        
        merged_groups = {k: v for k, v in unified_groups.items() if len(v) > 1}
        
        stats = {
            'original_categories': len(original_counts),
            'unified_categories': len(unified_counts),
            'reduction_count': len(original_counts) - len(unified_counts),
            'reduction_percentage': (len(original_counts) - len(unified_counts)) / len(original_counts) * 100,
            'merged_groups': merged_groups,
            'top_unified_categories': unified_counts.most_common(10)
        }
        
        return stats
    
    def print_unification_report(self, categories):
        """Print a detailed unification report"""
        stats = self.get_mapping_stats(categories)
        
        print("Category Unification Report")
        print("=" * 50)
        print(f"Original categories: {stats['original_categories']:,}")
        print(f"Unified categories: {stats['unified_categories']:,}")
        print(f"Categories merged: {stats['reduction_count']:,} ({stats['reduction_percentage']:.1f}% reduction)")
        
        print(f"\nTop 10 Unified Categories:")
        for i, (cat, count) in enumerate(stats['top_unified_categories'], 1):
            print(f"  {i:2d}. {cat}: {count:,} samples")
        
        print(f"\n🔄 Merged Category Groups:")
        for unified_cat, original_cats in stats['merged_groups'].items():
            if len(original_cats) > 1:
                sample_counts = []
                total_samples = 0
                from collections import Counter
                orig_counts = Counter(categories)
                
                for orig_cat in original_cats:
                    count = orig_counts[orig_cat]
                    sample_counts.append(f"{orig_cat}({count})")
                    total_samples += count
                
                print(f"  • {unified_cat}: {' + '.join(sample_counts)} = {total_samples:,} total")

class DataProcessor:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.unifier = CategoryUnifier()
    
    def load_all_data(self) -> pd.DataFrame:
        """Load all CSV files and combine them"""
        csv_files = glob.glob(str(self.data_dir / "*.csv"))
        
        if not csv_files:
            txt_files = glob.glob(str(self.data_dir / "*.txt"))
            if txt_files:
                csv_files = txt_files
        
        dataframes = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                print(f"Loaded {len(df)} rows from {file}")
                dataframes.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        if not dataframes:
            raise ValueError("No data files found in the dataset directory")
        
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"Total combined data: {len(combined_df)} rows")
        return combined_df
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        if pd.isna(text) or text == "":
            return ""
        
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def prepare_features_and_labels(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Prepare features (text) and labels (categories)"""
        df['product_name'] = df['product_name'].fillna('')
        df['product_brand'] = df['product_brand'].fillna('')
        df['category'] = df['category'].fillna('unknown')
        
        print("Applying multilingual category unification...")
        original_categories = df['category'].tolist()
        print(f"Before unification: {df['category'].nunique()} unique categories")
        
        unified_categories = self.unifier.unify_categories(original_categories)
        
        print(f"After unification: {len(set(unified_categories))} unique categories")
        
        self.unifier.print_unification_report(original_categories)

        features = []
        for _, row in df.iterrows():
            text_features = []
            
            if row['product_name']:
                text_features.append(self.clean_text(row['product_name']))
            
            if row['product_brand']:
                text_features.append(self.clean_text(row['product_brand']))
            
            combined_text = ' '.join(text_features)
            features.append(combined_text if combined_text.strip() else 'unknown product')
        
        labels = unified_categories

        print("Sample labels (post-unification):", labels[:20])
        print("Unique categories (post-unification):", len(set(labels)))
        
        return features, labels
    
    def get_data_stats(self, df: pd.DataFrame) -> dict:
        """Get basic statistics about the data"""
        return {
            'total_rows': len(df),
            'unique_categories': df['category'].nunique(),
            'unique_locales': df['locale'].nunique(),
            'categories_per_locale': df.groupby('locale')['category'].nunique().to_dict(),
            'top_categories': df['category'].value_counts().head(10).to_dict()
        }