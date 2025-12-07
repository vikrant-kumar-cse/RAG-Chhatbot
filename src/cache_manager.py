import json
import os
import re
import socket

# Try to import fuzzy matching library
try:
    from rapidfuzz import fuzz, process
    FUZZY_LIB = "rapidfuzz"
except ImportError:
    try:
        from fuzzywuzzy import fuzz
        from fuzzywuzzy import process
        FUZZY_LIB = "fuzzywuzzy"
    except ImportError:
        print("⚠️  WARNING: No fuzzy matching library found!")
        print("   Install one: pip install rapidfuzz  OR  pip install fuzzywuzzy")
        FUZZY_LIB = None

class CacheManager:
    def __init__(self, cache_file_path='data/medical_cache.json'):
        self.cache_file_path = cache_file_path
        self.cache_data = []
        self.fuzzy_available = FUZZY_LIB is not None
        
        if not self.fuzzy_available:
            print("⚠️  Fuzzy matching disabled - install rapidfuzz or fuzzywuzzy")
        else:
            print(f"✅ Using {FUZZY_LIB} for fuzzy matching")
        
        self.load_cache()
    
    def load_cache(self):
        """Load cache from JSON file"""
        try:
            if os.path.exists(self.cache_file_path):
                with open(self.cache_file_path, 'r', encoding='utf-8') as f:
                    self.cache_data = json.load(f)
                print(f"✅ Cache loaded: {len(self.cache_data)} questions")
            else:
                print(f"⚠️  Cache file not found: {self.cache_file_path}")
                self.cache_data = []
        except Exception as e:
            print(f"❌ Error loading cache: {str(e)}")
            self.cache_data = []
    
    def preprocess_text(self, text):
        """Clean and normalize text for matching"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s]', '', text)
        
        return text
    
    def find_match(self, user_question, threshold=85):
        """
        Find matching cached question using fuzzy matching
        
        Args:
            user_question (str): User's input question
            threshold (int): Minimum similarity score (0-100)
        
        Returns:
            dict: {
                'matched': bool,
                'answer': str or None,
                'confidence': float,
                'question': str or None
            }
        """
        if not user_question or not self.cache_data:
            return {
                'matched': False,
                'answer': None,
                'confidence': 0,
                'question': None
            }
        
        # If fuzzy matching not available, try exact match only
        if not self.fuzzy_available:
            return self._exact_match(user_question)
        
        # Preprocess user question
        processed_question = self.preprocess_text(user_question)
        
        # Prepare list of cached questions for matching
        cached_questions = [item['question'] for item in self.cache_data]
        
        # Find best match using fuzzy matching
        if FUZZY_LIB == "rapidfuzz":
            # RapidFuzz syntax
            best_match = process.extractOne(
                processed_question,
                [self.preprocess_text(q) for q in cached_questions],
                scorer=fuzz.token_sort_ratio
            )
            score = best_match[1] if best_match else 0
        else:
            # FuzzyWuzzy syntax
            best_match = process.extractOne(
                processed_question,
                [self.preprocess_text(q) for q in cached_questions],
                scorer=fuzz.token_sort_ratio
            )
            score = best_match[1] if best_match else 0
        
        if best_match and score >= threshold:
            # Get the original cache item
            match_index = [self.preprocess_text(q) for q in cached_questions].index(best_match[0])
            matched_item = self.cache_data[match_index]
            
            print(f"🎯 Cache HIT! Confidence: {score}%")
            print(f"   Matched: {matched_item['question'][:50]}...")
            
            return {
                'matched': True,
                'answer': matched_item['answer'],
                'confidence': score / 100,
                'question': matched_item['question'],
                'category': matched_item.get('category', 'general')
            }
        else:
            print(f"❌ Cache MISS. Best score: {score}%")
            return {
                'matched': False,
                'answer': None,
                'confidence': score / 100,
                'question': None
            }
    
    def _exact_match(self, user_question):
        """Fallback exact match when fuzzy matching unavailable"""
        processed_question = self.preprocess_text(user_question)
        
        for item in self.cache_data:
            if self.preprocess_text(item['question']) == processed_question:
                print(f"🎯 Cache HIT (Exact Match)!")
                return {
                    'matched': True,
                    'answer': item['answer'],
                    'confidence': 1.0,
                    'question': item['question'],
                    'category': item.get('category', 'general')
                }
        
        print(f"❌ Cache MISS (No exact match)")
        return {
            'matched': False,
            'answer': None,
            'confidence': 0,
            'question': None
        }
    
    def add_to_cache(self, question, answer, keywords=None, category='general'):
        """Add new entry to cache (for future expansion)"""
        try:
            new_id = max([item['id'] for item in self.cache_data], default=0) + 1
            
            new_entry = {
                'id': new_id,
                'question': question,
                'answer': answer,
                'keywords': keywords or [],
                'category': category
            }
            
            self.cache_data.append(new_entry)
            
            # Save to file
            with open(self.cache_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Added to cache: {question[:50]}...")
            return True
            
        except Exception as e:
            print(f"❌ Error adding to cache: {str(e)}")
            return False
    
    def get_cache_stats(self):
        """Get cache statistics"""
        categories = {}
        for item in self.cache_data:
            cat = item.get('category', 'general')
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            'total_questions': len(self.cache_data),
            'categories': categories
        }
    
    @staticmethod
    def check_internet_connection(timeout=3):
        """
        Check if internet connection is available
        
        Returns:
            bool: True if connected, False otherwise
        """
        try:
            # Try to connect to Google DNS
            socket.create_connection(("8.8.8.8", 53), timeout=timeout)
            return True
        except OSError:
            pass
        
        try:
            # Fallback: Try to connect to Cloudflare DNS
            socket.create_connection(("1.1.1.1", 53), timeout=timeout)
            return True
        except OSError:
            return False
    
    @staticmethod
    def check_openai_availability():
        """
        Check if OpenAI API is available
        
        Returns:
            bool: True if available, False otherwise
        """
        try:
            import requests
            response = requests.get("https://api.openai.com/v1/models", timeout=5)
            return response.status_code in [200, 401]  # 401 means API is up but needs auth
        except Exception:
            return False


# Initialize global cache manager
cache_manager = CacheManager()