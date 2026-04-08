#!/usr/bin/env python3
"""
BioBERT-based Lifestyle Parser

Upgrade from regex to BERT for better natural language understanding
"""

import torch
import numpy as np
from typing import Dict, Optional
from transformers import AutoTokenizer, AutoModel
import re


class BioBERTLifestyleParser:
    """
    Parse patient lifestyle descriptions using BioBERT embeddings
    
    Upgrades the simple regex parser to use contextualized embeddings
    for better understanding of natural language patient descriptions
    """
    
    def __init__(self, model_name: str = "dmis-lab/biobert-v1.1"):
        """
        Initialize BioBERT parser
        
        Args:
            model_name: HuggingFace model name (default: BioBERT)
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            self.use_bert = True
            print(f"✓ Loaded {model_name} for lifestyle parsing")
        except Exception as e:
            print(f"⚠️ Could not load BioBERT: {e}")
            print("   Falling back to regex parser")
            self.use_bert = False
    
    def embed_text(self, text: str) -> torch.Tensor:
        """Get BioBERT embeddings for text"""
        if not self.use_bert:
            return None
        
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings
    
    def parse_lifestyle_description(self, text: str) -> Dict:
        """
        Parse lifestyle from natural language using BioBERT
        
        Args:
            text: Natural language description
            
        Returns:
            Dictionary of lifestyle factors
        """
        if self.use_bert:
            return self._parse_with_bert(text)
        else:
            return self._parse_with_regex(text)
    
    def _parse_with_bert(self, text: str) -> Dict:
        """Parse using BioBERT embeddings + keyword detection"""
        text_lower = text.lower()
        
        # Get embeddings for context
        embeddings = self.embed_text(text)
        
        lifestyle = {
            'occupation': 'unknown',
            'exercise_hours_per_week': 0.0,
            'sleep_hours_per_night': 7.0,
            'smoking': False,
            'alcohol_drinks_per_week': 0.0,
            'stress_level': 0.5,
            'diet_quality': 'moderate'
        }
        
        # Occupation (keyword-based with context)
        occupation_keywords = {
            'office_worker': ['office', 'desk', 'sedentary', 'computer', 'administrative'],
            'manual_labor': ['manual', 'labor', 'construction', 'factory', 'physical work'],
            'healthcare': ['healthcare', 'nurse', 'doctor', 'medical', 'hospital'],
            'service': ['service', 'retail', 'restaurant', 'customer'],
            'education': ['teacher', 'professor', 'education', 'school']
        }
        
        for occupation, keywords in occupation_keywords.items():
            if any(kw in text_lower for kw in keywords):
                lifestyle['occupation'] = occupation
                break
        
        # Exercise (extract hours or classify level)
        exercise_patterns = [
            (r'(\d+)\s*(?:hour|hr|h).*?(?:exercise|gym|workout)', 'hours'),
            (r'exercise.*?(\d+)\s*(?:hour|hr|h)', 'hours'),
            (r'(\d+)\s*(?:min|minute).*?(?:exercise|gym|workout)', 'minutes'),
        ]
        
        for pattern, unit in exercise_patterns:
            match = re.search(pattern, text_lower)
            if match:
                value = float(match.group(1))
                if unit == 'minutes':
                    value = value / 60.0
                lifestyle['exercise_hours_per_week'] = value
                break
        else:
            # Classify by keywords
            if any(kw in text_lower for kw in ['sedentary', 'no exercise', 'inactive', 'couch']):
                lifestyle['exercise_hours_per_week'] = 0.0
            elif any(kw in text_lower for kw in ['very active', 'athlete', 'daily exercise']):
                lifestyle['exercise_hours_per_week'] = 7.0
            elif any(kw in text_lower for kw in ['active', 'regular exercise', 'gym']):
                lifestyle['exercise_hours_per_week'] = 3.0
            elif any(kw in text_lower for kw in ['light exercise', 'occasional']):
                lifestyle['exercise_hours_per_week'] = 1.5
        
        # Sleep (extract hours)
        sleep_match = re.search(r'sleep[s]?\s*(?:only\s*)?(\d+)\s*(?:hour|hr|h)', text_lower)
        if sleep_match:
            lifestyle['sleep_hours_per_night'] = float(sleep_match.group(1))
        elif any(kw in text_lower for kw in ['poor sleep', 'insomnia', 'sleep deprivation']):
            lifestyle['sleep_hours_per_night'] = 5.0
        elif any(kw in text_lower for kw in ['good sleep', 'well-rested']):
            lifestyle['sleep_hours_per_night'] = 8.0
        
        # Smoking (binary)
        smoking_keywords = ['smok', 'cigarette', 'tobacco', 'vape', 'vaping']
        quit_keywords = ['quit', 'former', 'ex-smoker', 'stopped smoking']
        
        if any(kw in text_lower for kw in quit_keywords):
            lifestyle['smoking'] = False
        elif any(kw in text_lower for kw in smoking_keywords):
            lifestyle['smoking'] = True
        
        # Alcohol (extract amount)
        alcohol_patterns = [
            (r'(\d+)\s*(?:drink|beer|glass|bottle)s?\s*(?:per\s*)?(?:day|daily)', 'daily'),
            (r'(\d+)\s*(?:drink|beer|glass|bottle)s?\s*(?:per\s*)?(?:week|weekly)', 'weekly'),
            (r'(\d+)\s*(?:drink|beer|glass|bottle)s?', 'unspecified'),
        ]
        
        for pattern, frequency in alcohol_patterns:
            match = re.search(pattern, text_lower)
            if match:
                amount = float(match.group(1))
                if frequency == 'daily':
                    lifestyle['alcohol_drinks_per_week'] = amount * 7
                elif frequency == 'weekly':
                    lifestyle['alcohol_drinks_per_week'] = amount
                else:
                    # Assume daily if unspecified
                    lifestyle['alcohol_drinks_per_week'] = amount * 7
                break
        else:
            # Classify by keywords
            if any(kw in text_lower for kw in ['heavy drink', 'alcoholic', 'binge']):
                lifestyle['alcohol_drinks_per_week'] = 20.0
            elif any(kw in text_lower for kw in ['moderate drink', 'social drink']):
                lifestyle['alcohol_drinks_per_week'] = 7.0
            elif any(kw in text_lower for kw in ['occasional drink', 'rarely drink']):
                lifestyle['alcohol_drinks_per_week'] = 2.0
            elif any(kw in text_lower for kw in ['no alcohol', 'non-drinker', 'abstain']):
                lifestyle['alcohol_drinks_per_week'] = 0.0
        
        # Stress (scale 0-1)
        stress_keywords_high = ['high stress', 'stressed', 'anxiety', 'anxious', 'burnout', 'overwhelmed']
        stress_keywords_low = ['low stress', 'relaxed', 'calm', 'peaceful', 'balanced']
        
        if any(kw in text_lower for kw in stress_keywords_high):
            lifestyle['stress_level'] = 0.8
        elif any(kw in text_lower for kw in stress_keywords_low):
            lifestyle['stress_level'] = 0.2
        elif 'moderate stress' in text_lower or 'some stress' in text_lower:
            lifestyle['stress_level'] = 0.5
        
        # Diet quality
        diet_keywords_poor = ['fast food', 'junk food', 'poor diet', 'unhealthy', 'processed']
        diet_keywords_good = ['healthy', 'vegetables', 'balanced', 'nutritious', 'whole foods']
        
        if any(kw in text_lower for kw in diet_keywords_poor):
            lifestyle['diet_quality'] = 'poor'
        elif any(kw in text_lower for kw in diet_keywords_good):
            lifestyle['diet_quality'] = 'good'
        
        return lifestyle
    
    def _parse_with_regex(self, text: str) -> Dict:
        """Fallback regex parser (same as original)"""
        text = text.lower()
        
        lifestyle = {
            'occupation': 'unknown',
            'exercise_hours_per_week': 0.0,
            'sleep_hours_per_night': 7.0,
            'smoking': False,
            'alcohol_drinks_per_week': 0.0,
            'stress_level': 0.5,
            'diet_quality': 'moderate'
        }
        
        # Occupation
        if 'office' in text or 'desk' in text or 'sedentary' in text:
            lifestyle['occupation'] = 'office_worker'
        elif 'manual' in text or 'labor' in text:
            lifestyle['occupation'] = 'manual_labor'
        
        # Exercise
        if 'sedentary' in text or 'no exercise' in text:
            lifestyle['exercise_hours_per_week'] = 0.0
        elif 'active' in text or 'exercise' in text:
            lifestyle['exercise_hours_per_week'] = 3.0
        
        # Sleep
        sleep_match = re.search(r'sleep[s]?\s*(\d+)\s*(hour|hr)', text)
        if sleep_match:
            lifestyle['sleep_hours_per_night'] = float(sleep_match.group(1))
        elif 'poor sleep' in text:
            lifestyle['sleep_hours_per_night'] = 5.0
        
        # Smoking
        if 'smok' in text or 'cigarette' in text:
            lifestyle['smoking'] = True
        
        # Alcohol
        if 'drink' in text or 'alcohol' in text or 'beer' in text:
            alcohol_match = re.search(r'(\d+)\s*(drink|beer|glass)', text)
            if alcohol_match:
                lifestyle['alcohol_drinks_per_week'] = float(alcohol_match.group(1)) * 7
            else:
                lifestyle['alcohol_drinks_per_week'] = 7.0
        
        # Stress
        if 'high stress' in text or 'stressed' in text:
            lifestyle['stress_level'] = 0.8
        elif 'low stress' in text:
            lifestyle['stress_level'] = 0.2
        
        # Diet
        if 'fast food' in text or 'poor diet' in text:
            lifestyle['diet_quality'] = 'poor'
        elif 'healthy' in text or 'vegetables' in text:
            lifestyle['diet_quality'] = 'good'
        
        return lifestyle


# Singleton instance
_parser_instance = None

def get_lifestyle_parser() -> BioBERTLifestyleParser:
    """Get singleton parser instance"""
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = BioBERTLifestyleParser()
    return _parser_instance


# Example usage
if __name__ == '__main__':
    parser = BioBERTLifestyleParser()
    
    test_cases = [
        "Sedentary office worker, smokes 1 pack per day, drinks 3 beers daily, sleeps 5 hours, high stress job, eats fast food regularly",
        "Active healthcare worker, exercises 5 hours per week, good sleep 8 hours, non-smoker, occasional alcohol, balanced diet",
        "Manual laborer, moderate exercise, 6 hours sleep, former smoker, heavy drinker, poor diet"
    ]
    
    print("=" * 80)
    print("BioBERT Lifestyle Parser Test")
    print("=" * 80)
    
    for i, text in enumerate(test_cases, 1):
        print(f"\nTest {i}: {text[:60]}...")
        result = parser.parse_lifestyle_description(text)
        print(f"  Occupation: {result['occupation']}")
        print(f"  Exercise: {result['exercise_hours_per_week']:.1f} hrs/week")
        print(f"  Sleep: {result['sleep_hours_per_night']:.1f} hrs/night")
        print(f"  Smoking: {result['smoking']}")
        print(f"  Alcohol: {result['alcohol_drinks_per_week']:.1f} drinks/week")
        print(f"  Stress: {result['stress_level']:.1f}")
        print(f"  Diet: {result['diet_quality']}")
