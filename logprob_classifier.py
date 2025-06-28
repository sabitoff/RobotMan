import math
import os
import yaml
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from functools import lru_cache
from openai import OpenAI
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# 1. CONFIGURATION AND CONSTANTS
# =============================================================================

class IntentConfig:
    """Centralized configuration for intent classification."""
    
    # Display constants
    PROBABILITY_BAR_LENGTH = 20
    QUERY_TRUNCATE_LENGTH = 50
    
    # Validation constants
    MAX_QUERY_LENGTH = 1000
    MIN_QUERY_LENGTH = 1
    
    # All 10 intent categories in order
    INTENT_CATEGORIES = [
        "concept_explanation",
        "fact_lookup_qa", 
        "document_navigation",
        "summarization_digest",
        "comparison_decision_support",
        "instruction_procedure",
        "rewrite_transform",
        "creative_generation",
        "data_extraction_structuring",
        "meta_clarification"
    ]
    
    # Confidence thresholds
    CONFIDENCE_THRESHOLDS = {
        "high": 0.7,
        "medium": 0.4,
        "low": 0.0
    }
    
    # Processing path mapping
    RAG_INTENTS = {
        "fact_lookup_qa",
        "document_navigation", 
        "summarization_digest",
        "comparison_decision_support",
        "data_extraction_structuring"
    }
    
    # Display names for intents
    INTENT_DISPLAY_NAMES = {
        "concept_explanation": "Concept/Explanation",
        "fact_lookup_qa": "Fact Lookup/Q&A",
        "document_navigation": "Document Navigation", 
        "summarization_digest": "Summarization/Digest",
        "comparison_decision_support": "Comparison/Decision Support",
        "instruction_procedure": "Instruction/Procedure",
        "rewrite_transform": "Rewrite/Transform",
        "creative_generation": "Creative Generation",
        "data_extraction_structuring": "Data Extraction/Structuring",
        "meta_clarification": "Meta/Clarification"
    }


class TokenMapper:
    """Enhanced token mapper with distinctive tokens and confidence scoring."""
    
    def __init__(self):
        # Enhanced token mappings with longer, more distinctive tokens
        # Priority: longer tokens > shorter tokens for better disambiguation
        self.token_to_intent = {
            # High-confidence distinctive tokens (primary)
            "CONCEPTUAL": "concept_explanation",
            "FACTUAL": "fact_lookup_qa", 
            "NAVIGATE": "document_navigation",
            "SUMMARIZE": "summarization_digest",
            "COMPARISON": "comparison_decision_support",
            "INSTRUCTION": "instruction_procedure",
            "REWRITE": "rewrite_transform", 
            "CREATIVE": "creative_generation",
            "EXTRACTION": "data_extraction_structuring",
            "CLARIFICATION": "meta_clarification",
            
            # Medium-confidence tokens (fallback)
            "CONCEPT": "concept_explanation",
            "FACT": "fact_lookup_qa",
            "NAV": "document_navigation", 
            "SUM": "summarization_digest",
            "COMPARE": "comparison_decision_support",
            "INSTRUCT": "instruction_procedure",
            "REWRITING": "rewrite_transform",
            "DRAFT": "creative_generation",
            "EXTRACT": "data_extraction_structuring",
            "CLARIFY": "meta_clarification",
            
            # Low-confidence tokens (backup)
            "CON": "concept_explanation",
            "FAC": "fact_lookup_qa",
            "DOC": "document_navigation",
            "SUMM": "summarization_digest", 
            "COMP": "comparison_decision_support",
            "PROC": "instruction_procedure",
            "REW": "rewrite_transform",
            "GEN": "creative_generation", 
            "DATA": "data_extraction_structuring",
            "META": "meta_clarification",
            
            # Legacy tokens for backward compatibility
            "IN": "instruction_procedure",
            "RE": "rewrite_transform", 
            "D": "creative_generation",
            "EX": "data_extraction_structuring",
            "CL": "meta_clarification"
        }
        
        # Token confidence scores based on distinctiveness
        self.token_confidence = {
            # High confidence: long, distinctive tokens
            "CONCEPTUAL": 0.95, "FACTUAL": 0.95, "NAVIGATE": 0.95,
            "SUMMARIZE": 0.95, "COMPARISON": 0.95, "INSTRUCTION": 0.95,
            "REWRITE": 0.95, "CREATIVE": 0.95, "EXTRACTION": 0.95,
            "CLARIFICATION": 0.95,
            
            # Medium confidence: standard tokens
            "CONCEPT": 0.85, "FACT": 0.85, "NAV": 0.80,
            "SUM": 0.80, "COMPARE": 0.85, "INSTRUCT": 0.85,
            "REWRITING": 0.85, "DRAFT": 0.85, "EXTRACT": 0.85,
            "CLARIFY": 0.85,
            
            # Lower confidence: shorter or ambiguous tokens
            "CON": 0.70, "FAC": 0.70, "DOC": 0.75,
            "SUMM": 0.75, "COMP": 0.70, "PROC": 0.75,
            "REW": 0.65, "GEN": 0.60, "DATA": 0.70,
            "META": 0.70,
            
            # Legacy tokens (lowest confidence)
            "IN": 0.50, "RE": 0.45, "D": 0.40,
            "EX": 0.55, "CL": 0.50
        }
    
    def map_token_to_intent(self, token: str) -> Optional[str]:
        """Map a token to an intent category."""
        return self.token_to_intent.get(token.strip().upper())
    
    def get_token_confidence(self, token: str) -> float:
        """Get confidence score for a token."""
        return self.token_confidence.get(token.strip().upper(), 0.3)  # Default low confidence
    
    def map_token_with_confidence(self, token: str) -> Tuple[Optional[str], float]:
        """Map token to intent and return confidence score."""
        clean_token = token.strip().upper()
        intent = self.token_to_intent.get(clean_token)
        confidence = self.token_confidence.get(clean_token, 0.3)
        return intent, confidence
    
    def map_token_sequence(self, tokens: List[str]) -> Dict[str, float]:
        """
        Map a sequence of tokens to intents with weighted confidence.
        
        Args:
            tokens: List of token strings
            
        Returns:
            Dictionary mapping intents to weighted confidence scores
        """
        intent_scores = {}
        total_weight = 0.0
        
        for i, token in enumerate(tokens):
            intent, confidence = self.map_token_with_confidence(token)
            if intent:
                # Weight tokens by position (first token gets highest weight)
                position_weight = 1.0 / (i + 1)  # 1.0, 0.5, 0.33, etc.
                weighted_score = confidence * position_weight
                
                if intent in intent_scores:
                    # Combine scores (take maximum for now)
                    intent_scores[intent] = max(intent_scores[intent], weighted_score)
                else:
                    intent_scores[intent] = weighted_score
                
                total_weight += position_weight
        
        # Normalize scores by total weight
        if total_weight > 0:
            intent_scores = {intent: score/total_weight for intent, score in intent_scores.items()}
        
        return intent_scores
    
    def validate_token_mappings(self) -> Dict[str, List[str]]:
        """
        Validate token mappings for completeness and consistency.
        
        Returns:
            Dictionary with validation results and any issues found
        """
        issues = {
            "missing_intents": [],
            "invalid_mappings": [],
            "duplicate_tokens": [],
            "coverage_report": []
        }
        
        # Check if all intents have at least one token mapping
        mapped_intents = set(self.token_to_intent.values())
        all_intents = set(IntentConfig.INTENT_CATEGORIES)
        
        missing_intents = all_intents - mapped_intents
        if missing_intents:
            issues["missing_intents"] = list(missing_intents)
            logger.warning("Some intents have no token mappings", missing_intents=missing_intents)
        
        # Check for invalid intent mappings
        for token, intent in self.token_to_intent.items():
            if intent not in all_intents:
                issues["invalid_mappings"].append(f"Token '{token}' maps to invalid intent '{intent}'")
                logger.error("Invalid token mapping", token=token, intent=intent)
        
        # Check for duplicate token mappings (same token mapping to different intents)
        token_counts = {}
        for token, intent in self.token_to_intent.items():
            if token in token_counts:
                issues["duplicate_tokens"].append(f"Token '{token}' maps to multiple intents")
            else:
                token_counts[token] = intent
        
        # Generate coverage report
        for intent in IntentConfig.INTENT_CATEGORIES:
            tokens = [token for token, mapped_intent in self.token_to_intent.items() if mapped_intent == intent]
            issues["coverage_report"].append(f"{intent}: {len(tokens)} tokens ({', '.join(tokens)})")
        
        return issues



# =============================================================================
# 2. CORE CLASSIFICATION LOGIC
# =============================================================================

@dataclass
class ClassificationResult:
    """Structured result from intent classification."""
    query: str
    predicted_intent: str
    confidence: float
    distribution: Dict[str, float]
    processing_path: str  # "RAG" or "LLM-only"
    uncertainty: float
    confidence_level: str  # "high", "medium", "low"
    
    # Enhanced multi-token analysis fields
    model_tokens: List[str] = None  # Raw tokens from model response
    token_analysis: List[Dict] = None  # Detailed token analysis
    sequence_confidence: float = 0.0  # Confidence from token sequence analysis
    
    def get_display_intent(self) -> str:
        """Get formatted display name for the predicted intent."""
        return IntentConfig.INTENT_DISPLAY_NAMES.get(
            self.predicted_intent, 
            self.predicted_intent.replace("_", " ").title()
        )
    
    def get_token_summary(self) -> str:
        """Get a summary of the token analysis."""
        if not self.model_tokens:
            return "Single token analysis"
        
        token_str = " → ".join([t.strip().upper() for t in self.model_tokens])
        return f"Multi-token: {token_str} (sequence confidence: {self.sequence_confidence:.3f})"


class LogprobClassifier:
    """
    Intent classifier using OpenAI's logprobs for calibrated confidence.
    
    This is the core classification engine that makes a single API call
    and returns complete results.
    """
    
    def __init__(self, api_key: str):
        """Initialize the classifier."""
        self.client = OpenAI(api_key=api_key)
        self.token_mapper = TokenMapper()
        self.config = IntentConfig()
        self._cache = {}  # Simple cache for repeated queries
        
        # Validate token mappings on initialization
        validation_results = self.token_mapper.validate_token_mappings()
        
        if validation_results["missing_intents"]:
            logger.warning("Token mapping validation found issues", 
                         missing_intents=validation_results["missing_intents"])
        
        if validation_results["invalid_mappings"]:
            logger.error("Invalid token mappings found", 
                        invalid_mappings=validation_results["invalid_mappings"])
        
        logger.info("Logprob intent classifier initialized", 
                   validation_summary=f"Coverage: {len(validation_results['coverage_report'])} intents")
    
    def classify_with_distribution(self, query: str) -> ClassificationResult:
        """
        Single method that performs classification and returns complete results.
        
        Args:
            query: The user query to classify
            
        Returns:
            ClassificationResult with all information
            
        Raises:
            ValueError: If query is invalid (empty, too long, etc.)
        """
        # Input validation
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        
        query = query.strip()
        if len(query) < IntentConfig.MIN_QUERY_LENGTH:
            raise ValueError(f"Query too short (minimum {IntentConfig.MIN_QUERY_LENGTH} characters)")
        
        if len(query) > IntentConfig.MAX_QUERY_LENGTH:
            raise ValueError(f"Query too long (maximum {IntentConfig.MAX_QUERY_LENGTH} characters)")
        
        # Check cache first
        cache_key = query.lower().strip()
        if cache_key in self._cache:
            logger.debug("Cache hit for query", query=query[:IntentConfig.QUERY_TRUNCATE_LENGTH])
            return self._cache[cache_key]
        
        try:
            # Make single API call
            response = self._make_api_call(query)
            
            # Create complete probability distribution first
            distribution = self._create_complete_distribution(response)
            
            # Find the intent with highest probability (this is the true prediction)
            predicted_intent = max(distribution.items(), key=lambda x: x[1])[0]
            confidence = distribution[predicted_intent]
            
            # Log the actual model tokens and analysis for debugging
            if response.choices[0].logprobs and response.choices[0].logprobs.content:
                model_response = response.choices[0].message.content.strip()
                model_tokens = [pos.token for pos in response.choices[0].logprobs.content if pos.top_logprobs]
                
                # Calculate token confidence scores
                token_analysis = []
                for token in model_tokens:
                    intent, conf = self.token_mapper.map_token_with_confidence(token)
                    token_analysis.append({
                        "token": token.strip().upper(),
                        "intent": intent,
                        "confidence": conf
                    })
                
                logger.info("Enhanced intent classification", 
                           model_response=model_response,
                           token_analysis=token_analysis,
                           predicted_intent=predicted_intent,
                           confidence=confidence,
                           num_tokens=len(model_tokens))
            
            # Calculate additional metrics
            uncertainty = self._calculate_uncertainty(distribution)
            confidence_level = self._get_confidence_level(confidence)
            processing_path = self._get_processing_path(predicted_intent)
            
            # Extract token information for enhanced analysis
            model_tokens = []
            token_analysis = []
            sequence_confidence = 0.0
            
            if response.choices[0].logprobs and response.choices[0].logprobs.content:
                model_tokens = [pos.token for pos in response.choices[0].logprobs.content if pos.top_logprobs]
                
                for token in model_tokens:
                    intent, conf = self.token_mapper.map_token_with_confidence(token)
                    token_analysis.append({
                        "token": token.strip().upper(),
                        "intent": intent,
                        "confidence": conf
                    })
                
                # Calculate sequence confidence if multiple tokens
                if len(model_tokens) > 1:
                    sequence_scores = self.token_mapper.map_token_sequence([t.strip() for t in model_tokens])
                    if predicted_intent in sequence_scores:
                        sequence_confidence = sequence_scores[predicted_intent]
            
            result = ClassificationResult(
                query=query,
                predicted_intent=predicted_intent,
                confidence=confidence,
                distribution=distribution,
                processing_path=processing_path,
                uncertainty=uncertainty,
                confidence_level=confidence_level,
                model_tokens=model_tokens,
                token_analysis=token_analysis,
                sequence_confidence=sequence_confidence
            )
            
            # Cache the result
            self._cache[cache_key] = result
            logger.debug("Cached result for query", query=query[:IntentConfig.QUERY_TRUNCATE_LENGTH])
            
            return result
            
        except Exception as e:
            logger.error("Error in classification", error=str(e), query=query[:IntentConfig.QUERY_TRUNCATE_LENGTH])
            return self._create_fallback_result(query)
    
    def _make_api_call(self, query: str):
        """Make the OpenAI API call with logprobs for multi-token responses."""
        prompt = f"""Classify this user query into one of these intent categories. Respond with 1-2 words using these distinctive tokens:

Primary tokens (use these first): CONCEPTUAL, FACTUAL, NAVIGATE, SUMMARIZE, COMPARISON, INSTRUCTION, REWRITE, CREATIVE, EXTRACTION, CLARIFICATION

Secondary tokens (if primary unclear): CONCEPT, FACT, NAV, SUM, COMPARE, INSTRUCT, DRAFT, EXTRACT, CLARIFY

Examples:
- "What is machine learning?" → CONCEPTUAL
- "When was Python created?" → FACTUAL  
- "Find the sales report" → NAVIGATE
- "Summarize this document" → SUMMARIZE
- "Compare these options" → COMPARISON
- "How to install software?" → INSTRUCTION
- "Rewrite this email" → REWRITE
- "Write a poem" → CREATIVE
- "Extract the data" → EXTRACTION
- "What do you mean?" → CLARIFICATION

Query: "{query}"

Classification:"""
        
        return self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=3,  # Allow up to 3 tokens for more nuanced responses
            logprobs=True,
            top_logprobs=15  # Increased to capture more token alternatives
        )
    

    
    def _create_complete_distribution(self, response) -> Dict[str, float]:
        """Create a complete probability distribution over all intents using multi-token analysis."""
        # Initialize all intents with tiny probabilities
        distribution = {intent: 1e-10 for intent in self.config.INTENT_CATEGORIES}
        
        if response.choices[0].logprobs and response.choices[0].logprobs.content:
            # Extract all tokens from the response
            response_tokens = []
            total_token_probability = 0.0
            
            # Process each token position in the response
            for token_position in response.choices[0].logprobs.content:
                position_probs = {}
                
                # Get probabilities for all top tokens at this position
                for token_info in token_position.top_logprobs:
                    token = token_info.token.strip().upper()
                    probability = math.exp(token_info.logprob)
                    
                    # Map token to intent with confidence
                    intent, token_confidence = self.token_mapper.map_token_with_confidence(token)
                    if intent:
                        # Apply token confidence weighting
                        weighted_prob = probability * token_confidence
                        
                        if intent in position_probs:
                            position_probs[intent] += weighted_prob
                        else:
                            position_probs[intent] = weighted_prob
                
                # Add position probabilities to overall distribution
                # Weight by position: first token = 1.0, second = 0.7, third = 0.4
                position_weight = max(0.4, 1.0 - (len(response_tokens) * 0.3))
                
                for intent, prob in position_probs.items():
                    if intent in distribution:
                        distribution[intent] += prob * position_weight
                
                response_tokens.append(position_probs)
            
            # If we have multiple tokens, also try sequence analysis
            if len(response_tokens) > 1:
                # Extract the actual token sequence from response
                actual_tokens = []
                for token_position in response.choices[0].logprobs.content:
                    if token_position.top_logprobs:
                        # Take the most likely token at this position
                        best_token = max(token_position.top_logprobs, key=lambda x: x.logprob)
                        actual_tokens.append(best_token.token.strip())
                
                # Analyze token sequence
                sequence_scores = self.token_mapper.map_token_sequence(actual_tokens)
                
                # Blend sequence scores with individual token scores
                for intent, score in sequence_scores.items():
                    if intent in distribution:
                        # Give sequence analysis 30% weight
                        distribution[intent] = distribution[intent] * 0.7 + score * 0.3
        
        # Normalize to ensure probabilities sum to 1.0
        total_prob = sum(distribution.values())
        if total_prob > 0:
            distribution = {intent: prob/total_prob for intent, prob in distribution.items()}
        
        return distribution
    
    def _calculate_uncertainty(self, distribution: Dict[str, float]) -> float:
        """Calculate entropy-based uncertainty measure."""
        # Calculate entropy
        entropy = -sum(p * math.log2(p) for p in distribution.values() if p > 1e-15)
        
        # Normalize by maximum possible entropy
        max_entropy = math.log2(len(distribution))
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Get confidence level category."""
        if confidence >= self.config.CONFIDENCE_THRESHOLDS["high"]:
            return "high"
        elif confidence >= self.config.CONFIDENCE_THRESHOLDS["medium"]:
            return "medium"
        else:
            return "low"
    
    def _get_processing_path(self, intent: str) -> str:
        """Determine processing path (RAG vs LLM-only)."""
        return "RAG" if intent in self.config.RAG_INTENTS else "LLM-only"
    
    def _create_fallback_result(self, query: str) -> ClassificationResult:
        """Create a safe fallback result when classification fails."""
        uniform_prob = 1.0 / len(self.config.INTENT_CATEGORIES)
        distribution = {intent: uniform_prob for intent in self.config.INTENT_CATEGORIES}
        
        return ClassificationResult(
            query=query,
            predicted_intent="concept_explanation",
            confidence=0.5,
            distribution=distribution,
            processing_path="LLM-only",
            uncertainty=1.0,
            confidence_level="low",
            model_tokens=[],
            token_analysis=[],
            sequence_confidence=0.0
        )


# =============================================================================
# 3. RESULT FORMATTING AND DISPLAY
# =============================================================================

class ResultFormatter:
    """Handles formatting and display of classification results."""
    
    @staticmethod
    def format_interactive_result(result: ClassificationResult) -> str:
        """Format result for interactive display."""
        output = []
        
        # Header
        output.append(f"\n📋 Results for: '{result.query}'")
        output.append("-" * 50)
        
        # Basic results
        output.append(f"🎯 Predicted Intent: {result.get_display_intent()}")
        output.append(f"📊 Confidence: {result.confidence:.3f} ({result.confidence:.1%})")
        
        # Enhanced token analysis
        if result.model_tokens and len(result.model_tokens) > 0:
            output.append(f"🔤 Token Analysis: {result.get_token_summary()}")
            
            if result.token_analysis and len(result.token_analysis) > 1:
                output.append("   Token Details:")
                for i, token_info in enumerate(result.token_analysis):
                    token = token_info['token']
                    intent = token_info['intent'] or 'Unknown'
                    conf = token_info['confidence']
                    output.append(f"   {i+1}. {token} → {intent} (confidence: {conf:.2f})")
        
        # Confidence level with emoji
        CONFIDENCE_EMOJI = {
            "high": "🟢",
            "medium": "🟡", 
            "low": "🔴"
        }
        emoji = CONFIDENCE_EMOJI.get(result.confidence_level, "⚪")
        output.append(f"📈 Confidence Level: {emoji} {result.confidence_level.title()}")
        
        # Complete probability distribution
        output.append("\n📊 Complete Probability Distribution:")
        
        # Sort by probability (highest first)
        sorted_intents = sorted(result.distribution.items(), key=lambda x: x[1], reverse=True)
        
        for intent_name, prob in sorted_intents:
            display_name = IntentConfig.INTENT_DISPLAY_NAMES.get(
                intent_name, intent_name.replace("_", " ").title()
            )
            
            # Create visual probability bar
            bar_length = int(prob * IntentConfig.PROBABILITY_BAR_LENGTH)
            bar = "█" * bar_length + "░" * (IntentConfig.PROBABILITY_BAR_LENGTH - bar_length)
            
            # Highlight the predicted intent
            if intent_name == result.predicted_intent:
                output.append(f"   🏆 {display_name:<30} {bar} {prob:.3f} ({prob:.1%}) ← SELECTED")
            else:
                output.append(f"     {display_name:<30} {bar} {prob:.3f} ({prob:.1%})")
        
        # Uncertainty measure
        output.append(f"\n📈 Model Uncertainty: {result.uncertainty:.3f} (0=certain, 1=completely uncertain)")
        if result.uncertainty < 0.2:
            output.append("   🟢 Very confident classification")
        elif result.uncertainty < 0.5:
            output.append("   🟡 Moderately confident classification") 
        else:
            output.append("   🔴 Uncertain classification - consider asking for clarification")
        
        # Processing path recommendation
        output.append(f"\n🛣️ Suggested Processing Path:")
        output.append(f"   → {result.processing_path}")
        
        # Enhancement info
        if result.model_tokens and len(result.model_tokens) > 1:
            output.append(f"\n🔬 Enhanced Analysis:")
            output.append(f"   • Multi-token response analyzed ({len(result.model_tokens)} tokens)")
            output.append(f"   • Sequence confidence: {result.sequence_confidence:.3f}")
            output.append(f"   • Token-level confidence weighting applied")
        
        return "\n".join(output)
    



# =============================================================================
# 4. CONFIGURATION UTILITIES
# =============================================================================

def load_config() -> Dict:
    """Load and validate configuration from YAML file."""
    config_path = Path("config/config.yaml")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required configuration keys
    if not config:
        raise ValueError("Configuration file is empty or invalid")
    
    if "openai" not in config:
        raise ValueError("Missing 'openai' section in configuration")
    
    if "api_key_env" not in config["openai"]:
        raise ValueError("Missing 'api_key_env' in openai configuration")
    
    # Validate IntentConfig consistency
    _validate_intent_config()
    
    logger.info("Configuration loaded and validated successfully")
    return config


def _validate_intent_config():
    """Validate IntentConfig for internal consistency."""
    # Check all intent categories have display names
    missing_display_names = []
    for intent in IntentConfig.INTENT_CATEGORIES:
        if intent not in IntentConfig.INTENT_DISPLAY_NAMES:
            missing_display_names.append(intent)
    
    if missing_display_names:
        raise ValueError(f"Missing display names for intents: {missing_display_names}")
    
    # Check confidence thresholds are valid
    for level, threshold in IntentConfig.CONFIDENCE_THRESHOLDS.items():
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Invalid confidence threshold for '{level}': {threshold} (must be 0.0-1.0)")
    
    # Check RAG_INTENTS is subset of INTENT_CATEGORIES
    invalid_rag_intents = IntentConfig.RAG_INTENTS - set(IntentConfig.INTENT_CATEGORIES)
    if invalid_rag_intents:
        raise ValueError(f"Invalid RAG intents not in INTENT_CATEGORIES: {invalid_rag_intents}")
    
    logger.debug("IntentConfig validation passed")


def get_api_key(config: Dict) -> str:
    """Get OpenAI API key from config or environment."""
    api_key_value = config["openai"]["api_key_env"]
    
    # Check if it's already an API key (starts with sk-)
    if api_key_value.startswith("sk-"):
        return api_key_value
    
    # Otherwise, treat it as an environment variable name
    api_key = os.getenv(api_key_value)
    if not api_key:
        raise ValueError(f"OpenAI API key not found in environment variable: {api_key_value}")
    
    return api_key


# =============================================================================
# 5. FACTORY PATTERN
# =============================================================================

class ClassifierFactory:
    """Factory for creating intent classifiers with proper configuration."""
    
    @staticmethod
    def create_classifier() -> LogprobClassifier:
        """Create a properly configured intent classifier."""
        try:
            config = load_config()
            api_key = get_api_key(config)
            classifier = LogprobClassifier(api_key)
            logger.info("Classifier created successfully via factory")
            return classifier
        except Exception as e:
            logger.error("Failed to create classifier", error=str(e))
            raise


# =============================================================================
# 6. DEMO FUNCTIONS
# =============================================================================




def interactive_classifier():
    """Interactive mode for testing custom queries."""
    print("🎯 Interactive Intent Classifier")
    print("=" * 50)
    
    # Create classifier using factory
    try:
        classifier = ClassifierFactory.create_classifier()
    except Exception as e:
        print(f"❌ Failed to initialize classifier: {str(e)}")
        return
    
    print("\nAvailable Intent Categories:")
    for i, intent in enumerate(IntentConfig.INTENT_CATEGORIES, 1):
        display_name = IntentConfig.INTENT_DISPLAY_NAMES[intent]
        print(f"{i:2d}. {display_name}")
    
    print("\n" + "=" * 50)
    
    while True:
        print("\nEnter your query (or 'quit' to exit):")
        query = input(">>> ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not query:
            print("⚠️ Please enter a query.")
            continue
        
        try:
            result = classifier.classify_with_distribution(query)
            print(ResultFormatter.format_interactive_result(result))
            
        except ValueError as e:
            print(f"⚠️ Invalid query: {str(e)}")
            logger.warning("Invalid query provided", error=str(e), query=query[:IntentConfig.QUERY_TRUNCATE_LENGTH])
        except Exception as e:
            print(f"❌ Error processing query: {str(e)}")
            logger.error("Unexpected error in interactive mode", error=str(e), query=query[:IntentConfig.QUERY_TRUNCATE_LENGTH])





# =============================================================================
# 7. MAIN INTERFACE
# =============================================================================

def show_help():
    """Show comprehensive help information."""
    print("\n📚 Intent Classification System - Help")
    print("=" * 60)
    print("""
🎯 WHAT THIS SYSTEM DOES:
This tool classifies user queries into 10 intent categories using OpenAI's 
logprobs feature for calibrated confidence scores and complete probability
distributions.

🔧 HOW IT WORKS:
1. Makes a single API call to GPT-4.1-nano with logprobs enabled
2. Maps model tokens to intent categories using discovered mappings
3. Creates complete probability distribution over all 10 categories
4. Provides confidence scores, uncertainty measures, and processing recommendations

📊 CONFIDENCE LEVELS:
• 🟢 High (≥70%): Trust the classification completely
• 🟡 Medium (40-69%): Good confidence, consider context
• 🔴 Low (<40%): Use safe fallback or ask for clarification

🎯 INTENT CATEGORIES:""")
    
    for i, intent in enumerate(IntentConfig.INTENT_CATEGORIES, 1):
        display_name = IntentConfig.INTENT_DISPLAY_NAMES[intent]
        rag_indicator = "📚 RAG" if intent in IntentConfig.RAG_INTENTS else "🤖 LLM"
        print(f"{i:2d}. {display_name} ({rag_indicator})")
    
    print("""
🛣️ PROCESSING PATHS:
• 📚 RAG (Retrieval-Augmented Generation): For queries needing external knowledge
• 🤖 LLM-only: For queries the model can answer directly

💡 EXAMPLE CLASSIFICATIONS:
• "Explain machine learning" → Concept/Explanation (🤖 LLM-only)
• "When was Python created?" → Fact Lookup (📚 RAG)
• "Summarize this document" → Summarization (📚 RAG)
• "Draft a welcome email" → Creative Generation (🤖 LLM-only)

⚙️ TECHNICAL DETAILS:
• Uses calibrated confidence via math.exp(logprob)
• Handles partial tokens (CON→concept, SUM→summarization)
• Normalizes probabilities to sum to 1.0
• Calculates entropy-based uncertainty measures
• Average processing time: ~500ms per query
""")
    
    input("\nPress Enter to continue...")


def main():
    """Main entry point with clear menu system."""
    print("🤖 OpenAI Logprobs Intent Classification System")
    print("=" * 60)
    print("GPT-4.1-nano version with calibrated confidence scoring")
    print("=" * 60)
    
    while True:
        print("\nSelect an option:")
        print("1. 🎯 Interactive Mode (test your own queries)")
        print("2. 📚 Show Help")
        print("3. 🚪 Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            interactive_classifier()
        elif choice == "2":
            show_help()
        elif choice == "3":
            print("👋 Thanks for using the Intent Classification System!")
            break
        else:
            print("⚠️ Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        print("Please check your configuration and try again.")

