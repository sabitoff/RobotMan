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
    """Handles mapping between OpenAI tokens and intent categories."""
    
    def __init__(self):
        # Token mappings discovered through testing with GPT-4.1-nano
        # These mappings handle both full tokens and partial tokens the model produces
        self.token_to_intent = {
            # Full tokens that work reliably
            "FACT": "fact_lookup_qa",
            "COMPARE": "comparison_decision_support",
            
            # Partial tokens the model actually produces
            "CON": "concept_explanation",      # for CONCEPT
            "NAV": "document_navigation",      # for NAVIGATE  
            "SUM": "summarization_digest",     # for SUMMARIZE
            "IN": "instruction_procedure",     # for INSTRUCT
            "RE": "rewrite_transform",         # for REWRITE
            "D": "creative_generation",        # for DRAFT
            "EX": "data_extraction_structuring", # for EXTRACT
            "CL": "meta_clarification",        # for CLARIFY
            
            # Full word backups (less common but possible)
            "CONCEPT": "concept_explanation",
            "NAVIGATE": "document_navigation",
            "SUMMARIZE": "summarization_digest",
            "INSTRUCT": "instruction_procedure", 
            "REWRITE": "rewrite_transform",
            "DRAFT": "creative_generation",
            "EXTRACT": "data_extraction_structuring",
            "CLARIFY": "meta_clarification"
        }
    
    def map_token_to_intent(self, token: str) -> Optional[str]:
        """Map a token to an intent category."""
        return self.token_to_intent.get(token.strip().upper())
    
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
    
    def get_display_intent(self) -> str:
        """Get formatted display name for the predicted intent."""
        return IntentConfig.INTENT_DISPLAY_NAMES.get(
            self.predicted_intent, 
            self.predicted_intent.replace("_", " ").title()
        )


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
            
            # Log the actual model token for debugging
            if response.choices[0].logprobs and response.choices[0].logprobs.content:
                model_token = response.choices[0].message.content.strip().upper()
                logger.info("Intent classified", 
                           model_token=model_token,
                           predicted_intent=predicted_intent,
                           confidence=confidence)
            
            # Calculate additional metrics
            uncertainty = self._calculate_uncertainty(distribution)
            confidence_level = self._get_confidence_level(confidence)
            processing_path = self._get_processing_path(predicted_intent)
            
            result = ClassificationResult(
                query=query,
                predicted_intent=predicted_intent,
                confidence=confidence,
                distribution=distribution,
                processing_path=processing_path,
                uncertainty=uncertainty,
                confidence_level=confidence_level
            )
            
            # Cache the result
            self._cache[cache_key] = result
            logger.debug("Cached result for query", query=query[:IntentConfig.QUERY_TRUNCATE_LENGTH])
            
            return result
            
        except Exception as e:
            logger.error("Error in classification", error=str(e), query=query[:IntentConfig.QUERY_TRUNCATE_LENGTH])
            return self._create_fallback_result(query)
    
    def _make_api_call(self, query: str):
        """Make the OpenAI API call with logprobs."""
        prompt = f"""Classify this query. Respond with ONLY one word from: CONCEPT, FACT, NAVIGATE, SUMMARIZE, COMPARE, INSTRUCT, REWRITE, DRAFT, EXTRACT, CLARIFY

Query: "{query}"

Answer:"""
        
        return self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1,
            logprobs=True,
            top_logprobs=10
        )
    

    
    def _create_complete_distribution(self, response) -> Dict[str, float]:
        """Create a complete probability distribution over all intents."""
        # Initialize all intents with tiny probabilities
        distribution = {intent: 1e-10 for intent in self.config.INTENT_CATEGORIES}
        
        if response.choices[0].logprobs and response.choices[0].logprobs.content:
            top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
            
            # Add actual probabilities from logprobs
            for token_info in top_logprobs:
                token = token_info.token.strip().upper()
                probability = math.exp(token_info.logprob)
                
                # Map to intent if valid
                intent = self.token_mapper.map_token_to_intent(token)
                if intent and intent in distribution:
                    distribution[intent] = probability
        
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
            confidence_level="low"
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

