"""
Enhanced Semantic Chunker for Steganography

Splits text into meaningful semantic chunks with advanced features:
1. Synonym expansion - Generate alternative phrasings for better matches
2. Concept decomposition - Break abstract concepts into concrete ones
3. Hierarchical chunking - Multiple granularity levels

These features improve semantic coverage and make steganographic
encoding more robust across diverse message types.
"""

import re
from typing import List, Optional, Dict, Tuple, Set
from dataclasses import dataclass, field


@dataclass
class EnhancedChunk:
    """
    Represents a semantic chunk with expansions.
    
    Attributes:
        original: The original chunk text
        normalized: Normalized form
        synonyms: Alternative phrasings
        concrete_forms: Concrete decompositions of abstract concepts
        sub_chunks: Finer-grained sub-chunks (hierarchical)
    """
    original: str
    normalized: str
    synonyms: List[str] = field(default_factory=list)
    concrete_forms: List[str] = field(default_factory=list)
    sub_chunks: List[str] = field(default_factory=list)
    
    def all_variants(self) -> List[str]:
        """Get all variants of this chunk for matching."""
        variants = [self.normalized]
        variants.extend(self.synonyms)
        variants.extend(self.concrete_forms)
        return list(dict.fromkeys(variants))  # Remove duplicates, preserve order


class SemanticChunker:
    """
    Enhanced semantic chunker with synonym expansion and concept decomposition.
    
    Features:
    1. Multiple chunking strategies (sentence, clause, phrase)
    2. Synonym expansion using WordNet-like mappings
    3. Abstract-to-concrete concept decomposition
    4. Hierarchical sub-chunking
    
    Example:
        >>> chunker = SemanticChunker(strategy="clause", expand_synonyms=True)
        >>> chunks = chunker.chunk("A secret meeting at the bank")
        >>> # Returns chunks with synonyms like "covert gathering", "financial institution"
    """
    
    # Clause delimiters
    CLAUSE_DELIMITERS = r',|\band\b|\bor\b|\bbut\b|\bthen\b|\bwhile\b|\bwhen\b'
    
    # Synonym mappings for common steganography-related concepts
    # These help map abstract/secret concepts to concrete visual ones
    SYNONYM_MAP: Dict[str, List[str]] = {
        # Meeting/gathering
        "meeting": ["gathering", "people together", "group discussion", "assembly"],
        "secret": ["hidden", "private", "quiet", "concealed"],
        "covert": ["hidden", "undercover", "secret", "stealthy"],
        "rendezvous": ["meeting point", "gathering place", "meetup"],
        
        # Communication
        "message": ["letter", "note", "communication", "text"],
        "information": ["data", "details", "facts", "content"],
        "signal": ["sign", "gesture", "indication", "wave"],
        "code": ["cipher", "symbol", "pattern", "secret writing"],
        
        # Locations
        "location": ["place", "spot", "area", "site"],
        "headquarters": ["main building", "office", "center"],
        "safe house": ["hidden shelter", "secure building", "protected place"],
        
        # Actions
        "transfer": ["handover", "exchange", "delivery", "passing"],
        "escape": ["flee", "run away", "departure", "exit"],
        "surveillance": ["watching", "monitoring", "observation"],
        "infiltrate": ["enter secretly", "sneak in", "penetrate"],
        
        # Time
        "dawn": ["sunrise", "early morning", "first light"],
        "dusk": ["sunset", "evening", "twilight"],
        "midnight": ["late night", "dark hours", "nighttime"],
        
        # Abstract concepts that need visual grounding
        "danger": ["warning sign", "risk", "threat", "hazard"],
        "safety": ["protection", "security", "shelter"],
        "success": ["victory", "achievement", "winning", "triumph"],
        "failure": ["defeat", "loss", "problem"],
        "money": ["cash", "currency", "coins", "bills", "payment"],
        "weapon": ["gun", "knife", "tool", "equipment"],
        
        # Technical terms
        "encryption": ["locked", "secured", "coded", "protected"],
        "network": ["connections", "web", "grid", "system"],
        "server": ["computer", "machine", "system"],
        "database": ["storage", "records", "files"],
        "algorithm": ["process", "method", "procedure"],
        
        # Military/Strategic
        "strategy": ["plan", "approach", "tactic", "method"],
        "operation": ["mission", "task", "action", "activity"],
        "target": ["goal", "objective", "destination", "aim"],
        "asset": ["resource", "valuable", "person", "item"],
        "extraction": ["removal", "rescue", "retrieval", "pickup"],
    }
    
    # Abstract to concrete mappings
    # Maps abstract concepts to more visually concrete descriptions
    ABSTRACT_TO_CONCRETE: Dict[str, List[str]] = {
        # Emotional/Abstract states
        "happiness": ["smiling person", "celebration", "party", "laughing"],
        "sadness": ["crying", "tears", "funeral", "rain"],
        "anger": ["yelling", "fighting", "red face", "argument"],
        "fear": ["hiding", "running away", "dark place", "scared face"],
        "love": ["couple kissing", "heart", "wedding", "holding hands"],
        "trust": ["handshake", "friends", "teamwork"],
        
        # Abstract actions
        "communication": ["people talking", "phone call", "letter writing"],
        "transportation": ["car driving", "train", "airplane", "walking"],
        "transaction": ["money exchange", "handshake", "shopping"],
        "education": ["classroom", "books", "teacher", "students"],
        "healthcare": ["doctor", "hospital", "medicine", "nurse"],
        
        # Abstract nouns
        "government": ["capitol building", "flag", "officials", "voting"],
        "technology": ["computer", "smartphone", "robot", "electronics"],
        "nature": ["trees", "mountains", "ocean", "animals"],
        "urban": ["city skyline", "buildings", "streets", "traffic"],
        "rural": ["farm", "countryside", "fields", "barn"],
        
        # Steganography-specific
        "secret meeting": ["people whispering", "private room", "closed door"],
        "hidden message": ["folded paper", "envelope", "coded text"],
        "covert operation": ["night scene", "shadows", "dark clothing"],
        "intelligence": ["documents", "files", "computer screen"],
        "mission": ["person walking", "destination", "journey"],
        
        # Financial
        "investment": ["money growing", "charts", "stocks"],
        "payment": ["wallet", "cash register", "credit card"],
        "debt": ["bills", "invoices", "worried person"],
        
        # Scientific
        "research": ["laboratory", "scientist", "experiments"],
        "discovery": ["light bulb", "eureka moment", "finding"],
        "experiment": ["lab equipment", "test tubes", "scientist"],
    }
    
    def __init__(
        self,
        strategy: str = "clause",
        min_chunk_length: int = 3,
        max_chunk_length: int = 100,
        expand_synonyms: bool = True,
        decompose_concepts: bool = True,
        hierarchical: bool = False,
        max_synonyms: int = 3,
        max_concrete: int = 2,
    ):
        """
        Initialize the enhanced chunker.
        
        Args:
            strategy: Chunking strategy ('sentence', 'clause', 'phrase')
            min_chunk_length: Minimum chunk length in characters
            max_chunk_length: Maximum chunk length in characters
            expand_synonyms: Whether to generate synonym expansions
            decompose_concepts: Whether to decompose abstract concepts
            hierarchical: Whether to generate hierarchical sub-chunks
            max_synonyms: Maximum synonym alternatives per chunk
            max_concrete: Maximum concrete forms per abstract concept
        """
        self.strategy = strategy
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length
        self.expand_synonyms = expand_synonyms
        self.decompose_concepts = decompose_concepts
        self.hierarchical = hierarchical
        self.max_synonyms = max_synonyms
        self.max_concrete = max_concrete
    
    def chunk(self, text: str) -> List[str]:
        """
        Split text into semantic chunks (simple interface).
        
        For advanced features, use chunk_enhanced().
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        enhanced = self.chunk_enhanced(text)
        return [c.normalized for c in enhanced]
    
    def chunk_enhanced(self, text: str) -> List[EnhancedChunk]:
        """
        Split text into enhanced semantic chunks with expansions.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of EnhancedChunk objects with synonyms and concrete forms
        """
        # Normalize text
        normalized = self._normalize(text)
        
        # Apply chunking strategy
        if self.strategy == "sentence":
            raw_chunks = self._chunk_sentences(normalized)
        elif self.strategy == "clause":
            raw_chunks = self._chunk_clauses(normalized)
        elif self.strategy == "phrase":
            raw_chunks = self._chunk_phrases(normalized)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Filter and clean chunks
        raw_chunks = self._filter_chunks(raw_chunks)
        
        # Build enhanced chunks
        enhanced_chunks = []
        for chunk in raw_chunks:
            enhanced = EnhancedChunk(
                original=chunk,
                normalized=chunk,
            )
            
            # Add synonym expansions
            if self.expand_synonyms:
                enhanced.synonyms = self._expand_synonyms(chunk)
            
            # Add concrete decompositions
            if self.decompose_concepts:
                enhanced.concrete_forms = self._decompose_abstract(chunk)
            
            # Add hierarchical sub-chunks
            if self.hierarchical:
                enhanced.sub_chunks = self._create_sub_chunks(chunk)
            
            enhanced_chunks.append(enhanced)
        
        return enhanced_chunks
    
    def get_all_variants(self, text: str) -> List[List[str]]:
        """
        Get all query variants for each chunk.
        
        Useful for hierarchical encoding where we want to try
        multiple queries per chunk.
        
        Args:
            text: Input text
            
        Returns:
            List of variant lists, one per chunk
        """
        enhanced = self.chunk_enhanced(text)
        return [chunk.all_variants() for chunk in enhanced]
    
    def _normalize(self, text: str) -> str:
        """Normalize text for chunking."""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'["\'\(\)\[\]\{\}]', '', text)
        return text.strip()
    
    def _chunk_sentences(self, text: str) -> List[str]:
        """Split on sentence boundaries."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _chunk_clauses(self, text: str) -> List[str]:
        """Split on clause boundaries."""
        chunks = re.split(self.CLAUSE_DELIMITERS, text, flags=re.IGNORECASE)
        return [c.strip() for c in chunks if c.strip()]
    
    def _chunk_phrases(self, text: str) -> List[str]:
        """Split on phrase boundaries."""
        delimiters = r',|\band\b|\bor\b|\bof\b|\bin\b|\bwith\b|\bto\b|\bfor\b'
        chunks = re.split(delimiters, text, flags=re.IGNORECASE)
        return [c.strip() for c in chunks if c.strip()]
    
    def _filter_chunks(self, chunks: List[str]) -> List[str]:
        """Filter chunks based on length constraints."""
        filtered = []
        for chunk in chunks:
            if len(chunk) < self.min_chunk_length:
                continue
            if len(chunk) > self.max_chunk_length:
                chunk = chunk[:self.max_chunk_length]
            chunk = chunk.strip(' ,.-:;')
            if chunk:
                filtered.append(chunk)
        return filtered
    
    def _expand_synonyms(self, chunk: str) -> List[str]:
        """
        Generate synonym expansions for a chunk.
        
        Looks for known words/phrases and generates alternatives.
        """
        synonyms = []
        chunk_lower = chunk.lower()
        
        # Check for exact phrase matches first
        for phrase, alternatives in self.SYNONYM_MAP.items():
            if phrase in chunk_lower:
                for alt in alternatives[:self.max_synonyms]:
                    expanded = chunk_lower.replace(phrase, alt)
                    if expanded != chunk_lower and expanded not in synonyms:
                        synonyms.append(expanded)
        
        # Check for individual word matches
        words = chunk_lower.split()
        for i, word in enumerate(words):
            if word in self.SYNONYM_MAP:
                for alt in self.SYNONYM_MAP[word][:self.max_synonyms]:
                    new_words = words.copy()
                    new_words[i] = alt
                    expanded = " ".join(new_words)
                    if expanded not in synonyms:
                        synonyms.append(expanded)
        
        return synonyms[:self.max_synonyms]
    
    def _decompose_abstract(self, chunk: str) -> List[str]:
        """
        Decompose abstract concepts into concrete visual forms.
        
        This helps match abstract messages to visual media.
        """
        concrete = []
        chunk_lower = chunk.lower()
        
        # Check for abstract concepts
        for abstract, concretes in self.ABSTRACT_TO_CONCRETE.items():
            if abstract in chunk_lower:
                for c in concretes[:self.max_concrete]:
                    # Replace abstract with concrete
                    decomposed = chunk_lower.replace(abstract, c)
                    if decomposed != chunk_lower and decomposed not in concrete:
                        concrete.append(decomposed)
                
                # Also add the concrete form directly
                for c in concretes[:self.max_concrete]:
                    if c not in concrete:
                        concrete.append(c)
        
        return concrete[:self.max_concrete * 2]
    
    def _create_sub_chunks(self, chunk: str) -> List[str]:
        """
        Create hierarchical sub-chunks for finer-grained matching.
        """
        sub_chunks = []
        words = chunk.split()
        
        # Create bigrams
        if len(words) >= 2:
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                if len(bigram) >= self.min_chunk_length:
                    sub_chunks.append(bigram)
        
        # Create trigrams
        if len(words) >= 3:
            for i in range(len(words) - 2):
                trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                if len(trigram) >= self.min_chunk_length:
                    sub_chunks.append(trigram)
        
        return sub_chunks
    
    def __repr__(self) -> str:
        features = []
        if self.expand_synonyms:
            features.append("synonyms")
        if self.decompose_concepts:
            features.append("concepts")
        if self.hierarchical:
            features.append("hierarchical")
        feature_str = "+".join(features) if features else "basic"
        return f"SemanticChunker(strategy={self.strategy}, features={feature_str})"


# Convenience function
def chunk_message(
    message: str,
    strategy: str = "clause",
    expand: bool = True,
) -> List[EnhancedChunk]:
    """
    Convenience function to chunk a message with enhancements.
    
    Args:
        message: The message to chunk
        strategy: Chunking strategy
        expand: Whether to expand synonyms and decompose concepts
        
    Returns:
        List of EnhancedChunk objects
    """
    chunker = SemanticChunker(
        strategy=strategy,
        expand_synonyms=expand,
        decompose_concepts=expand,
    )
    return chunker.chunk_enhanced(message)
