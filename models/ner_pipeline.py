"""
Biomedical NER Pipeline

Wrapper for d4data/biomedical-ner-all model.
"""

from typing import List, Dict, Any, Optional
from transformers import pipeline as hf_pipeline

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class NERPipeline:
    """
    Biomedical Named Entity Recognition pipeline.

    Recognizes:
    - Gene_or_gene_product
    - Drug/Chemical
    - Disease
    - Protein
    """

    def __init__(
        self,
        model_path: str = None,
        device: int = None
    ):
        """
        Initialize NER Pipeline.

        Args:
            model_path: Path to NER model
            device: Device ID (0 for cuda:0, -1 for CPU)
        """
        self.model_path = model_path or settings.NER_MODEL_PATH

        # Convert device string to device ID
        if device is None:
            if "cuda" in settings.DEVICE:
                try:
                    device = int(settings.DEVICE.split(":")[-1])
                except:
                    device = 0
            else:
                device = -1

        logger.info("Loading NER Pipeline", path=self.model_path, device=device)

        try:
            self.pipeline = hf_pipeline(
                "ner",
                model=self.model_path,
                aggregation_strategy="simple",
                device=device
            )
            logger.info("NER Pipeline loaded successfully")
        except Exception as e:
            logger.error("Failed to load NER model", error=str(e))
            self.pipeline = None

    def __call__(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from text.

        Args:
            text: Input text

        Returns:
            List of entity dictionaries with keys:
            - entity_group: Entity type
            - word: Entity text
            - score: Confidence score
            - start: Start position
            - end: End position
        """
        if not self.pipeline:
            logger.warning("NER pipeline not available")
            return []

        try:
            results = self.pipeline(text)
            logger.debug(
                "NER extraction completed",
                text=text[:100],
                num_entities=len(results)
            )
            return results
        except Exception as e:
            logger.error("NER extraction failed", error=str(e))
            return []

    def extract_genes(self, text: str) -> List[str]:
        """
        Extract gene names only.

        Args:
            text: Input text

        Returns:
            List of gene names
        """
        entities = self(text)

        gene_like_labels = {
            'Gene_or_gene_product',
            'Gene',
            'Protein',
            'Coreference',
            'Diagnostic_procedure'
        }

        genes = []
        for entity in entities:
            if entity.get('entity_group') in gene_like_labels:
                word = entity['word'].strip().replace('##', '')  # Clean BERT tokens
                if len(word) > 1:
                    genes.append(word.upper())

        return list(set(genes))  # Deduplicate

    def extract_drugs(self, text: str) -> List[str]:
        """
        Extract drug/chemical names only.

        Args:
            text: Input text

        Returns:
            List of drug names
        """
        entities = self(text)

        drug_like_labels = {
            'Drug',
            'Chemical',
            'Medication',
            'Pharmacologic_substance'
        }

        drugs = []
        for entity in entities:
            if entity.get('entity_group') in drug_like_labels:
                word = entity['word'].strip().replace('##', '')
                if len(word) > 1:
                    drugs.append(word)

        return list(set(drugs))

    def is_available(self) -> bool:
        """Check if NER pipeline is available"""
        return self.pipeline is not None
