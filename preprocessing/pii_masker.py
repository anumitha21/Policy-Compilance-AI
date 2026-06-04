
# preprocessing/pii_masker.py

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine


class PIIMasker:

    TARGET_ENTITIES = [
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "CREDIT_CARD",
        "IBAN_CODE",
        "US_BANK_NUMBER",
        "US_SSN"
    ]

    def __init__(self):

        self.analyzer = AnalyzerEngine()

        self.anonymizer = AnonymizerEngine()

    def mask_clause(
        self,
        clause: dict
    ) -> dict:

        text = clause["text"]

        analyzer_results = self.analyzer.analyze(
            text=text,
            language="en",
            entities=self.TARGET_ENTITIES
        )

        anonymized = self.anonymizer.anonymize(
            text=text,
            analyzer_results=analyzer_results
        )

        clause["masked_text"] = anonymized.text

        return clause

    def mask_clauses(
        self,
        clauses: list[dict]
    ) -> list[dict]:

        processed = []

        for clause in clauses:

            processed.append(
                self.mask_clause(clause)
            )

        return processed