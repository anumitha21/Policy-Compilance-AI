# ingestion/contract_parser.py

import re
import pdfplumber


class ContractParser:

    CLAUSE_PATTERN = re.compile(
        r"""
        (?=
            (?:^|\n)Clause\s+\d+  |
            \n\d+\.               |
            \n\(\d+\)
        )
        """,
        re.IGNORECASE | re.VERBOSE | re.MULTILINE
    )

    CLAUSE_START = re.compile(
        r"^(?:Clause\s+\d+|\d+\.|\(\d+\))",
        re.IGNORECASE
    )

    def load_pdf(
        self,
        pdf_path: str
    ) -> str:

        pages = []

        with pdfplumber.open(pdf_path) as pdf:

            for page in pdf.pages:

                text = page.extract_text()

                if text:
                    pages.append(text)

        return "\n".join(pages)

    def split_clauses(
        self,
        contract_text: str
    ) -> list[dict]:

        raw_clauses = re.split(
            self.CLAUSE_PATTERN,
            contract_text
        )

        clauses = []

        clause_counter = 1

        for clause in raw_clauses:

            clause = clause.strip()

            if not clause:
                continue

            if not self.CLAUSE_START.match(clause):
                print(f"[PARSER] Skipping preamble: {clause[:80]!r}")
                continue

            clauses.append(
                {
                    "clause_id":
                        f"CLAUSE_{clause_counter}",

                    "text":
                        clause,

                    "raw_text":
                        clause
                }
            )

            clause_counter += 1

        return clauses

    def parse_contract(
        self,
        pdf_path: str
    ) -> list[dict]:

        print(
            f"Parsing contract: {pdf_path}"
        )

        contract_text = self.load_pdf(
            pdf_path
        )

        clauses = self.split_clauses(
            contract_text
        )

        print(
            f"Extracted {len(clauses)} clauses"
        )

        return clauses