from .fasta import FastaRecord, deduplicate_fasta_records, parse_fasta, write_fasta
from .http import HttpClient, HttpError, HttpResponse
from .uniprot import UniProtRecord

__all__ = [
    "FastaRecord",
    "HttpClient",
    "HttpError",
    "HttpResponse",
    "UniProtRecord",
    "deduplicate_fasta_records",
    "parse_fasta",
    "write_fasta",
]
