"""The 14 additional CUAD categories, with the audited C2 visibility contract."""
from functools import cached_property
from .base import load_source, generic_term_predict
from .cuad import CUADAdapter


class CUADXAdapter(CUADAdapter):
    source = 'CUADX'
    visible_prefix = 'CUADX'
    pseudonymiser_version = 'audited-cuadx-cache-v1'

    @cached_property
    def api(self):
        return load_source('cuadx_sources', self.permutation_seed)

    def keyword_predict(self, question, text):
        return generic_term_predict(self.questions()[question]['options'], text)
