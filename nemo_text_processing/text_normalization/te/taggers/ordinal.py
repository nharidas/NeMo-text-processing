# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.te.graph_utils import (
    NEMO_ALL_DIGIT,
    NEMO_ONE_DIGIT,
    NEMO_TELUGU_SIGMA,
    GraphFst,
)
from nemo_text_processing.text_normalization.te.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.te.utils import get_abs_path


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying Telugu ordinals, e.g.
        21వ -> ordinal { integer: "ఇరవై ఒకటవ" }

    Ordinals reuse cardinal verbalization, then rewrite only the final morpheme:
        - input exceptions (1వ, 1st-15th) from exceptions.tsv
        - append-వ endings (round tens, 100) from ordinal_endings.tsv
        - default: replace final vowel sign with వ from char_rewrites.tsv
    """

    def __init__(self, cardinal: CardinalFst, deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        exceptions = pynini.string_file(get_abs_path("data/ordinal/exceptions.tsv"))
        word_endings = pynini.string_file(get_abs_path("data/ordinal/ordinal_endings.tsv"))
        char_rewrite = pynini.string_file(get_abs_path("data/ordinal/char_rewrites.tsv"))

        suffixes = pynini.project(pynini.string_file(get_abs_path("data/ordinal/suffixes.tsv")), "input")
        suffix = pynutil.delete(suffixes)

        ordinal_rewrite = word_endings | char_rewrite
        cardinal_to_ordinal = (NEMO_TELUGU_SIGMA + ordinal_rewrite).optimize()

        numbers_except_standalone_one = pynini.difference(
            pynini.closure(NEMO_ALL_DIGIT, 1), NEMO_ONE_DIGIT
        ).optimize()

        ordinal_graph = (
            numbers_except_standalone_one @ cardinal.final_graph @ cardinal_to_ordinal
        ) + suffix

        graph = pynini.union(exceptions, ordinal_graph).optimize()

        final_graph = pynutil.insert('integer: "') + graph + pynutil.insert('"')
        self.fst = self.add_tokens(final_graph).optimize()