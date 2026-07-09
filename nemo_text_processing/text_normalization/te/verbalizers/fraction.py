# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
    MINUS,
    NEMO_NOT_QUOTE,
    GraphFst,
)


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing Telugu fractions.

    e.g.
    fraction { integer_part: "ఇరవై మూడు" numerator: "నాలుగు" denominator: "ఆరు" }
    -> ఇరవై మూడు మరియు నాలుగు బై ఆరు

    e.g.
    fraction { numerator: "నాలుగు" denominator: "ఆరు" }
    -> నాలుగు బై ఆరు

    e.g.
    fraction { morphosyntactic_features: "అర" }
    -> అర

    e.g.
    fraction { integer_part: "నూట ముప్పై మూడు" morphosyntactic_features: "అర" }
    -> నూట ముప్పై మూడు మరియు అర
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="fraction", kind="verbalize", deterministic=deterministic)

        optional_sign = pynini.closure(
            pynini.cross("negative: \"true\"", MINUS) + pynutil.delete(" "),
            0,
            1,
        )

        integer = (
            pynutil.delete("integer_part: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        numerator = (
            pynutil.delete("numerator: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\" ")
        )

        denominator = (
            pynutil.delete("denominator: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        morphosyntactic_features = (
            pynutil.delete("morphosyntactic_features: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        insert_bai = pynutil.insert(" బై ")
        insert_mariyu = pynutil.insert(" మరియు ")

        # numerator + denominator
        # నాలుగు బై ఆరు
        fraction_default = numerator + insert_bai + denominator

        # integer_part + numerator + denominator
        # ఇరవై మూడు మరియు నాలుగు బై ఆరు
        mixed_fraction_default = (
            integer
            + pynutil.delete(" ")
            + insert_mariyu
            + fraction_default
        )

        # morphosyntactic_features only
        # అర / పావు / ముప్పావు / ఒకటిన్నర / రెండున్నర
        lexical_fraction = morphosyntactic_features

        # integer_part + morphosyntactic_features
        # నూట ముప్పై మూడు మరియు అర
        mixed_lexical_fraction = (
            integer
            + pynutil.delete(" ")
            + insert_mariyu
            + morphosyntactic_features
        )

        self.graph = optional_sign + (
            mixed_lexical_fraction
            | lexical_fraction
            | mixed_fraction_default
            | fraction_default
        )

        delete_tokens = self.delete_tokens(self.graph)
        self.fst = delete_tokens.optimize()