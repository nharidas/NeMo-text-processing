# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved. 
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

import logging
import os
import pynini
from pynini.lib import pynutil
from nemo_text_processing.text_normalization.te.graph_utils import (
    NEMO_SPACE,
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)
 
from nemo_text_processing.text_normalization.te.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.te.taggers.punctuation import PunctuationFst
from nemo_text_processing.text_normalization.te.taggers.word import WordFst
 
class ClassifyFst(GraphFst):
    """
    Final class that composes all other classification grammars. This class
    can process an entire sentence including punctuation.
    For deployment, this grammar will be compiled and exported to OpenFst
    Finite State Archive (FAR) File. More details to deployment at
    NeMo/tools/text_processing_deployment.
    Args:
        input_case: accepting either "lower_cased" or "cased" input.
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
        cache_dir: path to a dir with .far grammar file. Set to None to
            avoid using cache.
        overwrite_cache: set to True to overwrite .far files
        whitelist: path to a file with whitelist replacements
    """
    
    def __init__(
        self,
        input_case: str,
        deterministic: bool = True,
        cache_dir: str = None,
        overwrite_cache: bool = False,
        whitelist: str = None,
    ): 
        super().__init__(name="tokenize_and_classify", kind="classify", deterministic=deterministic)
        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True) 
            whitelist_file = os.path.basename(whitelist) if whitelist else ""
            far_file = os.path.join(
                cache_dir,
                f"te_tn_{deterministic}_deterministic_{input_case}_{whitelist_file}_tokenize.far",
            )
        if not overwrite_cache and far_file and os.path.exists(far_file): 
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logging.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
 
            logging.info(f"Creating ClassifyFst grammars.")
            # --- Active taggers ---
            cardinal = CardinalFst(deterministic=deterministic)
            cardinal_graph = cardinal.fst
            punctuation = PunctuationFst(deterministic=deterministic)
            punct_graph = punctuation.fst
            word = WordFst(punctuation=punctuation, deterministic=deterministic)
            word_graph = word.fst
            classify = (
                pynutil.add_weight(cardinal_graph, 1.1)
            )
 
            classify = pynini.union(classify, pynutil.add_weight(word_graph, 100))
            token = pynutil.insert("tokens { ") + classify + pynutil.insert(" }")
            graph = token + pynini.closure(
                pynini.compose(pynini.closure(NEMO_WHITE_SPACE, 1), delete_extra_space) + token
            )
            graph = delete_space + graph + delete_space 
            self.fst = graph.optimize()
 
            if far_file:
 
                generator_main(far_file, {"tokenize_and_classify": self.fst})
 
                logging.info(f"ClassifyFst grammars are saved to {far_file}.")