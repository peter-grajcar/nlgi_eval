#!/usr/bin/env python

# Copyright 2020 Zdeněk Kasner
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications Copyright 2022 Peter Grajcar. See GitHub history for details.

import logging
import heapq

from lm_scorer.models.auto import AutoLMScorer as LMScorer
from pprint import pprint as pp
from utils.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class SentenceScorer:
    def __init__(self, reduce_mode="gmean", device="cuda"):
        if device == "cpu":
            logger.warning("Running LMScorer on CPU. Scoring may be slow.")

        self.model = LMScorer.from_pretrained("gpt2", device=device, batch_size=1)
        self.reduce_mode = reduce_mode
        self.tokenizer = Tokenizer()

    def score(self, sentence):
        sentence = self.tokenizer.detokenize(sentence)

        return self.model.sentence_score(sentence, reduce=self.reduce_mode, log=True)

    def select_best(self, sentences):
        return heapq.nlargest(1, sentences, lambda sent: self.score(sent))[0]
