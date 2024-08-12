# Omid Davar @ 2023


import numpy as np
from Scripts.DataManager.GraphConstructor.GraphConstructor import GraphConstructor
from torch_geometric.data import HeteroData
from Scripts.Configs.ConfigClass import Config
import torch
import os
from typing import List, Dict, Tuple

import stanza
import copy
from stanza.pipeline.core import DownloadMethod


class TagDepTokenGraphConstructor(GraphConstructor):

    class _Variables(GraphConstructor._Variables):
        def __init__(self):
            super(TagDepTokenGraphConstructor._Variables, self).__init__()
            self.nlp_pipeline: str = ''

    def __init__(self, texts: List[str], save_path: str, config: Config,
                 load_preprocessed_data=False, naming_prepend='', use_compression=True, use_sentence_nodes=False, use_general_node=True, start_data_load=0, end_data_load=-1, num_general_nodes=1):

        super(TagDepTokenGraphConstructor, self)\
            .__init__(texts, self._Variables(), save_path, config, load_preprocessed_data, naming_prepend, use_compression, start_data_load, end_data_load)
        self.settings = {"dep_token_weight": 1, "token_token_weight": 2, "tag_token_weight": 1,
                         "general_token_weight": 1, "general_sentence_weight": 1, "token_sentence_weight": 1}
        self.use_sentence_nodes = use_sentence_nodes
        self.use_general_node = use_general_node

        self.var.graph_num = len(self.raw_data)

        # farsi
        self.nlp = config.nlp

        self.token_lemma = stanza.Pipeline(
            "fa", download_method=DownloadMethod.REUSE_RESOURCES, processors=["tokenize", "lemma", "pos"])

        self.dependencies = ['acl', 'acl:relcl', 'advcl', 'advcl:relcl', 'advmod', 'advmod:emph', 'advmod:lmod', 'amod', 'appos', 'aux', 'aux:pass', 'case', 'cc', 'cc:preconj', 'ccomp', 'clf', 'compound', 'compound:lvc', 'compound:prt', 'compound:redup', 'compound:svc', 'conj', 'cop', 'csubj', 'csubj:outer', 'csubj:pass', 'dep', 'det', 'det:numgov', 'det:nummod', 'det:poss', 'discourse',
                             'dislocated', 'expl', 'expl:impers', 'expl:pass', 'expl:pv', 'fixed', 'flat', 'flat:foreign', 'flat:name', 'goeswith', 'iobj', 'list', 'mark', 'nmod', 'nmod:poss', 'nmod:tmod', 'nsubj', 'nsubj:outer', 'nsubj:pass', 'nummod', 'nummod:gov', 'obj', 'obl', 'obl:agent', 'obl:arg', 'obl:lmod', 'obl:tmod', 'orphan', 'parataxis', 'punct', 'reparandum', 'root', 'vocative', 'xcomp']

        self.tags = ['NOUN', 'DET', 'PROPN', 'NUM', 'VERB', 'PART', 'PRON',
                     'SCONJ', 'ADJ', 'ADP', 'PUNCT', 'ADV', 'AUX', 'SYM', 'INTJ', 'CCONJ', 'X']

        self.num_general_nodes = num_general_nodes

        self.word_ids = self.get_word_by_id()

    def to_graph(self, text: str):
        # farsi
        doc_sentences = []
        doc = []
        doc.append(doc_sentences)
        token_list = self.token_lemma(text)
        for idx, sentence in enumerate(token_list.sentences):
            doc_sentences.append((sentence.text, sentence.tokens[0].text, idx))
            for word in sentence.words:
                doc.append((idx, word.text, word.lemma,
                            word.upos, word.head, word.deprel))

        # if len(doc) < 2:
        #     return
        if self.use_sentence_nodes:
            return self.__create_graph_with_sentences(doc)
        else:
            return self.__create_graph(doc, use_general_node=self.use_general_node)

    def __find_dep_index(self, dependency: str):
        for dep_idx in range(len(self.dependencies)):
            if self.dependencies[dep_idx] == dependency:
                return dep_idx
        return -1  # means not found

    def __build_initial_dependency_vectors(self, dep_length: int):
        # return torch.zeros((dep_length, self.nlp.vocab.vectors_length), dtype=torch.float32)
        # return torch.nn.functional.one_hot(torch.arange(0, dep_length), num_classes=-1).to(torch.float32)
        return torch.arange(0, dep_length)

    def __find_tag_index(self, tag: str):
        for tag_idx in range(len(self.tags)):
            if self.tags[tag_idx] == tag:
                return tag_idx
        return -1  # means not found

    def __build_initial_tag_vectors(self, tags_length: int):
        # return torch.zeros((tags_length, self.nlp.vocab.vectors_length), dtype=torch.float32)
        # return torch.nn.functional.one_hot(torch.arange(0, tags_length), num_classes=-1).to(torch.float32)
        return torch.arange(0, tags_length)

    def __build_initial_general_vector(self, num: int = 1):
        return torch.zeros((num, self.nlp.get_dimension()), dtype=torch.float32)

    def __create_graph_with_sentences(self, doc, for_compression=False):
        data = self.__create_graph(doc, for_compression, False)
        sentence_embeddings = np.array(
            [self.nlp.get_sentence_vector(sent[0]) for sent in doc[0]])
        data['sentence'].x = torch.tensor(
            sentence_embeddings, dtype=torch.float32)
        if self.use_general_node:
            if for_compression:
                data['general'].x = torch.full((1,), 0, dtype=torch.float32)
            else:
                data['general'].x = self.__build_initial_general_vector()
        sentence_general_edge_index = []
        general_sentence_edge_index = []
        sentence_word_edge_index = []
        word_sentence_edge_index = []
        sentence_general_edge_attr = []
        general_sentence_edge_attr = []
        sentence_word_edge_attr = []
        word_sentence_edge_attr = []
        if self.use_general_node:
            for i, _x in enumerate(doc[0]):
                # connecting sentences to general node
                sentence_general_edge_index.append([i, 0])
                general_sentence_edge_index.append([0, i])
                sentence_general_edge_attr.append(
                    self.settings['general_sentence_weight'])
                # different weight for directed edges can be set in the future
                general_sentence_edge_attr.append(
                    self.settings['general_sentence_weight'])
        sent_index = -1
        doc_copy = copy.deepcopy(doc)
        for i, token in enumerate(doc_copy[1:]):
            # connecting words to sentences
            for j, sent_start in enumerate(doc_copy[0]):
                if token[1] == sent_start[1] and token[0] == sent_start[2]:
                    sent_index += 1
                    doc_copy[0].pop(j)
            word_sentence_edge_index.append([i, sent_index])
            sentence_word_edge_index.append([sent_index, i])
            word_sentence_edge_attr.append(
                self.settings['token_sentence_weight'])
            sentence_word_edge_attr.append(
                self.settings['token_sentence_weight'])
        if self.use_general_node:
            data['general', 'general_sentence', 'sentence'].edge_index = torch.transpose(
                torch.tensor(general_sentence_edge_index, dtype=torch.int32), 0, 1)
            data['sentence', 'sentence_general', 'general'].edge_index = torch.transpose(
                torch.tensor(sentence_general_edge_index, dtype=torch.int32), 0, 1)
            data['general', 'general_sentence', 'sentence'].edge_attr = torch.tensor(
                general_sentence_edge_attr, dtype=torch.float32)
            data['sentence', 'sentence_general', 'general'].edge_attr = torch.tensor(
                sentence_general_edge_attr, dtype=torch.float32)
        data['word', 'word_sentence', 'sentence'].edge_index = torch.transpose(
            torch.tensor(word_sentence_edge_index, dtype=torch.int32), 0, 1)
        data['sentence', 'sentence_word', 'word'].edge_index = torch.transpose(
            torch.tensor(sentence_word_edge_index, dtype=torch.int32), 0, 1)
        data['word', 'word_sentence', 'sentence'].edge_attr = torch.tensor(
            word_sentence_edge_attr, dtype=torch.float32)
        data['sentence', 'sentence_word', 'word'].edge_attr = torch.tensor(
            sentence_word_edge_attr, dtype=torch.float32)
        return data

    def __create_graph(self, doc, for_compression=False, use_general_node=True):
        # nodes size is dependencies + tokens
        data = HeteroData()
        dep_length = len(self.dependencies)
        tag_length = len(self.tags)
        data['dep'].length = dep_length
        data['tag'].length = tag_length
        if for_compression:
            data['dep'].x = torch.full((dep_length,), -1, dtype=torch.float32)
            data['word'].x = [-1 for i in range(len(doc))]
            data['tag'].x = torch.full((tag_length,), -1, dtype=torch.float32)
            if use_general_node:
                data['general'].x = torch.full((1,), -1, dtype=torch.float32)
        else:
            data['dep'].x = self.__build_initial_dependency_vectors(dep_length)
            data['word'].x = torch.zeros(
                (len(doc), self.nlp.get_dimension()), dtype=torch.float32)
            data['tag'].x = self.__build_initial_tag_vectors(tag_length)
            if use_general_node:
                data['general'].x = self.__build_initial_general_vector()
        word_dep_edge_index = []
        dep_word_edge_index = []
        word_tag_edge_index = []
        tag_word_edge_index = []
        word_word_edge_index = []
        word_general_edge_index = []
        general_word_edge_index = []
        word_dep_edge_attr = []
        dep_word_edge_attr = []
        word_tag_edge_attr = []
        tag_word_edge_attr = []
        word_word_edge_attr = []
        word_general_edge_attr = []
        general_word_edge_attr = []
        for i, token in enumerate(doc[1:]):
            token_id = self.nlp.get_word_id(token[2])
            if token_id != -1:
                if for_compression:
                    data['word'].x[i] = token_id
                else:
                    data['word'].x[i] = torch.tensor(
                        self.nlp.get_word_vector(token[2]))
            # adding dependency edges
            if token[5] != 'root':
                dep_idx = self.__find_dep_index(token[5])
                if dep_idx != -1:
                    word_dep_edge_index.append([token[4], dep_idx])
                    word_dep_edge_attr.append(
                        self.settings["dep_token_weight"])
                    dep_word_edge_index.append([dep_idx, i])
                    dep_word_edge_attr.append(
                        self.settings["dep_token_weight"])
            # adding tag edges
            tag_idx = self.__find_tag_index(token[3])
            if tag_idx != -1:
                word_tag_edge_index.append([i, tag_idx])
                word_tag_edge_attr.append(self.settings["tag_token_weight"])
                tag_word_edge_index.append([tag_idx, i])
                tag_word_edge_attr.append(self.settings["tag_token_weight"])
            # adding sequence edges
            if i != len(doc) - 1:
                # using zero vectors for edge features
                word_word_edge_index.append([i, i + 1])
                word_word_edge_attr.append(self.settings["token_token_weight"])
                word_word_edge_index.append([i + 1, i])
                word_word_edge_attr.append(self.settings["token_token_weight"])
            # adding general node edges
            if use_general_node:
                word_general_edge_index.append([i, 0])
                word_general_edge_attr.append(
                    self.settings["general_token_weight"])
                general_word_edge_index.append([0, i])
                general_word_edge_attr.append(
                    self.settings["general_token_weight"])
        data['dep', 'dep_word', 'word'].edge_index = torch.transpose(torch.tensor(
            dep_word_edge_index, dtype=torch.int32), 0, 1) if len(dep_word_edge_index) > 0 else torch.empty(2, 0, dtype=torch.int32)
        data['word', 'word_dep', 'dep'].edge_index = torch.transpose(torch.tensor(
            word_dep_edge_index, dtype=torch.int32), 0, 1) if len(word_dep_edge_index) > 0 else torch.empty(2, 0, dtype=torch.int32)
        data['tag', 'tag_word', 'word'].edge_index = torch.transpose(torch.tensor(
            tag_word_edge_index, dtype=torch.int32), 0, 1) if len(tag_word_edge_index) > 0 else torch.empty(2, 0, dtype=torch.int32)
        data['word', 'word_tag', 'tag'].edge_index = torch.transpose(torch.tensor(
            word_tag_edge_index, dtype=torch.int32), 0, 1) if len(word_tag_edge_index) > 0 else torch.empty(2, 0, dtype=torch.int32)
        data['word', 'seq', 'word'].edge_index = torch.transpose(torch.tensor(word_word_edge_index, dtype=torch.int32), 0, 1) if len(
            word_word_edge_index) > 0 else torch.empty(2, 0, dtype=torch.int32)
        data['dep', 'dep_word', 'word'].edge_attr = torch.tensor(
            dep_word_edge_attr, dtype=torch.float32)
        data['word', 'word_dep', 'dep'].edge_attr = torch.tensor(
            word_dep_edge_attr, dtype=torch.float32)
        data['tag', 'tag_word', 'word'].edge_attr = torch.tensor(
            tag_word_edge_attr, dtype=torch.float32)
        data['word', 'word_tag', 'tag'].edge_attr = torch.tensor(
            word_tag_edge_attr, dtype=torch.float32)
        data['word', 'seq', 'word'].edge_attr = torch.tensor(
            word_word_edge_attr, dtype=torch.float32)
        if use_general_node:
            data['general', 'general_word', 'word'].edge_index = torch.transpose(torch.tensor(
                general_word_edge_index, dtype=torch.int32), 0, 1) if len(general_word_edge_index) > 0 else torch.empty(2, 0, dtype=torch.int32)
            data['word', 'word_general', 'general'].edge_index = torch.transpose(torch.tensor(
                word_general_edge_index, dtype=torch.int32), 0, 1) if len(word_general_edge_index) > 0 else torch.empty(2, 0, dtype=torch.int32)
            data['general', 'general_word', 'word'].edge_attr = torch.tensor(
                general_word_edge_attr, dtype=torch.float32)
            data['word', 'word_general', 'general'].edge_attr = torch.tensor(
                word_general_edge_attr, dtype=torch.float32)
        return data

    def draw_graph(self, idx: int):
        # TODO : do this part if needed
        pass

    def to_graph_indexed(self, text: str):
        doc_sentences = []
        doc = []
        doc.append(doc_sentences)
        token_list = self.token_lemma(str(text))
        for idx, sentence in enumerate(token_list.sentences):
            doc_sentences.append((sentence.text, sentence.tokens[0].text, idx))
            for word in sentence.words:
                doc.append((idx, word.text, word.lemma,
                            word.upos, word.head, word.deprel))
        # if len(doc) < 2:
        #     return
        if self.use_sentence_nodes:
            return self.__create_graph_with_sentences(doc, for_compression=True)
        else:
            return self.__create_graph(doc, for_compression=True, use_general_node=self.use_general_node)

    def get_word_by_id(self):
        words_id = {}
        for word in self.nlp.get_words():
            words_id[self.nlp.get_word_id(word)] = word
        return words_id

    def prepare_loaded_data(self, graph):
        words = torch.zeros(
            (len(graph['word'].x), self.nlp.get_dimension()), dtype=torch.float32)

        for i in range(len(graph['word'].x)):
            if self.word_ids.get(int(graph['word'].x[i])) is not None:
                words[i] = torch.tensor(
                    self.nlp.get_word_vector(self.word_ids[int(graph['word'].x[i])]))
        graph['word'].x = words
        graph['dep'].x = self.__build_initial_dependency_vectors(
            len(self.dependencies))
        graph['tag'].x = self.__build_initial_tag_vectors(len(self.tags))
        if self.use_general_node:
            graph = self._add_multiple_general_nodes(
                graph, self.use_sentence_nodes, self.num_general_nodes)
        for t in graph.edge_types:
            if len(graph[t].edge_index) == 0:
                graph[i].edge_index = torch.empty(2, 0, dtype=torch.int32)
        return graph

    def _add_multiple_general_nodes(self, graph, use_sentence_nodes, num_general_nodes):
        if not use_sentence_nodes:
            graph['general'].x = self.__build_initial_general_vector(
                num=self.num_general_nodes)
            if self.num_general_nodes > 1:
                # connecting other general nodes
                general_word_edge_index = torch.transpose(torch.tensor(
                    graph['general', 'general_word', 'word'].edge_index, dtype=torch.int32), 0, 1).tolist()
                word_general_edge_index = torch.transpose(torch.tensor(
                    graph['word', 'word_general', 'general'].edge_index, dtype=torch.int32), 0, 1).tolist()
                general_word_edge_attr = graph['general',
                                               'general_word', 'word'].edge_attr.tolist()
                word_general_edge_attr = graph['word',
                                               'word_general', 'general'].edge_attr.tolist()
                for j in range(1, num_general_nodes):
                    for i in range(len(graph['word'].x)):
                        word_general_edge_index.append([i, j])
                        general_word_edge_index.append([j, i])
                        word_general_edge_attr.append(
                            self.settings["general_token_weight"])
                        general_word_edge_attr.append(
                            self.settings["general_token_weight"])
                graph['general', 'general_word', 'word'].edge_index = torch.transpose(
                    torch.tensor(general_word_edge_index, dtype=torch.int32), 0, 1)
                graph['word', 'word_general', 'general'].edge_index = torch.transpose(
                    torch.tensor(word_general_edge_index, dtype=torch.int32), 0, 1)
                graph['general', 'general_word', 'word'].edge_attr = torch.tensor(
                    general_word_edge_attr, dtype=torch.float32)
                graph['word', 'word_general', 'general'].edge_attr = torch.tensor(
                    word_general_edge_attr, dtype=torch.float32)
        else:
            graph['general'].x = self.__build_initial_general_vector(
                num=self.num_general_nodes)
            if self.num_general_nodes > 1:
                # connecting other general nodes
                general_sentence_edge_index = torch.transpose(torch.tensor(
                    graph['general', 'general_sentence', 'sentence'].edge_index, dtype=torch.int32), 0, 1).tolist()
                sentence_general_edge_index = torch.transpose(torch.tensor(
                    graph['sentence', 'sentence_general', 'general'].edge_index, dtype=torch.int32), 0, 1).tolist()
                general_sentence_edge_attr = graph['general',
                                                   'general_sentence', 'sentence'].edge_attr.tolist()
                sentence_general_edge_attr = graph['sentence',
                                                   'sentence_general', 'general'].edge_attr.tolist()
                for j in range(1, num_general_nodes):
                    for i in range(len(graph['sentence'].x)):
                        sentence_general_edge_index.append([i, j])
                        general_sentence_edge_index.append([j, i])
                        sentence_general_edge_attr.append(
                            self.settings["general_sentence_weight"])
                        general_sentence_edge_attr.append(
                            self.settings["general_sentence_weight"])
                graph['general', 'general_sentence', 'sentence'].edge_index = torch.transpose(
                    torch.tensor(general_sentence_edge_index, dtype=torch.int32), 0, 1)
                graph['sentence', 'sentence_general', 'general'].edge_index = torch.transpose(
                    torch.tensor(sentence_general_edge_index, dtype=torch.int32), 0, 1)
                graph['general', 'general_sentence', 'sentence'].edge_attr = torch.tensor(
                    general_sentence_edge_attr, dtype=torch.float32)
                graph['sentence', 'sentence_general', 'general'].edge_attr = torch.tensor(
                    sentence_general_edge_attr, dtype=torch.float32)
        return graph
