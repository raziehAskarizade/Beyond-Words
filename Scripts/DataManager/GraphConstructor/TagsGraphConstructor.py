
import pandas as pd
from Scripts.DataManager.GraphConstructor.GraphConstructor import GraphConstructor
from torch_geometric.data import Data
from Scripts.Configs.ConfigClass import Config
import spacy
import torch
import numpy as np


class TagsGraphConstructor(GraphConstructor):
    
    class _Variables(GraphConstructor._Variables):
        def __init__(self):
            super(TagsGraphConstructor._Variables, self).__init__()
            self.nlp_pipeline: str = ''
    def __init__(self, texts: List[str], save_path: str, config: Config,
                 lazy_construction=True, load_preprocessed_data=False, naming_prepend=''):

        super(TagsGraphConstructor, self)\
            .__init__(texts, self._Variables(), save_path, config, lazy_construction, load_preprocessed_data,
                      naming_prepend)
        self.tags = self.nlp.get_pipe("tagger").labels
        self.settings = {"tokens_tag_weight" : 1, "token_token_weight" : 2}
        if self.load_preprocessed_data:
            if not self.lazy_construction:
                self.load_all_data()
            else:
                self.load_var()
        else:
            self.var.nlp_pipeline = self.config.spacy.pipeline
            self.var.graph_num = len(self.raw_data)
            self.nlp = spacy.load(self.var.nlp_pipeline)

            if not self.lazy_construction:
                for i in range(len(self.raw_data)):
                    if i not in self._graphs:
                        if i % 100 == 0:
                            print(f'i: {i}')
                        self._graphs[i] = self.to_graph(self.raw_data[i])
                        self.var.graphs_name[i] = f'{self.naming_prepend}_{i}'
            self.save_all_data()

    def to_graph(self, text: str):
        doc = self.nlp(text)
        if len(doc) < 2:
            return
        return self.__create_graph(doc)
    def __find_tag_index(self , tag : str):
        for tag_idx in range(len(self.tags)):
            if self.tags[tag_idx] == tag:
                return tag_idx
        return -1 # means not found
    def __create_graph(self , doc):
        tags_length = len(self.tags)
        node_attr = torch.zeros((len(doc) + tags_length, self.nlp.vocab.vectors_length), dtype=torch.float32)
        edge_index = []
        edge_attr = []
        
        # for idx in range(tags_length):
            # if vevtorizing of dependencies is needed, do it here
            # node_attr[idx] = sth ...
        for token in doc:
            token_id = self.nlp.vocab.strings[token.lemma_]
            if token_id in self.nlp.vocab.vectors:
                node_attr[token.i + tags_length - 1] = torch.tensor(self.nlp.vocab.vectors[token_id])
            tag_idx = self.__find_tag_index(token.tag_)
            if tag_idx != -1:
                edge_index.append([token.i + tags_length - 1 , tag_idx])
                edge_attr.append(self.settings["tokens_tag_weight"])
            # adding sequential edges between tokens - uncomment the codes for vectorized edges
            if token.i != len(doc):
                # using zero vectors for edge features
                edge_index.append([token.i + tags_length - 1 , token.i + 1])
                # self.edge_attr.append(torch.zeros((self.nlp.vocab.vectors_length,), dtype=torch.float32))
                edge_attr.append(self.settings["token_token_weight"])
        # self.node_attr = node_attr
        edge_index = torch.transpose(torch.tensor(edge_index, dtype=torch.long) , 0 , 1)
        # self.edge_attr = edge_attr # vectorized edge attributes
        return Data(x=node_attr, edge_index=edge_index,edge_attr=edge_attr)
    def draw_graph(self , idx : int):
        node_tokens = []
        doc = self.nlp(self.raw_data[idx])
        for d in self.tags:
            node_tokens.append(d)
        for t in doc:
            node_tokens.append(t.lemma_)
        graph_data = self.get_graph(idx)
        g = to_networkx(graph_data)
        layout = nx.spring_layout(g)
        nx.draw(g, pos=layout)
        words_dict = {i: node_tokens[i] for i in range(len(node_tokens))}
        # edge_labels_dict = {(graph_data.edge_index[0][i].item() , graph_data.edge_index[1][i].item()) : { "dep" : graph_data.edge_attr[i]} for i in range(len(graph_data.edge_attr))}
        # nx.set_edge_attributes(g , edge_labels_dict)
        nx.draw_networkx_labels(g, pos=layout, labels=words_dict)
        # nx.draw_networkx_edge_labels(g, pos=layout)

        

