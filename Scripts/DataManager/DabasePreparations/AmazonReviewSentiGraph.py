import pandas as pd

from Scripts.Configs.ConfigClass import Config
from os import path

from Scripts.DataManager.GraphConstructor.CoOccurrenceGraphConstructor import CoOccurrenceGraphConstructor
from Scripts.DataManager.GraphConstructor.GraphConstructor import TextGraphType


class AmazonReviewSentiGraph:

    def __init__(self, config: Config, graph_type: TextGraphType = TextGraphType.CO_OCCURRENCE):
        self.config = config
        train_df = pd.read_csv(path.join(config.data_root_dir, r'Amazon-Review\train_sm.csv'))
        test_df = pd.read_csv(path.join(config.data_root_dir, r'Amazon-Review\test_sm.csv'))
        train_df.columns = ['Polarity', 'Title', 'Review']
        test_df.columns = ['Polarity', 'Title', 'Review']
        self.train_df = train_df[['Polarity', 'Review']]
        self.test_df = test_df[['Polarity', 'Review']]
        self.graph_type = graph_type
        self.graph_constructors = self.__set_graph_constructors(graph_type)

    def __set_graph_constructors(self, graph_type: TextGraphType):
        graph_constructors = {}
        if TextGraphType.CO_OCCURRENCE in graph_type:
            graph_constructors[TextGraphType.CO_OCCURRENCE] = self.__get_co_occurrence_graph()
        if TextGraphType.DEPENDENCY in graph_type:
            pass
        if TextGraphType.SEQUENTIAL in graph_type:
            pass
        if TextGraphType.TAGS in graph_type:
            pass
        return graph_constructors

    def __get_co_occurrence_graph(self):
        return CoOccurrenceGraphConstructor(self.train_df['Review'], 'AmazonReview', self.config,
                                         lazy_construction=True,
                                         load_preprocessed_data=True, naming_prepend='graph')

