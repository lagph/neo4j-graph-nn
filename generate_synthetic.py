import torch
from torch.distributions import Categorical, Bernoulli, Normal
import names
import yaml
import random
from graph_dbs import Neo4jGraph


PATH = 'config/books.yaml'

config = yaml.load(open(PATH), Loader=yaml.FullLoader)

NUM_COUNTRIES = len(config['countries'])
NUM_TOPICS = len(config['topics'])
NUM_AUTHORS = config['num-authors']
AUTHOR_IDS = list(range(NUM_AUTHORS))
random.shuffle(AUTHOR_IDS)


def generate_author_graph(config, topic_popularities, country_popularities):

    topic_popularities = torch.tensor(topic_popularities)

    country_popularities = torch.tensor(country_popularities)

    author_country_id = Categorical(
        logits=torch.zeros(NUM_COUNTRIES)).sample().item()

    author_main_topic_id = Categorical(
        logits=torch.zeros(NUM_TOPICS)).sample().item()

    off_topic_logits = torch.zeros(NUM_TOPICS)

    off_topic_logits[author_main_topic_id] = -1000  # sets prob -> 0

    author_off_topic_id = Categorical(logits=off_topic_logits).sample().item()

    author_base_popularity = country_popularities[author_country_id]

    author_popularity = Normal(
        author_base_popularity, config['author-popularity-std']).sample().item()

    num_books = Normal(config['num-books-mean'],
                       config['num-books-std']).sample().floor().int()

    is_on_topic = Bernoulli(
        probs=torch.empty(num_books.item()).fill_
        (1 - config['off-topic-probability'])).sample().byte()

    topics = torch.where(is_on_topic, torch.empty(num_books.item(),
                                                  dtype=torch.long).fill_(author_main_topic_id),
                         torch.empty(num_books.item(), dtype=torch.long).fill_(
        author_off_topic_id))

    popularities = (topic_popularities[topics] + author_popularity)

    popularities[~ is_on_topic] /= config['off-topic-effect-size']

    sales = torch.max(Normal(
        popularities, config['sales-std']).sample(),
        torch.zeros_like(popularities))

    return author_country_id, topics, sales, popularities


def generate_graph_dicts(config):
    topic_popularities = [config['topics'][i]['popularity']
                          for i in range(NUM_TOPICS)]

    country_popularities = [config['countries'][i]['popularity']
                            for i in range(NUM_COUNTRIES)]

    graph_dicts = []
    for i in range(config['num-authors']):
        data = {'author': {'name': names.get_full_name(), 'id': AUTHOR_IDS[i]}}

        country, topics, sales, expected_sales = generate_author_graph(config,
                                                                       topic_popularities,
                                                                       country_popularities)

        data['author']['country'] = config['countries'][country]['name']

        data['books'] = [{'topic': config['topics'][t.item()]['name'],
                          'sales': sales[i].item(),
                          'expected_sales': expected_sales[i].item(),
                          }
                         for i, t in enumerate(topics)]
        graph_dicts.append(data)

    return graph_dicts


def poupulate_knowledge_graph(graph_dicts, graph):
    for graph_dict in graph_dicts:
        graph_dict['author']['kind'] = 'Author'
        for i, book in enumerate(graph_dict['books']):
            book['kind'] = 'Book'
            book['id'] = i
            graph.commit_relation(src=graph_dict['author'], rel={
                                  'kind': 'WROTE'}, dst=book)


def main():
    graph = Neo4jGraph("bolt://localhost:7687", auth=("neo4j", "test"))

    gd = generate_graph_dicts(config)

    poupulate_knowledge_graph(gd, graph)


if __name__ == "__main__":
    main()
