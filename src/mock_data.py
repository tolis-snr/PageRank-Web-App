from src.models import NodeBase, EdgeBase, GraphBase

mock_nodes = [
    NodeBase(id="1", title="Attention Is All You Need", abstract="We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely."),
    NodeBase(id="2", title="BERT: Pre-training of Deep Bidirectional Transformers", abstract="We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers."),
    NodeBase(id="3", title="The PageRank Citation Ranking: Bringing Order to the Web", abstract="The importance of a Web page is an inherently subjective matter, which depends on the readers interests, knowledge and attitudes. But there is still much that can be said objectively about the relative importance of Web pages. This paper describes PageRank, a method for rating Web pages objectively and mechanically, effectively measuring the human interest and attention devoted to them."),
    NodeBase(id="4", title="Graph Neural Networks: A Review of Methods and Applications", abstract="Graphs are a kind of data structure which models a set of objects (nodes) and their relationships (edges). Recently, researchers have generalized deep learning methods to graph data."),
    NodeBase(id="5", title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", abstract="Large pre-trained language models have been shown to store factual knowledge in their parameters, and achieve state-of-the-art results when fine-tuned on downstream NLP tasks. However, their ability to access and precisely manipulate knowledge is still limited. Here, we build RAG models where the parametric memory is a pre-trained seq2seq model and the non-parametric memory is a dense vector index of Wikipedia."),
    NodeBase(id="6", title="Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks", abstract="We present Sentence-BERT (SBERT), a modification of the pretrained BERT network that use siamese and triplet network structures to derive semantically meaningful sentence embeddings that can be compared using cosine-similarity."),
    NodeBase(id="7", title="Node2Vec: Scalable Feature Learning for Networks", abstract="We propose node2vec, an algorithmic framework for learning continuous feature representations for nodes in networks."),
    NodeBase(id="8", title="Graph RAG: Graph based Retrieval Augmented Generation", abstract="A novel approach to improve the context provided to Large Language Models. By utilizing Graph structures alongside traditional vector search, we can retrieve highly relevant structural information to answer complex queries.")
]

mock_edges = [
    EdgeBase(source="2", target="1"), # BERT cites Attention
    EdgeBase(source="5", target="2"), # RAG cites BERT
    EdgeBase(source="6", target="2"), # SBERT cites BERT
    EdgeBase(source="4", target="7"), # GNN review cites Node2Vec
    EdgeBase(source="8", target="5"), # Graph RAG cites RAG
    EdgeBase(source="8", target="4"), # Graph RAG cites GNN
    EdgeBase(source="8", target="3"), # Graph RAG cites PageRank
]

MOCK_GRAPH = GraphBase(nodes=mock_nodes, edges=mock_edges)
