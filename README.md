
# Literature Mining for Neurodegenerative disease
[report.pdf](https://github.com/jacons/LMining-Neurodegenerative/files/11532442/report.pdf)

[presentation.pdf](https://github.com/jacons/LMining-Neurodegenerative/files/11525248/presentation.pdf)


This study aims
to propose and evaluate our framework that integrates multiple ML and IR technique for extracting Disease-Drug pathways
(DDAs) from the literature in a specific domain.
The pipeline incorporates algorithms such as BERN, (Node-rank)Page-Rank and Dijkstra.
Throughout the article, we delve into the retrieval of relevant documents,
the processing of gathered information to acquire knowledge,
the discovery or consolidation of existing pathways and finally assess the result with a benchmark database.

The Literature mining (LM) is the process of extracting useful information and knowledge from large collections
of textual data such as scientific articles, patents and clinical reports.
It involves using various techniques
from Natural Language Processing (NLP), Machine Learning (ML), Data Mining (DM) and Information
Retrieval (IR) to analyze and interpret the text and identify patterns and relationships.
The applications
of LM are vast and include drug discovery, personalized medicine, clinical decision support, and scientific
discovery by leveraging the power of NLP and ML.
LM enables researchers to extract knowledge from large
collections of textual data that would otherwise be impossible to process manually.
For example, we can cite
a large database as Scopus, Google Scholar or PubMed.
The latter, e.g., claims to have more than 35 million 
documents available, and thanks to the large amount of data and the ease of retrieving information, Pubmeb
is ideal as a tool of information retrieval for our objective.
The aim of this study is to implement a custom framework1 that starts from the "retrieving of information"
and ends up with the analysis of Knowledge graph (KG) 2. Our pipeline encloses algorithms like BERN,
Node-Rank (based on Page-Rank) and Dijkstra.
Throughout the article, we will explore in detail how we retrieve
the documents of interest, how the information gathered is processed to acquire knowledge and how to
discover new pathways3 or consolidate those already existing.
The report is structured in the following way.
In chapter 2 we depict one of the most popular tools of NLP
used in the literature mining i.e., Named Entity Recognition (NER), we illustrate basically how it works and
because it is so important for our purpose.
In chapter 3 we explain in detail our implementation based on the
composition of BERN, Node-Rank and Dijkstra algorithms.
In chapter 4, we show our experimental results
using a benchmark database.
In the end, on chapter 5 we draw conclusions, and we point out the several
limitations that we encountered.


## Authors
- [@jacons](https://www.github.com/jacons)
- [@GeremiaPompei](https://github.com/GeremiaPompei)


