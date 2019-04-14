# Relation Discovery with Out-of-Relation Knowledge Base as Supervision

This code is based on the code for paper [Discrete-State Variational Autoencoders for Joint Discovery and Factorization of Relations](https://transacl.org/ojs/index.php/tacl/article/viewFile/761/190) by Diego Marcheggiani and Ivan Titov.

## Training Models

Create a python 3.6 environment using pyenv or conda. Then install python packages with pip:
`pip install -r requirements.txt`

Extract data to `./data` directory.

- Baseline model: `python -m main oie`
- RegDVAE: 
  - First get the KB embeddings: `python -m main ext_kb -out model/ext_kb/m_001`
  - `python -m main oie_reg -lekd m001`

## Text Features

 - lexicalized dependency path between arguments (entities) of the relation,
 - first entity
 - second entity
 - entity types of the first and second entity
 - trigger word
 - id of the sentence
 - raw sentence
 - pos tags of the entire sentence
 - relation between the two entities if any (used only for evaluation)

## Dependencies
- [python3.6](https://www.python.org)
- [numpy>=1.4](http://http://www.numpy.org/)
- [scipy](http://https://www.scipy.org/)
- [nltk](http://http://www.nltk.org/)
- [tensorflow>=r1.7](https://www.tensorflow.org/)
- [recordclass](https://bitbucket.org/intellimath/recordclass)
- [pandas](https://pandas.pydata.org/)
