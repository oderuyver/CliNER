===============================
CliNER
===============================

Clinical Named Entity Recognition system (CliNER) is an open-source natural language processing system for named entity recognition in clinical text of electronic health records.  CliNER system is designed to follow best practices in clinical concept extraction, as established in i2b2 2010 shared task.

CliNER is implemented as a sequence classification task, where every token is predicted IOB-style as either: Problem, Test, Treatment, or None. Coomand line flags let you specify two different sequence classification algorithms:
    1. CRF (default) - with linguistic and domain-specific features
    2. LSTM - [Upcoming: with the option of specifying pretrained word embeddings]

Please note that for optimal performance, CliNER requires the users to obtain a Unified Medical Language System (UMLS) license, since UMLS Metathesaurus is used as one of the knowledge sources for the above classifiers.


* Free software: Apache v2.0 license
* Documentation: http://cliner.readthedocs.org.



Installation
--------


        pip install -r requirements.txt

        wget http://text-machine.cs.uml.edu/cliner/samples/doc_1.txt

        python cliner/galen predict --txt doc_1.txt --out data/predictions --model models/word-lstm.galen  --format i2b2



Optional
--------

There are a few external resources that are not packaged with CliNER but can improve prediction performance for feature extraction with the CRF.

    GENIA

        The GENIA tagger identifies named entities in biomedical text.

        > wget http://www.nactem.ac.uk/tsujii/GENIA/tagger/geniatagger-3.0.2.tar.gz
        > tar xzvf geniatagger-3.0.2.tar.gz
        > cd geniatagger-3.0.2
        > make

        Edit config.txt so that GENIA references the geniatagger executable just built. (e.g. "GENIA   /someuser/CliNER/geniatagger-3.0.2/geniatagger")

    UMLS

        > TODO



Getting Data
--------

The Data Use and Confidentiality Agreement (DUA) with i2b2 forbids us from redistributing the i2b2 data. In order to gain access to the data, you must go to:

https://www.i2b2.org/NLP/DataSets/AgreementAR.php

to register and sign the DUA. Then you will be able to request the data through them.




