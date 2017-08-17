===============================
CliNER
===============================

Clinical Named Entity Recognition system (CliNER) is an open-source natural language processing system for named entity recognition in clinical text of electronic health records.  CliNER system is designed to follow best practices in clinical concept extraction, as established in i2b2 2010 shared task.

CliNER is implemented as a sequence classification task, where every token is predicted IOB-style as either: Problem, Test, Treatment, or None. Coomand line flags let you specify two different sequence classification algorithms:
    1. CRF (default) - with linguistic and domain-specific features
    2. LSTM

Please note that for optimal performance, CliNER requires the users to obtain a Unified Medical Language System (UMLS) license, since UMLS Metathesaurus is used as one of the knowledge sources for the above classifiers.


* Free software: Apache v2.0 license
* Documentation: http://cliner.readthedocs.org.



Installation
--------


        pip install -r requirements.txt

        wget http://text-machine.cs.uml.edu/cliner/samples/doc_1.txt

        wget http://text-machine.cs.uml.edu/cliner/models/silver.model -o models/silver.model

        python cliner predict --txt doc_1.txt --out data/predictions --model models/silver.model  --format i2b2



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




Notes
--------

The cliner pipeline assumes that the clinical text has been preprocessed to be tokenized, as in accordance with the i2b2 format. I have included a simple tokenization script (see: `tools/tok.py`) that you can use or modify as you wish.

The silver model does come with some degradation of performance. Given that the alternative is no model, I think this is okay, but be aware that if you have the i2b2 training data, then you can build a model that performs even better on the i2b2 test data.


Original Model (trained on i2b2-train data with UMLS + GENIA feats)

    TESTING 1.1 -  Exact span for all concepts together
                         TP    FN    FP   Recall Precision F1
    Class Exact Span -> 23358 4904  7696  0.826  0.752     0.788

    TESTING 1.2 -  Exact span for separate concept classes
                                                      TP    FN    FP   Recall   Precision  F1
    Exact Span With Matching Class for Problem   ->  9478  2291  3077  0.805    0.755      0.779
    Exact Span With Matching Class for Treatment ->  6881  1402  2398  0.831    0.742      0.784
    Exact Span With Matching Class for Test      ->  6999  1211  2221  0.852    0.759      0.803


Silver Model (trained on mimic data that was annotated by Original Model)

    TESTING 1.1 -  Exact span for all concepts together
                         TP    FN    FP    Recall Precision F1
    Class Exact Span -> 20771 5504  10283  0.791  0.669     0.725

    TESTING 1.2 -  Exact span for separate concept classes
                                                     TP    FN    FP   Recall  Precision  F1
    Exact Span With Matching Class for Problem   -> 8735  2875  3820  0.752   0.696      0.7229464100972481
    Exact Span With Matching Class for Treatment -> 5961  1278  3318  0.823   0.642      0.721758082092263
    Exact Span With Matching Class for Test      -> 6075  1351  3145  0.818   0.659      0.7299050823020545
