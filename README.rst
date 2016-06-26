===============================
CliNER
===============================

Clinical Named Entity Recognition system (CliNER) is an open-source natural language processing system for named entity recognition in clinical text of electronic health records.  CliNER system is designed to follow best practices in clinical concept extraction, as established in i2b2 2010 shared task.

This project is a ground-up rewrite of cliner using a simpler structure and modern Deep Learning tools. The rewrite has been named "galen" to avoid confusion with versions named "CliNER". For the original cliner tool, see https://github.com/text-machine-lab/CliNER




Installation
--------


        pip install -r requirements.txt

        cp -r lib/keras $(python -c "import numpy, re; print (re.search('''from '(.*)/numpy/__init__.pyc''', str(numpy)).groups()[0])")

        wget http://text-machine.cs.uml.edu/cliner/samples/doc_1.txt

        python cliner/galen predict --txt doc_1.txt --out data/predictions --model models/word-lstm.galen  --format i2b2




Getting Data
--------

The Data Use and Confidentiality Agreement (DUA) with i2b2 forbids us from redistributing the i2b2 data. In order to gain access to the data, you must go to:

https://www.i2b2.org/NLP/DataSets/AgreementAR.php

to register and sign the DUA. Then you will be able to request the data through them.




