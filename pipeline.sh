# pipeline.sh
#!/bin/bash

python3 datacreation.py
python3 modelpreprocessing.py
python3 modelpreparation.py
python3 modeltesting.py