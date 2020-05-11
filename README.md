# Prerequesites

Python (<= 3.7)
[Spark 2.4.5](https://www.apache.org/dyn/closer.lua/spark/spark-2.4.5/spark-2.4.5-bin-hadoop2.7.tgz)

# Installation

1. Create a virtual environment
```bash
python3 -m venv env
```

2. Activate the virtual environment
```bash
source env/bin/activate
```

3. Install dependencies
```bash
pip3 install -r requirements.txt
```

4. Set env vars
```bash
export SPARK_HOST=<spark host url>
```

# Running the CLI
```bash
python3 cli.py term-term-relevance <file-path>
```