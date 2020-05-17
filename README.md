# Prerequesites

Python (<= 3.7)

[Spark 2.4.5](https://www.apache.org/dyn/closer.lua/spark/spark-2.4.5/spark-2.4.5-bin-hadoop2.7.tgz)

Java8 (won't work with Java11)

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
export PYSPARK_DRIVER_PYTHON=<python path>
export PYSPARK_PYTHON=<python path>
```
5. Start Master/Slave Server
```bash
start-master.sh
```
To view the Spark Web UI, open a web browser and enter the localhost IP address on port 8080
```bash
http://127.0.0.1:8080/
```
Start a slave server
```bash
start-slave.sh <spark://master:port>
```
# Running the CLI
```bash
python3 cli.py <file-path>
```
