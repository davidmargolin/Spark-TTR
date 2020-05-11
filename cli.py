import os
import click
from pyspark import SparkContext
from dotenv import load_dotenv
load_dotenv()

sc = SparkContext(master=os.getenv("SPARK_HOST"), appName="First App")

@click.group()
def main():
    """
    CLI for Project 2
    """
    pass

@main.command()
@click.argument('file')
def term_term_relevance(file):
    """
    Outputs a list of term-term pairs sorted by their similarity descending
    """
    lines=sc.textFile(file)
    cnt = lines.count()
    click.echo("Loaded {} lines".format(cnt))

if __name__ == "__main__":
    main()