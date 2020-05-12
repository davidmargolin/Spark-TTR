import os
import click
import math
from pyspark import SparkContext
from dotenv import load_dotenv
load_dotenv()

sc = SparkContext(master=os.getenv("SPARK_HOST"), appName="Term Term Relevance")

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

    # comments use sample.txt input
    
    # "docId1 w1 w2 w3 w1" 
    # "docId2 w1 w4 w3"
    lines=sc.textFile(file)

    # 2
    cnt = lines.count()
    click.echo("Loaded {} documents".format(cnt))

    # (docId1, [w1, w2, w3, w1]) 
    # (docId2, [w1, w4, w3])
    documentData=lines.map(lambda x: x.split()).map(lambda x:(x[0],x[1:]))

    # (w1, {"docId1": 1/4}) 
    # (w2, {"docId1": 1/4}) 
    # (w3, {"docId1": 1/4})
    # (w1, {"docId1": 1/4})
    # (w1, {"docId2": 1/3})
    # (w4, {"docId2": 1/3})
    # (w3, {"docId2": 1/3})
    def term_document_count(document):
        documentId = document[0]
        words = document[1]
        wordCount = len(words)
        return list(map(lambda word: (word, {documentId: 1/wordCount}), words))
    matrix=documentData.flatMap(term_document_count)

    # (w1, {"docId1": 2/4, "docId2": 1/3}) 
    # (w2, {"docId1": 1/4}) 
    # (w3, {"docId1": 1/4, "docId2": 1/3}) 
    # (w4, {"docId2": 1/3})
    def combine_counts(aStripe, bStripe):
        for key in aStripe:
            if key in bStripe:
                bStripe[key]+=aStripe[key]
            else:
                bStripe[key]=aStripe[key]
        return bStripe
    tf=matrix.reduceByKey(combine_counts)
    
    # (w1, {"docId1": 2/4*log(2/2), "docId2": 1/3*log(2/2)}) 
    # (w2, {"docId1": 1/4*log(2/1)}) 
    # (w3, {"docId1": 1/4*log(2/2), "docId2": 1/3*log(2/2)}) 
    # (w4, {"docId2": 1/3*log(2/1)})
    def multiply_idf(input):
        word = input[0]
        stripes = input[1]
        multiplier = math.log10(cnt/len(stripes))
        for doc in stripes:
            stripes[doc] *= multiplier
        return (word, stripes)
    tfidf = tf.map(multiply_idf)

    tfidf.saveAsTextFile("tfidf_matrix")

if __name__ == "__main__":
    main()