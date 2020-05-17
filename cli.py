import os
import re
import click
import math
from pyspark import SparkContext
from dotenv import load_dotenv
load_dotenv()

sc = SparkContext(master=os.getenv("SPARK_HOST"), appName="Term Term Relevance")


@click.command()
@click.argument('file', type=click.Path(exists=True))
@click.argument('query')
def term_term_relevance(file, query):
    """
    Outputs a list of term-term pairs sorted by their similarity descending
    """

    # comments use sample.txt input

    # "docId1 w1 w2 w3 w1" 
    # "docId2 w1 w4 w3"
    lines=sc.textFile(file)

    # 2
    cnt = lines.count()
    click.echo("Loaded {} documents. Generating matrix.".format(cnt))

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
        filteredWords = list(filter(lambda x: (re.match(r"gene_.*_gene", x) != None), words))
        return list(map(lambda word: (word, {documentId: 1/wordCount}), filteredWords))
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


    click.echo("Matrix completed. Generating TTR for {}...".format(query))

    # Sub question 2 starts here:

    # {"docId1": 2/4*log(2/2), "docId2": 1/3*log(2/2)}  #if queryTerm == "w1"
    # return the query term row from a tfidf matrix
    try:
        target_stripes_dict = tfidf.filter(lambda x:x[0]==query).values().first()
    except:
        click.echo("Error: term does not exist in document")
        return

    # sqrt(docId1^2 + docId2^2)
    # helper function:
    # input a dict obj, and return the sqrt of sum of all values' square
    def return_sqrt(dict):
        res = 0
        for item in dict:
            res += math.pow(dict[item], 2) 
        return math.sqrt(res)
    lsqrt = return_sqrt(target_stripes_dict)

    nonTermDocs = tfidf.filter(lambda x:x[0]!=query)

    # calculate the cosine similiarity
    def calculate_similarity(input):
        word = input[0]
        stripes = input[1]
        sumOfProduct = 0
        for key in stripes:
            if key in target_stripes_dict:
                sumOfProduct += stripes[key] * target_stripes_dict[key]
        rsqrt = return_sqrt(stripes)
        similarity = sumOfProduct / (lsqrt * rsqrt)
        return (similarity, word)
    similarity_matrix = nonTermDocs.map(calculate_similarity)

    similarity_matrix_desc = similarity_matrix.sortByKey(ascending=False).values().take(5)
    [click.echo(str(index+1) + ". " + word) for (index, word) in enumerate(similarity_matrix_desc)]


if __name__ == "__main__":
    term_term_relevance()