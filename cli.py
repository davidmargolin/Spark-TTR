import os
import click
import math
from pyspark import SparkContext
from dotenv import load_dotenv
load_dotenv()

sc = SparkContext(master=os.getenv("SPARK_HOST"), appName="Term Term Relevance")


@click.command()
@click.argument('file', type=click.Path(exists=True))
#@click.argument('queryTerm', nargs=1)
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
    #print(lines.take(2))
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

    # Sub question 2 starts here:
    queryTerm = input("Please enter a query term:")

    # (w1, {"docId1": 2/4*log(2/2), "docId2": 1/3*log(2/2)})    #if queryTerm == "w1"
    # helper function: return the query term row from a tfidf matrix
    def return_term(input):
        word = input[0]
        stripes = input[1]
        if word == queryTerm:
            return stripes
    tfidf_filtered = tfidf.filter(return_term)
    

    # {"docId1": 2/4*log(2/2), "docId2": 1/3*log(2/2)}      
    def return_stripes(input):
        word = input[0]
        stripes = input[1]
        return stripes
    target_stripes = tfidf_filtered.map(return_stripes)
    target_stripes_list = target_stripes.take(1)
    target_stripes_dict = target_stripes_list[0]


    # sqrt(docId1^2 + docId2^2)
    # helper function:
    # input a dict obj, and return the sqrt of sum of all values' square
    def return_sqrt(dict):
        res = 0
        for item in dict:
            sq = dict[item] * dict[item] 
            res += sq
        return math.sqrt(res)
    lsqrt = return_sqrt(target_stripes_dict)

    
    # calculate the cosine similiarity
    def calculate_similarity(input):
        word = input[0]
        stripes = input[1]
        sumOfProduct = 0
        stripesDict = {}
        for key in stripes:
            if key in target_stripes_dict.keys():
                product = stripes[key] * target_stripes_dict[key]
                sumOfProduct += product
                stripesDict[key] = stripes[key]
            else:
                sumOfProduct += 0
                stripesDict[key] = stripes[key]
        rsqrt = return_sqrt(stripesDict)
        similarity = sumOfProduct / (lsqrt * rsqrt)
        return (word, queryTerm,similarity)
    similarity_matrix = tfidf.map(calculate_similarity)
    
    # formatting the output in descending order
    similarity_matrix_desc = similarity_matrix.sortBy(lambda x: x[2],False)
    similarity_matrix_desc.saveAsTextFile("similarity_matrix_desc")


if __name__ == "__main__":
    term_term_relevance()