from pyspark.mllib.clustering import KMeans
from numpy import array
from math import sqrt
from pyspark import SparkContext
import sys
from numpy import sum
from scipy.spatial.distance import euclidean

def f(x):print x
if len(sys.argv) != 4:
        print >> sys.stderr, "Usage: clustring.py <input_file> <output file1> <output file2>"
        exit(-1)
sc=SparkContext(appName="stocks_clustring")
data = sc.textFile(sys.argv[1])
parsedData = data.map(lambda line: line.strip().split(',')).map(lambda (k,line) : (k,array([float(x) for x in line.split(' ')])))
data2 = parsedData.map(lambda (k,v): v)
clusters = KMeans.train(data2, 10, maxIterations=10,runs=10, initializationMode="random")
distance = data2.map(lambda point :(clusters.predict(point),euclidean(point,clusters.centers[clusters.predict(point)])))
radius = distance.reduceByKey(max)
radius =  radius.sortByKey(1)
count = data2.map(lambda point :(clusters.predict(point),1)).reduceByKey(lambda x,y : x+y)
group = radius.join(count)
densities = group.map(lambda (cluster, (radius, number)): (cluster, number / ((radius)**2))).map(lambda (a, b): (b, a)).sortByKey(1, 1).map(lambda (a, b): (b, a))
low = densities.first()
high =densities.map(lambda (a,b):(b,a)).sortByKey(0,1).map(lambda (a,b):(b,a)).first()
final = parsedData.map(lambda (k,v): (k,clusters.predict(v)))
low_data = final.filter(lambda (lable,cluster): cluster == low[0])
high_data = final.filter(lambda(lable,cluster): cluster == high[0])
print "stocks in low density cluster"
low_data.foreach(f)
print "stocks in high density cluster"
high_data.foreach(f)
low_data.repartition(1).saveAsTextFile(sys.argv[2])
high_data.repartition(1).saveAsTextFile(sys.argv[3])